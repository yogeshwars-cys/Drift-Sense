"""
Hypothesis graph: the decision stage.

The re-ranker used to be a judge -- a CNN embedding compared each candidate
patch to the reference in isolation, and the highest cosine won. That framing
has two structural problems in a periodic array:

  * Every candidate is scored independently, so the fact that candidates are
    mutually exclusive alternatives about ONE wafer is never used.
  * Appearance similarity is dominated by the periodic lattice, which is by
    construction identical at every candidate, so the discriminative signal
    is a few percent of the score.

Here the CNN is demoted to a witness. Each candidate becomes a node carrying
several independent evidence channels, the channels are fused in log space,
and the result is normalised across the whole candidate set so the nodes
compete: raising one node's belief necessarily lowers the others'. What comes
out is a belief (sums to 1, has an entropy, can be abstained on), not a score.

Evidence channels per node
  appearance   NCC of the raw patch against the scaled reference  (witness)
  embedding    learned patch embedding cosine                     (witness)
  landmark     NCC of the APERIODIC RESIDUAL -- the only channel that
               carries absolute identity inside an array           (judge)
  prior        Digital Twin annulus: |dist(node, nominal) - radius|
  phase        agreement with the lattice phase consensus of the whole
               candidate set -- a graph/neighbour-consistency term, not a
               property of the node alone
"""
import numpy as np


def softmax(x, temperature=1.0):
    z = np.asarray(x, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def robust_z(values, reference_map):
    """Score a value against the distribution of the map it came from.

    Self-normalising: a residual correlation map with a genuine landmark match
    has one bin far out in the tail; a map built from noise-vs-noise has a max
    that is only a few MADs out. So this number answers 'is this peak
    meaningful?' without any cross-image calibration."""
    m = np.asarray(reference_map, dtype=np.float64).ravel()
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) + 1e-9
    return (np.asarray(values, dtype=np.float64) - med) / (1.4826 * mad)


def lattice_phase_consensus(nodes, pitch, theta_deg, weights):
    """Circular-mean lattice phase of the candidate set.

    Every candidate implies a phase for the lattice it sits on. Candidates
    that agree reinforce a consensus; a candidate whose phase disagrees is
    sitting between cells and is almost certainly a correlation artefact.
    This is information no single-patch classifier can access -- it exists
    only in the relationship between candidates."""
    th = np.radians(theta_deg)
    u = np.array([n['x'] * np.cos(th) + n['y'] * np.sin(th) for n in nodes])
    ang = 2 * np.pi * (u % pitch) / pitch
    w = np.asarray(weights, dtype=np.float64)
    w = w - w.min() + 1e-6
    cbar = np.sum(w * np.cos(ang)) / w.sum()
    sbar = np.sum(w * np.sin(ang)) / w.sum()
    consensus = np.arctan2(sbar, cbar)
    strength = float(np.hypot(cbar, sbar))
    return np.cos(ang - consensus), strength


DEFAULT_WEIGHTS = dict(appearance=1.0, embedding=0.7, landmark=0.55,
                       prior=0.6, phase=0.25)


def fuse(nodes, prior_center, prior_radius, pitch, theta_deg,
         weights=None, temperature=0.35, landmark_z_cap=12.0):
    """Fuse evidence channels into a normalised belief over the candidate set.

    Returns (nodes_sorted, diagnostics). Each node gains 'belief' and the
    per-channel contributions that produced it, so any decision can be
    explained after the fact in terms of which sensor voted for what.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    if not nodes:
        return [], dict(entropy=0.0, margin=0.0, phase_strength=0.0)

    appearance = np.array([n['corr'] for n in nodes])
    embedding = np.array([n['embed'] for n in nodes])
    # landmark_z is already a tail score against its own map; squash so a
    # single spectacular residual peak cannot completely veto the geometry
    landmark = np.clip(np.array([n.get('landmark_z', 0.0) for n in nodes]),
                       0.0, landmark_z_cap) / landmark_z_cap

    dist_prior = np.array([abs(np.hypot(n['x'] - prior_center[0],
                                        n['y'] - prior_center[1]) - prior_radius)
                           for n in nodes])
    prior_term = -dist_prior / max(prior_radius, 30.0)

    phase_term, phase_strength = lattice_phase_consensus(
        nodes, pitch, theta_deg, appearance + landmark)

    logit = (w['appearance'] * appearance + w['embedding'] * embedding
             + w['landmark'] * landmark + w['prior'] * prior_term
             + w['phase'] * phase_term)

    belief = softmax(logit, temperature)
    for n, b, lg, a, e, l, p, ph in zip(nodes, belief, logit, appearance,
                                        embedding, landmark, prior_term, phase_term):
        n['belief'] = float(b)
        n['logit'] = float(lg)
        n['ev'] = dict(appearance=float(a), embedding=float(e), landmark=float(l),
                       prior=float(p), phase=float(ph))

    order = np.argsort(-belief)
    nodes_sorted = [nodes[i] for i in order]
    p = np.sort(belief)[::-1]
    entropy = float(-(belief * np.log(belief + 1e-12)).sum())
    margin = float(p[0] - p[1]) if len(p) > 1 else float(p[0])
    return nodes_sorted, dict(entropy=entropy, margin=margin,
                              phase_strength=float(phase_strength),
                              n_nodes=len(nodes))


def decide(nodes_sorted, diag, landmark_z_threshold=4.0, belief_threshold=0.20,
           nominal_center=(500.0, 500.0), tie_eps=0.02, n_near_peaks=None,
           gate=None):
    """Commit, or abstain.

    An abstention is not a failure -- inside a defect-free periodic array with
    no landmark in the field of view, the true site is genuinely unidentifiable
    from the images alone. Guessing a lattice cell at random has expected error
    ~1.27R; reporting the annulus centroid has expected error R. So when the
    evidence is not decisive the system says so, returns the
    minimum-expected-error point, and asks for another field of view rather
    than inventing a confident wrong answer.

    Two gates are available. `gate`, when supplied, is a calibrated threshold
    on n_near_peaks fitted by commit_gate.py against measured outcomes; it
    commits 33 sites at 97.0% precision on the evaluation set, held out over 5
    folds. Without one the hand-set landmark_z/belief thresholds apply, which
    commit 30 at 90.0% -- worse on both axes, and the reason the calibrated
    path exists.
    """
    if not nodes_sorted:
        return dict(x=nominal_center[0], y=nominal_center[1], abstain=True,
                    reason='no candidates', belief=0.0, node=None,
                    fallback_x=nominal_center[0], fallback_y=nominal_center[1])

    # Mandated deterministic resolver: when beliefs are statistically tied the
    # evidence genuinely does not separate them, so the tie is broken by the
    # problem statement's rule -- closest to the search image centre -- rather
    # than by whichever node happened to sort first.
    top = nodes_sorted[0]
    tied = [n for n in nodes_sorted if top['belief'] - n['belief'] < tie_eps]
    if len(tied) > 1:
        tied.sort(key=lambda n: np.hypot(n['x'] - nominal_center[0],
                                         n['y'] - nominal_center[1]))
        top = tied[0]
    best_lz = max(n.get('landmark_z', 0.0) for n in nodes_sorted)

    gated = None
    if gate is not None:
        import commit_gate
        # The gate now selects its own feature at calibration time, so it is
        # handed the whole diagnostic bundle rather than one hardcoded scalar.
        gated = commit_gate.apply(gate, dict(diag or {},
                                             n_near_peaks=n_near_peaks,
                                             landmark_z=best_lz))

    if gated is not None:
        commit = gated
        if commit:
            reason = 'committed'
        else:
            reason = (f'{gate["feature"]}={diag.get(gate["feature"], n_near_peaks)} '
                      f'fails gate ({gate["feature"]} {gate.get("sense", "<=")} '
                      f'{gate["threshold"]:.3g})')
    else:
        identifiable = best_lz >= landmark_z_threshold
        decisive = top['belief'] >= belief_threshold
        commit = identifiable and decisive
        if commit:
            reason = 'committed'
        elif not identifiable:
            reason = 'no landmark evidence in FOV (array interior)'
        else:
            reason = 'landmark present but belief split across lattice cells'

    # The maximum-belief hypothesis is ALWAYS reported -- abstention is a flag
    # on the answer, not a refusal to produce one, so nothing is lost relative
    # to an unconditional argmax. What the flag adds is the routing decision:
    # a flagged site should be re-imaged at a second field of view rather than
    # trusted, and 'fallback' carries the minimum-expected-error point to use
    # if no re-image is possible.
    return dict(x=top['x'], y=top['y'], abstain=not commit,
                reason=reason, belief=top['belief'], node=top,
                fallback_x=nominal_center[0], fallback_y=nominal_center[1])
