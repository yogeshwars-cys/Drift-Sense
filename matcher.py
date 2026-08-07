"""
Drift-Sense hybrid matcher.

Architecture (the CNN is a witness, not the judge):

    Search image + reference
              |
      Lattice sensor  -- measures pitch, orientation, and hence the unknown
              |          ~10x magnification directly from the two spectra
              +--> raw appearance channel (multi-scale NCC)
              +--> APERIODIC RESIDUAL channel (spectral notch removes the
              |    lattice; only landmarks survive -- the sole carrier of
              |    absolute identity inside a periodic array)
              +--> learned embedding channel (CNN)
              +--> Digital Twin annulus prior
              |
      Hypothesis graph -- candidates become mutually exclusive nodes, evidence
              |           is fused in log space and normalised across the set
              |
      Belief + decision -- commit, or abstain with the minimum-expected-error
                           estimate when the evidence is not there

Stage 3 (sub-pixel parabolic refinement) is unchanged and still runs on the
raw correlation surface of the committed node.
"""
import numpy as np
import cv2

from common import SEARCH_SIZE, REF_SIZE
from reranker_model import EmbedNet, embed
from lattice import estimate_scale, aperiodic_residual, rotate
from hypothesis_graph import fuse, decide, robust_z

NOMINAL_CENTER = (SEARCH_SIZE / 2, SEARCH_SIZE / 2)

SEARCH_PITCH_BAND = (5.0, 40.0)
REF_PITCH_BAND = (45.0, 320.0)


def _window(shape, center, radius):
    H, W = shape
    cx0, cy0 = center
    x_lo, x_hi = int(max(0, cx0 - radius)), int(min(W, cx0 + radius))
    y_lo, y_hi = int(max(0, cy0 - radius)), int(min(H, cy0 + radius))
    return x_lo, y_lo, x_hi, y_hi


def stage1_multiscale(search_u8, ref_u8, prior_center, prior_radius, margin=90,
                      scale_range=(8.3, 11.7), n_scales=8, top_k_per_scale=6,
                      nms_radius=25, rotations=(0.0,)):
    """Coarse appearance search over scale AND relative capture rotation.

    The scale bracket is supplied by the lattice sensor rather than swept
    blindly, so the reference footprint is typically within 1px of truth. The
    rotation cannot be supplied the same way -- see the note in lattice.py --
    so it is swept here, against the correlation surface that is actually
    sensitive to it. Candidates carry the rotation that found them so the
    refinement stage can reuse it.
    """
    x_lo, y_lo, x_hi, y_hi = _window(search_u8.shape, prior_center,
                                     prior_radius + margin)
    sub = search_u8[y_lo:y_hi, x_lo:x_hi]

    candidates = []
    for rot_deg in rotations:
        ref_rot = rotate(ref_u8, rot_deg)
        for scale in np.linspace(*scale_range, n_scales):
            foot = max(20, int(round(REF_SIZE / scale)))
            if sub.shape[0] <= foot + 2 or sub.shape[1] <= foot + 2:
                continue
            ref_small = cv2.resize(ref_rot, (foot, foot), interpolation=cv2.INTER_AREA)
            corr = cv2.matchTemplate(sub, ref_small, cv2.TM_CCOEFF_NORMED)
            cm = corr.copy()
            for _ in range(top_k_per_scale):
                idx = np.unravel_index(np.argmax(cm), cm.shape)
                score = float(cm[idx])
                if not np.isfinite(score) or score <= -1e8:
                    break
                py, px = idx
                candidates.append((score, float(px + foot // 2 + x_lo),
                                   float(py + foot // 2 + y_lo), foot,
                                   float(rot_deg)))
                y0s, y1s = max(0, py - nms_radius), min(cm.shape[0], py + nms_radius)
                x0s, x1s = max(0, px - nms_radius), min(cm.shape[1], px + nms_radius)
                cm[y0s:y1s, x0s:x1s] = -1e9
    return candidates


def landmark_evidence_map(res_search, res_ref, foot, window):
    """Correlate the two APERIODIC residuals.

    This is the channel the old architecture was missing entirely. Raw NCC and
    the CNN both see a patch that is ~95% periodic lattice, identical at every
    candidate; here the lattice has been spectrally subtracted, so what is
    being correlated is purely the landmark content -- array boundary, dropped
    or doubled via, gate crossing. In a true array interior this map is
    noise-vs-noise and stays flat, which is exactly the signal we need in order
    to know that we should abstain."""
    x_lo, y_lo, x_hi, y_hi = window
    sub = np.ascontiguousarray(res_search[y_lo:y_hi, x_lo:x_hi], dtype=np.float32)
    if sub.shape[0] <= foot + 2 or sub.shape[1] <= foot + 2:
        return None
    tmpl = cv2.resize(res_ref, (foot, foot), interpolation=cv2.INTER_AREA)
    tmpl = np.ascontiguousarray(tmpl, dtype=np.float32)
    if float(tmpl.std()) < 1e-6:
        return None
    return cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)


def peaks_from_map(cmap, window, foot, top_k=6, nms_radius=25, rot_deg=0.0):
    """Let the landmark channel PROPOSE candidates, not just score them.

    Appearance-driven Stage 1 can miss the true site entirely when the lattice
    correlation surface is flat; if the residual channel sees a landmark it is
    allowed to put its own hypothesis on the table."""
    x_lo, y_lo = window[0], window[1]
    cm = cmap.copy()
    out = []
    for _ in range(top_k):
        idx = np.unravel_index(np.argmax(cm), cm.shape)
        score = float(cm[idx])
        if not np.isfinite(score) or score <= -1e8:
            break
        py, px = idx
        out.append((score, float(px + foot // 2 + x_lo), float(py + foot // 2 + y_lo),
                    foot, float(rot_deg)))
        y0s, y1s = max(0, py - nms_radius), min(cm.shape[0], py + nms_radius)
        x0s, x1s = max(0, px - nms_radius), min(cm.shape[1], px + nms_radius)
        cm[y0s:y1s, x0s:x1s] = -1e9
    return out


def dedupe_spatial(candidates, radius=30, keep=15):
    candidates = sorted(candidates, key=lambda c: -c[0])
    kept = []
    for c in candidates:
        if all(np.hypot(c[1] - k[1], c[2] - k[2]) > radius for k in kept):
            kept.append(c)
        if len(kept) >= keep:
            break
    return kept


def crop_patch(search_u8, x, y, foot):
    half = foot // 2
    x0, y0 = int(round(x - half)), int(round(y - half))
    x1, y1 = x0 + foot, y0 + foot
    if x0 < 0 or y0 < 0 or x1 > search_u8.shape[1] or y1 > search_u8.shape[0]:
        return None
    return search_u8[y0:y1, x0:x1]


def sample_map(cmap, window, foot, x, y):
    """Read the landmark evidence map at a candidate's location."""
    if cmap is None:
        return None
    px = int(round(x - foot // 2 - window[0]))
    py = int(round(y - foot // 2 - window[1]))
    if 0 <= py < cmap.shape[0] and 0 <= px < cmap.shape[1]:
        return float(cmap[py, px])
    return None


def subpixel_refine(search_u8, ref_small, x, y, foot):
    half = foot // 2
    pad = 4
    x0, y0 = int(round(x - half - pad)), int(round(y - half - pad))
    x1, y1 = x0 + foot + 2 * pad, y0 + foot + 2 * pad
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(search_u8.shape[1], x1), min(search_u8.shape[0], y1)
    sub = search_u8[y0c:y1c, x0c:x1c]
    if sub.shape[0] <= foot or sub.shape[1] <= foot:
        return float(x), float(y)
    corr = cv2.matchTemplate(sub, ref_small, cv2.TM_CCOEFF_NORMED)
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def parab(a, b, c):
        denom = (a - 2 * b + c)
        return 0.0 if abs(denom) < 1e-6 else 0.5 * (a - c) / denom

    dx = parab(corr[py, px - 1], corr[py, px], corr[py, px + 1]) if 0 < px < corr.shape[1] - 1 else 0.0
    dy = parab(corr[py - 1, px], corr[py, px], corr[py + 1, px]) if 0 < py < corr.shape[0] - 1 else 0.0
    return float(x0c + px + foot / 2 + dx), float(y0c + py + foot / 2 + dy)


def match(search_u8, ref_u8, elapsed_time_s, twin, model,
          nominal_center=NOMINAL_CENTER, fallback_radius=900.0,
          landmark_z_threshold=4.0, belief_threshold=0.20, gate=None):
    radius, uncertainty = twin.predict(elapsed_time_s)
    window_r = radius + uncertainty

    # ---- lattice sensor: measure the magnification instead of sweeping ----
    scale_est, lat_s, lat_r, scale_ok = estimate_scale(search_u8, ref_u8)
    if scale_ok:
        scale_range, n_scales = (scale_est - 0.7, scale_est + 0.7), 8
    else:
        scale_range, n_scales = (8.5, 11.5), 10

    # Rotation is swept over a single angle, i.e. not swept. Measured, not
    # assumed: derotating by the ORACLE angle lifts the NCC at the true site by
    # +0.077 (+11.3%), so the penalty is real -- but sweeping 7 rotations and
    # keeping the best per candidate scores 40.0% against 46.0% for no sweep.
    # Maximising over a nuisance parameter lifts the periodic DISTRACTORS at
    # least as much as the true peak (each gets its own best-case angle, while
    # the true peak already sits near its optimum), so discrimination falls
    # even as absolute correlation rises. The knob is kept because it is now
    # measured rather than guessed; it stays at 1.
    rotations = (0.0,)

    candidates = stage1_multiscale(search_u8, ref_u8, nominal_center, window_r,
                                   scale_range=scale_range, n_scales=n_scales,
                                   rotations=rotations)
    used_fallback = False
    if not candidates:
        used_fallback = True
        candidates = stage1_multiscale(search_u8, ref_u8, nominal_center,
                                       fallback_radius, scale_range=scale_range,
                                       n_scales=n_scales, rotations=rotations)
    if not candidates:
        d = decide([], {}, nominal_center=nominal_center)
        return dict(x=d['x'], y=d['y'], confidence=0.0, belief=0.0, abstain=True,
                    abstain_reason=d['reason'], twin_radius=radius,
                    twin_uncertainty=uncertainty, used_fallback=True,
                    n_candidates=0, n_near_peaks=0, is_periodic_degenerate=True,
                    scale_est=scale_est, scale_ok=scale_ok, landmark_z=0.0,
                    entropy=0.0, top_candidates=[])

    # ---- aperiodic residual channel ----
    win = _window(search_u8.shape, nominal_center, window_r + 90)
    foot_ref = max(20, int(round(REF_SIZE / scale_est)))
    # The rotation that produced the strongest appearance candidate is the best
    # available estimate of the capture misalignment, so the residual channel
    # and the embedding anchor are built from a reference derotated by it
    # rather than from the raw reference.
    best_rot = max(candidates, key=lambda c: c[0])[4] if candidates else 0.0
    ref_work = rotate(ref_u8, best_rot)

    spec_diag = {}
    res_s = aperiodic_residual(search_u8, SEARCH_PITCH_BAND, diagnostics=spec_diag)
    res_r = aperiodic_residual(ref_work, REF_PITCH_BAND)
    lmap = landmark_evidence_map(res_s, res_r, foot_ref, win)
    if lmap is not None:
        candidates = candidates + peaks_from_map(lmap, win, foot_ref,
                                                 rot_deg=best_rot)

    # ---- build hypothesis nodes ----
    deduped = dedupe_spatial(candidates)
    patches, valid = [], []
    for c in deduped:
        p = crop_patch(search_u8, c[1], c[2], c[3])
        if p is not None:
            patches.append(p)
            valid.append(c)
    if not valid:
        d = decide([], {}, nominal_center=nominal_center)
        return dict(x=d['x'], y=d['y'], confidence=0.0, belief=0.0, abstain=True,
                    abstain_reason=d['reason'], twin_radius=radius,
                    twin_uncertainty=uncertainty, used_fallback=used_fallback,
                    n_candidates=0, n_near_peaks=0, is_periodic_degenerate=True,
                    scale_est=scale_est, scale_ok=scale_ok, landmark_z=0.0,
                    entropy=0.0, top_candidates=[])

    emb_anchor = embed(model, [ref_work])[0]
    embed_sims = embed(model, patches) @ emb_anchor

    nodes = []
    for (score, x, y, foot, rot_deg), esim in zip(valid, embed_sims):
        raw = sample_map(lmap, win, foot_ref, x, y)
        nodes.append(dict(corr=float(score), x=float(x), y=float(y), foot=int(foot),
                          rotation_deg=float(rot_deg), scale=float(REF_SIZE / foot),
                          embed=float(esim), landmark_raw=raw))
    if lmap is not None:
        raws = np.array([n['landmark_raw'] if n['landmark_raw'] is not None else np.nan
                         for n in nodes])
        zs = robust_z(np.nan_to_num(raws, nan=0.0), lmap)
        for n, z in zip(nodes, zs):
            n['landmark_z'] = float(z) if n['landmark_raw'] is not None else 0.0
    else:
        for n in nodes:
            n['landmark_z'] = 0.0

    # ---- fuse into beliefs and decide ----
    ranked, diag = fuse(nodes, nominal_center, radius,
                        pitch=lat_s['pitch'], theta_deg=lat_s['theta_deg'])

    # How many candidates the correlation surface cannot separate. Computed
    # before the decision because it is the gate's strongest input, not an
    # after-the-fact diagnostic.
    top_corr = max(c[0] for c in candidates)
    n_near_peaks = sum(1 for c in candidates if top_corr - c[0] <= 0.02)

    d = decide(ranked, diag, landmark_z_threshold=landmark_z_threshold,
               belief_threshold=belief_threshold, nominal_center=nominal_center,
               n_near_peaks=n_near_peaks, gate=gate)

    node = d['node']
    # refine against the reference at the rotation that actually found this node
    ref_small = cv2.resize(rotate(ref_u8, node.get('rotation_deg', 0.0)),
                           (node['foot'], node['foot']),
                           interpolation=cv2.INTER_AREA)
    x_out, y_out = subpixel_refine(search_u8, ref_small, node['x'], node['y'],
                                   node['foot'])

    best_lz = max(n.get('landmark_z', 0.0) for n in ranked)
    return dict(x=x_out, y=y_out,
                confidence=float(d['belief']), belief=float(d['belief']),
                abstain=bool(d['abstain']), abstain_reason=d['reason'],
                entropy=diag['entropy'], margin=diag['margin'],
                phase_strength=diag['phase_strength'],
                landmark_z=float(best_lz),
                fallback_x=d['fallback_x'], fallback_y=d['fallback_y'],
                scale_est=float(scale_est), scale_ok=bool(scale_ok),
                rotation_deg=float(node.get('rotation_deg', 0.0)),
                harmonic_cap_hit=bool(spec_diag.get('harmonic_cap_hit', False)),
                n_harmonics_found=int(spec_diag.get('n_harmonics_found', 0)),
                twin_radius=radius, twin_uncertainty=uncertainty,
                used_fallback=used_fallback, n_candidates=len(ranked),
                n_near_peaks=n_near_peaks,
                is_periodic_degenerate=bool(best_lz < landmark_z_threshold),
                top_candidates=[{k: v for k, v in n.items() if k != 'node'}
                                for n in ranked[:5]])
