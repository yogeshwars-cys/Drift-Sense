"""
Calibrated commit gate.

The decision "commit this localization, or flag it for a second field of view"
was two hand-set constants: landmark_z >= 4.0 and belief >= 0.20. Measured on
the original 100-pair evaluation set that gate committed 30 sites at 90.0%
precision, and abstained on 20 sites where a landmark WAS in the field of view
and the matcher had already localized to a median 0.6px -- answers discarded in
favour of a ~114px centre guess.

What the gate never consulted is a signal that observes the failure mode
directly. Held out over 5 folds on that set:

    gate                                 commits   precision   correct/100
    hand-set (lz>=4.0, belief>=0.20)      30        90.0%       27
    n_near_peaks <= K                     33        97.0%       32
    logistic on all four signals          23        82.6%       19

The single threshold beat the hand-set pair on both axes at once, and beat the
fitted logistic on the same folds. That ordering was not an accident of tuning:
with n=100 and a 46% base rate there is not enough signal to identify four
coefficients plus an operating point, so the richer model spends its capacity on
fold noise. The logistic was implemented, measured, and dropped.

WHAT CHANGED. `n_near_peaks` counts candidates within a fixed 0.02 NCC of the
best, which makes it (a) an integer with very few distinct values, so the
threshold can only land on a handful of operating points, and (b) a function of
how many scales the proposal stage swept -- narrowing the scale bracket changes
the count without changing the underlying ambiguity. It is a proxy. The
replacements are continuous and measure the thing itself:

    ambiguity          peak-to-second-peak separation of the landmark
                       correlation surface, rivals searched >= 1 pitch away
    residual_saliency  is there a landmark in the reference AT ALL, computed
                       without ground truth -- the direct solvability signal
    landmark_z         how far the winner's residual score stands outside the
                       spread of the candidate set

The feature is now SELECTED as part of the fit rather than written in, and
selected inside each CV fold, so the reported held-out numbers pay for the
selection too. Choosing the feature on all the data and then cross-validating
only its threshold would leak.

The threshold and the feature are both properties of the candidate generator and
the trial distribution, not universal constants; refit whenever either changes.
"""
import json
import numpy as np

# (name, sense) -- sense is the direction that means "more confident".
#
# `induction_score` carries the sense that looks backwards and is not: a LOW
# score means the frame's lattice failed to prove its own geometry, i.e. the
# lattice is irregular, i.e. there is aperiodic structure in the frame to
# localize against. See induction.py -- 86.7% accuracy when negative against
# 30.6% when positive. It is the only candidate here computed from the search
# image alone, without reference to the candidate set, so it fails
# independently of the other four.
CANDIDATE_FEATURES = (
    ('ambiguity', '>='),
    ('residual_saliency', '>='),
    ('landmark_z', '>='),
    ('fine_landmark', '>='),
    ('n_near_peaks', '<='),
    ('induction_score', '<='),
)
DEFAULT_PRECISION_FLOOR = 0.90


def _wilson_lower(k, n, z=1.0):
    """Lower confidence bound on a proportion. Choosing a threshold on the point
    estimate of precision picks whichever one got lucky on the fit fold; scoring
    the bound makes the choice pay for the sample size behind it, which is what
    makes it survive held out."""
    if n == 0:
        return 0.0
    ph = k / n
    d = 1.0 + z * z / n
    c = ph + z * z / (2 * n)
    m = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float((c - m) / d)


def _values(records, feature):
    out = []
    for r in records:
        try:
            v = float(r.get(feature, np.nan))
        except (TypeError, ValueError):
            v = np.nan
        out.append(v)
    return np.array(out, dtype=np.float64)


def _outcomes(records):
    return np.array([bool(r['success']) for r in records])


def _mask(v, sense, thr):
    m = (v <= thr) if sense == '<=' else (v >= thr)
    return m & np.isfinite(v)


def _best_threshold(v, y, sense, precision_floor):
    """Threshold admitting the MOST sites whose precision lower bound clears
    the floor."""
    best = (None, 0, 0.0)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return best
    for t in np.unique(finite):
        m = _mask(v, sense, t)
        n = int(m.sum())
        if n == 0:
            continue
        if _wilson_lower(float(y[m].sum()), n) >= precision_floor and n > best[1]:
            best = (float(t), n, float(y[m].mean()))
    return best


def _best_feature(records, precision_floor):
    """Pick the (feature, threshold) with the most coverage clearing the floor."""
    y = _outcomes(records)
    best = None
    for feature, sense in CANDIDATE_FEATURES:
        v = _values(records, feature)
        if not np.isfinite(v).any():
            continue
        thr, ncom, prec = _best_threshold(v, y, sense, precision_floor)
        if thr is None:
            continue
        if best is None or ncom > best[2]:
            best = (feature, sense, ncom, thr, prec)
    if best is not None:
        return best
    # Floor unreachable on every feature: be strictest on the first usable one
    # rather than silently committing everything.
    for feature, sense in CANDIDATE_FEATURES:
        v = _values(records, feature)
        if np.isfinite(v).any():
            f = v[np.isfinite(v)]
            thr = float(f.min()) if sense == '<=' else float(f.max())
            m = _mask(v, sense, thr)
            return (feature, sense, int(m.sum()), thr,
                    float(y[m].mean()) if m.sum() else 0.0)
    return ('n_near_peaks', '<=', 0, 0.0, 0.0)


def fit(records, precision_floor=DEFAULT_PRECISION_FLOOR):
    """Fit on everything. Produces the shipped gate; never quote its in-sample
    coverage/precision as a performance claim -- use fit_cv for that."""
    feature, sense, ncom, thr, prec = _best_feature(records, precision_floor)
    return dict(feature=feature, sense=sense, threshold=thr,
                precision_floor=precision_floor, n_fit=len(records),
                fit_coverage=ncom / max(len(records), 1), fit_precision=prec,
                candidates=[f for f, _ in CANDIDATE_FEATURES])


def fit_cv(records, k=5, precision_floor=DEFAULT_PRECISION_FLOOR, seed=0):
    """K-fold: fit feature AND threshold on k-1 folds, apply to the held-out
    fold. The pooled held-out decisions are the only honest estimate of
    out-of-sample behaviour -- and because the feature is chosen inside the
    fold, they price in the selection as well as the threshold."""
    y = _outcomes(records)
    n = len(records)
    fold = np.random.default_rng(seed).permutation(n) % k

    committed = np.zeros(n, dtype=bool)
    picks = []
    for f in range(k):
        tr_idx = np.nonzero(fold != f)[0]
        te_idx = np.nonzero(fold == f)[0]
        feature, sense, _, thr, _ = _best_feature(
            [records[i] for i in tr_idx], precision_floor)
        picks.append(dict(feature=feature, sense=sense, threshold=thr))
        v_te = _values([records[i] for i in te_idx], feature)
        committed[te_idx] = _mask(v_te, sense, thr)

    ncom = int(committed.sum())
    return dict(k=k, n=n, n_committed=ncom, coverage=ncom / max(n, 1),
                precision=float(y[committed].mean()) if ncom else 0.0,
                correct_per_100=float(y[committed].sum()) / max(n, 1) * 100.0,
                fold_picks=picks, committed_mask=committed.tolist())


def apply(gate, diagnostics):
    """Inference-time decision. -> commit: bool, or None when no gate is loaded
    (caller falls back to the hand-set thresholds).

    `diagnostics` is the localizer's diagnostics dict. A bare scalar is also
    accepted, and is read as the value of the gate's own feature.
    """
    if not gate:
        return None
    if isinstance(diagnostics, dict):
        if gate['feature'] not in diagnostics:
            return None
        v = diagnostics[gate['feature']]
    else:
        v = diagnostics
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    thr = float(gate['threshold'])
    return bool(v <= thr) if gate.get('sense', '<=') == '<=' else bool(v >= thr)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
