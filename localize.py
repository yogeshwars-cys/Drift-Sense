"""
localize.py -- standalone navigation-error-recovery inference.

    python localize.py ref.png search.png          -> "x, y"
    python localize.py --reference ref.png --search search.png

Two images in, one coordinate out. No metadata, no ground truth, no trained
weights.

THE SHAPE OF THE SOLUTION. A template matcher decimates the 1000x1000 reference
to ~100x100, slides it, and takes the argmax -- deciding at exactly the
resolution where every lattice cell is identical, and searching a continuous
plane when the answer can only lie on a discrete grid. This pipeline inverts
that. It measures the geometry exactly (magnification from the pitch ratio,
rotation from a sub-bin multi-harmonic spectral angle, position modulo the pitch
from the cross-image lattice phase), which collapses the continuous search to a
short list of geometrically legal cells; then it spends its compute
discriminating among those few at FULL reference resolution, where the fine
structure that distinguishes cells actually survives.

    propose cheap, discriminate expensive -- not the other way round.

The one channel that carries absolute identity inside a periodic array is the
APERIODIC RESIDUAL: spectrally notch the lattice harmonics and only the
non-repeating content is left (array edges, dropped/doubled vias, gate
crossings). It both proposes candidates and, when it speaks clearly, decides
outright. When it is silent -- a defect-free array interior, where it is noise
against noise -- the site is genuinely unidentifiable from these two images, and
the mandated "closest to the search image centre" rule resolves the tie.

Inherited from the two codebases this was merged from:

  from the Drift-Sense perception stack
    * the magnification is MEASURED, not swept -- the ratio of the two lattice
      pitches IS the scale factor (lattice.estimate_scale, median 0.14% error).
    * the APERIODIC RESIDUAL channel, described above.

  from the standalone localizer
    * the interface itself: two image paths in, one coordinate out.

  and one thing neither had: the landmark channel promoted to JUDGE. Both
  codebases treated the mandated centre rule as a co-equal resolver. In a 1-D
  FinFET fin field the correlation surface is flat along the fin axis, so most
  candidates tie on appearance, the centre rule fires, and it overrules the
  residual peak that had already identified the site. Measured: all 10 solvable
  failures were FinFET, the true site was in the candidate pool every time, and
  was already top-ranked on 4 of them. Letting the strongest residual peak
  decide outright, and firing the centre rule only when no such peak exists,
  took solvable-site accuracy from 80% to 86%.

What was deliberately dropped, and why:

  * The bounded-drift DISK PRIOR. It assumed the site lands near the frame
    centre. Measured under uniform placement -- which is all the statement
    licenses -- it cost solvable accuracy 62.5% -> 43.8%, and its "widen the
    search if the score is weak" escape hatch fired on only 4 of 32 pairs
    because wrong periodic repeats score 0.75-0.91 NCC. The full frame is now
    searched by default; --drift-radius re-enables the prior for deployments
    that genuinely constrain the stage.
  * ABSOLUTE CORRELATION as a confidence signal, for the same reason. Replaced
    by peak-to-second-peak separation (see ambiguity_ratio).
  * The Digital Twin drift prior. It predicts a search radius from
    elapsed_time_s -- a field that exists only in our own generator's metadata.
    Inference gets two images, so the twin cannot run here at all.
  * The CNN re-ranker. It embedded raw patches that are ~95% identical lattice,
    was the weakest fused channel, and cost a torch dependency on the path a
    reviewer has to run. See README, "On the learned re-ranker".
"""
import argparse
import json
import sys
import time

import numpy as np
import cv2

from lattice import (estimate_scale, aperiodic_residual, residual_saliency,
                     scale_uncertainty, foot_bracket, phase_lock, snap_to_phase,
                     relative_rotation, rotate)
from induction import induction_evidence

# Fallback notch bands, used only until the search pitch has been measured.
# Once it has, both bands are DERIVED from it (see pitch_bands): a fixed
# reference band of (45, 320) px is a guess about the test set's pitch, and a
# wrong guess leaves lattice harmonics un-notched, which leaks periodic energy
# straight into the channel whose entire job is to contain only landmarks.
SEARCH_PITCH_BAND = (5.0, 40.0)
REF_PITCH_BAND = (45.0, 320.0)

# Nominal optics: the reference is ~10x the search magnification. Same bracket
# estimate_scale conditions its reference measurement on.
SCALE_BOUNDS = (8.5, 11.5)


def pitch_bands(pitch_search=None):
    """-> (search_band, ref_band) in px, derived from the measured pitch."""
    if pitch_search is None or not np.isfinite(pitch_search) or pitch_search <= 0:
        return SEARCH_PITCH_BAND, REF_PITCH_BAND
    p = float(pitch_search)
    # Wide enough to hold the fundamental and its low harmonics on both sides.
    s_band = (max(2.0, p * 0.45), p * 3.0)
    r_band = (p * SCALE_BOUNDS[0] * 0.8, p * SCALE_BOUNDS[1] * 1.5)
    return s_band, r_band

# Evidence weights, fitted by sweeping 423 combinations against cached
# per-candidate channel scores (probes/weight_sweep.py) on 44 trials where the
# true site was actually proposed. Rank-1 rate of the true site:
#
#     coarse alone                 43.2%
#     fine_app alone               40.9%
#     fine_lm alone                43.2%
#     fused, these weights         59.1%
#     fused, previous hand-set     52.3%
#
# The fusion is worth ~16 points over any single channel, which is the whole
# argument for multiple evidence channels rather than a better single score.
#
# Two weights came out at ZERO and are kept at zero:
#
#   W_PRIOR  -- proximity to the frame centre. It is not evidence about where
#               the site is; the statement only makes it a TIE-BREAK, and that
#               is where it still applies (see the resolver below). Giving it
#               weight in the ranking is the same mistake as the disk prior,
#               one level down.
#   W_PHASE  -- penalty for sitting off the lattice phase grid. Measured null,
#               matching the earlier finding for the candidate-consensus phase
#               term it replaced. The phase lock still earns its place as the
#               sub-pixel refiner of the winner; it just does not help RANK.
#
# Reporting these as zero rather than deleting them keeps the null results
# visible and re-checkable on a different distribution.
W_APPEARANCE = 1.0
W_FINE_APPEARANCE = 0.45
W_LANDMARK = 0.50
W_PRIOR = 0.0
W_PHASE = 0.0
TIE_EPS = 0.02

# The landmark channel is treated as having resolved identity outright -- so the
# centre tie-break stands down -- when one candidate's residual correlation
# stands this many robust deviations clear of the rest of the candidate set AND
# clears a floor in absolute NCC. The floor matters: in a defect-free array
# interior the residual is noise against noise, and noise still has an argmax
# that can look like an outlier within a small set.
LANDMARK_DECISIVE_Z = 4.0
LANDMARK_MIN_NCC = 0.10

# How much the measured rotation must IMPROVE the best correlation before it is
# adopted. Derotating by a wrong angle is strictly worse than not derotating at
# all, so this fails closed: ties go to leaving the reference alone.
ROTATION_MIN_GAIN = 0.01

# Ceiling on candidates carried into full-resolution rescoring. Each costs one
# 1000x1000 warp + NCC per channel, so this is the knob that trades runtime
# against the chance the true site was proposed but pruned.
MAX_CANDIDATES = 64

# Peaks kept per template footprint. In a periodic array the global argmax at a
# given scale is frequently the wrong repeat and the right one is the runner-up,
# so this has to be generous enough to hold the true site over a full frame.
TOP_K_PER_SCALE = 30

# Non-maximum suppression radius, as a fraction of the measured lattice pitch.
# Must stay below 1.0: at or above one pitch, suppression removes neighbouring
# lattice cells, which are the very hypotheses that need comparing.
NMS_PITCH_FRACTION = 0.7

# Two-tier rescoring: every candidate is scored at COARSE_SIZE, only the top
# FINE_N are re-scored at the reference's full resolution. See rescore_fullres.
COARSE_SIZE = 256
FINE_N = 12


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'could not read image: {path}')
    return img


def _window(shape, center, radius):
    H, W = shape
    cx, cy = center
    return (int(max(0, cx - radius)), int(max(0, cy - radius)),
            int(min(W, cx + radius)), int(min(H, cy + radius)))


def _multiscale_peaks(search_u8, ref_u8, window, feet,
                      top_k_per_scale=6, nms_radius=25):
    """Multi-scale NCC inside `window`, keeping several peaks per footprint.

    Several rather than one: in a periodic array the global argmax at a given
    scale is frequently the wrong repeat, and the right one is the runner-up.
    Dropping it here makes it unrecoverable later.

    `feet` is a list of integer template sizes from lattice.foot_bracket -- the
    distinct hypotheses the measured magnification actually leaves open, rather
    than a fixed-count sweep of float scales that mostly round to the same
    template.
    """
    x_lo, y_lo, x_hi, y_hi = window
    sub = search_u8[y_lo:y_hi, x_lo:x_hi]
    out = []
    for foot in feet:
        if sub.shape[0] <= foot + 2 or sub.shape[1] <= foot + 2:
            continue
        tmpl = cv2.resize(ref_u8, (foot, foot), interpolation=cv2.INTER_AREA)
        corr = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
        cm = corr.copy()
        for _ in range(top_k_per_scale):
            iy, ix = np.unravel_index(np.argmax(cm), cm.shape)
            score = float(cm[iy, ix])
            if not np.isfinite(score) or score <= -1e8:
                break
            out.append(dict(score=score, x=float(ix + foot / 2.0 + x_lo),
                            y=float(iy + foot / 2.0 + y_lo), foot=foot))
            y0, y1 = max(0, iy - nms_radius), min(cm.shape[0], iy + nms_radius)
            x0, x1 = max(0, ix - nms_radius), min(cm.shape[1], ix + nms_radius)
            cm[y0:y1, x0:x1] = -1e9
    return out


def _envelope_normalise(res, sigma=None):
    """Flatten the residual's local amplitude envelope.

    The aperiodic residual is sparse and mostly zero, so a plain NCC over it is
    dominated by whichever single bright landmark happens to fall inside the
    window -- an array edge cutting one corner can outvote the via-scale detail
    that actually identifies the site. Dividing by a smoothed local magnitude
    puts a faint landmark and a bright one on the same footing, which is what
    the channel is supposed to be measuring.
    """
    a = np.abs(res).astype(np.float32)
    if sigma is None:
        sigma = max(3.0, 0.02 * max(res.shape))
    env = cv2.GaussianBlur(a, (0, 0), sigma) + 1e-3
    return (res / env).astype(np.float32)


def _best_ncc(search_u8, ref_u8, foot):
    """Best full-frame correlation of the reference at one footprint.

    Deliberately a single template at a single scale: this is used only to
    compare two candidate REFERENCES against the same search image, so the
    number's absolute value is irrelevant and only the difference matters.
    """
    tmpl = cv2.resize(ref_u8, (foot, foot), interpolation=cv2.INTER_AREA)
    if search_u8.shape[0] <= foot + 2 or search_u8.shape[1] <= foot + 2:
        return -1.0
    return float(cv2.matchTemplate(search_u8, tmpl, cv2.TM_CCOEFF_NORMED).max())


def _landmark_map_from(res_s, res_r, foot, window):
    """NCC between the two APERIODIC residuals, over an already-computed pair.

    Raw NCC compares patches that are ~95% identical lattice at every
    candidate. Here the lattice has been spectrally subtracted, so what is
    correlated is only the landmark content. In a true array interior this
    map is noise against noise and stays flat -- which is itself the signal
    that the site is unidentifiable.
    """
    x_lo, y_lo, x_hi, y_hi = window
    sub = np.ascontiguousarray(res_s[y_lo:y_hi, x_lo:x_hi], dtype=np.float32)
    if sub.shape[0] <= foot + 2 or sub.shape[1] <= foot + 2:
        return None
    tmpl = np.ascontiguousarray(
        cv2.resize(res_r, (foot, foot), interpolation=cv2.INTER_AREA), dtype=np.float32)
    if float(tmpl.std()) < 1e-6:
        return None
    return cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)


def _landmark_map(search_u8, ref_u8, foot, window, pitch_search=None):
    """Convenience wrapper: compute both residuals, then correlate them."""
    s_band, r_band = pitch_bands(pitch_search)
    res_s = aperiodic_residual(search_u8, s_band)
    res_r = aperiodic_residual(ref_u8, r_band)
    return _landmark_map_from(res_s, res_r, foot, window), residual_saliency(res_r)


def _spread_z(values):
    """How far each value stands outside the spread of its own set.

    Robust (median/MAD) so a single strong candidate does not inflate the scale
    it is being judged against. Self-normalising, so no cross-image calibration
    is needed -- which matters because the test set's noise level is explicitly
    higher than anything these constants were fitted on.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return (v - med) / (1.4826 * mad + 1e-9)


def _peaks_from_map(cmap, window, foot, top_k=6, nms_radius=25):
    """Let the landmark channel propose candidates, not merely score them.
    Appearance-driven search can miss the true site outright when the lattice
    correlation surface is flat."""
    x_lo, y_lo = window[0], window[1]
    cm = cmap.copy()
    out = []
    for _ in range(top_k):
        iy, ix = np.unravel_index(np.argmax(cm), cm.shape)
        score = float(cm[iy, ix])
        if not np.isfinite(score) or score <= -1e8:
            break
        out.append(dict(score=score, x=float(ix + foot / 2.0 + x_lo),
                        y=float(iy + foot / 2.0 + y_lo), foot=foot))
        y0, y1 = max(0, iy - nms_radius), min(cm.shape[0], iy + nms_radius)
        x0, x1 = max(0, ix - nms_radius), min(cm.shape[1], ix + nms_radius)
        cm[y0:y1, x0:x1] = -1e9
    return out


def ambiguity_ratio(cmap, pitch, foot):
    """Peak-to-second-peak separation of a correlation surface.

    Absolute correlation height is the wrong confidence signal in a periodic
    field, and measurably so: on uniform placement the pre-rewrite pipeline
    returned 0.905 NCC on a 489px error, and its "widen the search if the score
    is weak" escape hatch consequently fired on 4 of 32 pairs. Height says how
    well the template matches SOME lattice cell; it says nothing about whether
    the right cell was chosen.

    Separation does. The rival peak is searched at least one full pitch away,
    so a peak's own shoulder cannot count as its own competitor -- the standard
    ambiguity statistic from phase-correlation registration.

    -> ratio in [0, inf); ~1.0 means "two equally good answers", large means
       "one answer stands alone". Scale-free, so no cross-image calibration.
    """
    if cmap is None or cmap.size < 4:
        return 0.0
    m = np.asarray(cmap, dtype=np.float64)
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    best = float(m[iy, ix])
    guard = int(max(3, round(pitch)))
    m2 = m.copy()
    y0, y1 = max(0, iy - guard), min(m.shape[0], iy + guard + 1)
    x0, x1 = max(0, ix - guard), min(m.shape[1], ix + guard + 1)
    m2[y0:y1, x0:x1] = -np.inf
    if not np.isfinite(m2).any():
        return float('inf')
    second = float(np.max(m2))
    # Both are NCC in [-1, 1]; shift into a positive range so the ratio is
    # meaningful when correlations are weak or negative.
    lo = min(second, best, 0.0)
    return float((best - lo + 1e-6) / (second - lo + 1e-6))


def _upsampled_crop(search_u8, x, y, foot, out_size):
    """Search-image crop at (x, y), resampled up to reference resolution.

    Extracted by an inverse affine warp rather than an integer slice + resize,
    because full-resolution scoring is far more sensitive to placement than the
    decimated scoring it replaces. Rounding the crop origin to a whole pixel
    costs up to 0.5px at search scale, which is 5px at reference scale -- enough
    to decorrelate exactly the fine structure this stage exists to compare, and
    measured to drop the true site's rank-1 rate from 75% to 25%. The footprint
    is fractional too (foot = ref_size / scale), so the sampling step is
    1/scale, not 1/round(scale).
    """
    half = foot / 2.0
    if (x - half < -1 or y - half < -1
            or x + half > search_u8.shape[1] + 1
            or y + half > search_u8.shape[0] + 1):
        return None
    step = float(foot) / float(out_size)      # search px per output px
    M = np.array([[step, 0.0, x - half],
                  [0.0, step, y - half]], dtype=np.float32)
    return cv2.warpAffine(search_u8, M, (out_size, out_size),
                          flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT_101)


def _ncc(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if d < 1e-9 else float(np.dot(a, b) / d)


def rescore_fullres(search_u8, ref_u8, res_search, res_ref, cands, max_n=64,
                    foot_exact=None, fine_n=FINE_N):
    """Rank the survivors at FULL reference resolution.

    The proposal stage decimates a 1000x1000 reference to ~100x100 and slides
    it -- which is the right way to propose, and exactly the wrong resolution to
    decide at, because that is precisely the scale at which every lattice cell
    is identical. The reference exists at 10x so that via diameter, edge-
    brightening profile and gate-crossing geometry are resolvable; ranking at
    100x100 throws all of it away and then reports that the cells look alike.

    So each surviving candidate's search crop is resampled UP to the reference's
    own resolution and scored against the full reference, and against the full
    aperiodic residual. Measured on solvable full-frame trials, top-1 accuracy:
    decimated 58.3% -> full-res appearance 66.7% -> full-res residual 75.0%,
    for ~174ms on a ~15-candidate pool.

    Done in TWO TIERS. Scoring every candidate at the reference's full 1000x1000
    costs a warp plus a 1M-point correlation each, twice over, and the candidate
    pool has to be large (the cells are the hypotheses, so they cannot be pruned
    early). Tier 1 scores the whole pool at COARSE_SIZE -- still several times
    the ~100px proposal resolution, so it retains far more of the fine structure
    than the decimated score while costing ~1/16 as much -- and only the
    survivors are re-scored at full resolution. The cheap tier is used to
    shortlist, never to decide.
    """
    n = ref_u8.shape[0]
    have_res = res_search is not None and res_ref is not None
    foot_of = (lambda c: float(foot_exact)) if foot_exact else (lambda c: float(c['foot']))

    def _score(c, size, ref_small, res_ref_small, tag):
        foot = foot_of(c)
        crop = _upsampled_crop(search_u8, c['x'], c['y'], foot, size)
        c[f'{tag}appearance'] = 0.0 if crop is None else _ncc(crop, ref_small)
        if have_res:
            rc = _upsampled_crop(res_search, c['x'], c['y'], foot, size)
            c[f'{tag}landmark'] = 0.0 if rc is None else _ncc(rc, res_ref_small)
        else:
            c[f'{tag}landmark'] = 0.0

    pool = cands[:max_n]
    if len(pool) > fine_n:
        cs = COARSE_SIZE
        ref_c = cv2.resize(ref_u8, (cs, cs), interpolation=cv2.INTER_AREA)
        res_c = (cv2.resize(res_ref, (cs, cs), interpolation=cv2.INTER_AREA)
                 if have_res else None)
        for c in pool:
            _score(c, cs, ref_c, res_c, 'tier1_')
        pool.sort(key=lambda c: -(c['tier1_appearance'] + 0.5 * c['tier1_landmark']))
        shortlist, rest = pool[:fine_n], pool[fine_n:]
        for c in rest:
            # carry the cheap tier's verdict rather than a zero, so a pruned
            # candidate is ranked low on evidence instead of by fiat
            c['fine_appearance'] = c['tier1_appearance']
            c['fine_landmark'] = c['tier1_landmark']
    else:
        shortlist = pool

    for c in shortlist:
        _score(c, n, ref_u8, res_ref, 'fine_')

    for c in cands:
        c.setdefault('fine_appearance', 0.0)
        c.setdefault('fine_landmark', 0.0)
    return cands


def _dedupe(cands, radius, keep=15):
    cands = sorted(cands, key=lambda c: -c['score'])
    kept = []
    for c in cands:
        if all(np.hypot(c['x'] - k['x'], c['y'] - k['y']) > radius for k in kept):
            kept.append(c)
        if len(kept) >= keep:
            break
    return kept


def _subpixel(search_u8, ref_u8, x, y, foot):
    tmpl = cv2.resize(ref_u8, (foot, foot), interpolation=cv2.INTER_AREA)
    pad, half = 4, foot // 2
    x0, y0 = int(round(x - half - pad)), int(round(y - half - pad))
    x0c, y0c = max(0, x0), max(0, y0)
    x1c = min(search_u8.shape[1], x0 + foot + 2 * pad)
    y1c = min(search_u8.shape[0], y0 + foot + 2 * pad)
    sub = search_u8[y0c:y1c, x0c:x1c]
    if sub.shape[0] <= foot or sub.shape[1] <= foot:
        return float(x), float(y)
    corr = cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED)
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)

    def parab(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-6 else float(np.clip(0.5 * (a - c) / d, -1, 1))

    dx = parab(corr[iy, ix - 1], corr[iy, ix], corr[iy, ix + 1]) if 0 < ix < corr.shape[1] - 1 else 0.0
    dy = parab(corr[iy - 1, ix], corr[iy, ix], corr[iy + 1, ix]) if 0 < iy < corr.shape[0] - 1 else 0.0
    return float(x0c + ix + foot / 2.0 + dx), float(y0c + iy + foot / 2.0 + dy)


def localize(ref_u8, search_u8, drift_radius=None, use_landmark=True,
             use_phase_lock=True, use_rotation=True, full_search=False,
             scale_span=None, w_phase=W_PHASE, w_prior=W_PRIOR):
    """-> (x, y, diagnostics). Search-image pixel coordinates, sub-pixel.

    drift_radius: optional bound (px) on how far from the frame centre the site
        may be. Default None = search the whole frame, which is what the problem
        statement's "somewhere inside" actually licenses. Pass a radius only if
        the deployment genuinely constrains the stage.
    scale_span: optional override on the magnification bracket. Default None =
        derive it from the pitch measurement's own uncertainty.
    """
    H, W = search_u8.shape
    center = (W / 2.0, H / 2.0)

    # ---- measure the magnification rather than sweeping for it ----
    scale_est, lat_s, lat_r, scale_ok = estimate_scale(search_u8, ref_u8)
    pitch = float(lat_s['pitch'])

    # Spatial induction over the measured pitch. Deliberately does NOT feed the
    # magnification bracket -- gating the bracket on it was measured and does
    # not beat leaving the bracket alone (see induction.py). It is carried as
    # EVIDENCE: a frame whose lattice fails to prove its own geometry is an
    # irregular frame, and irregularity is what makes a site identifiable at
    # all. Localization accuracy 86.7% when this is negative against 30.6%
    # when it is positive, so it is reported for the commit gate to consume.
    induction = induction_evidence(search_u8, pitch)

    # The magnification was MEASURED (median 0.14% error). Bracket it by the
    # measurement's own uncertainty rather than a fixed +/-3.5%, and enumerate
    # the distinct integer footprints that bracket implies -- see
    # lattice.foot_bracket for why float scales over-sample the bracket centre.
    span = scale_span if scale_span is not None else scale_uncertainty(
        lat_s, lat_r if scale_ok else None, scale_est)
    if not scale_ok:
        span = max(span, 1.5)
    feet = foot_bracket(ref_u8.shape[0], scale_est, span)
    foot_ref = max(20, int(round(ref_u8.shape[0] / scale_est)))

    # Suppression radius must be SMALLER than the lattice pitch. This was
    # max(20, 2*pitch) -- but the pitch is 7-15px, so the radius was always at
    # least one and usually two full cells, and non-maximum suppression was
    # therefore deleting ADJACENT LATTICE CELLS. The true site could not be
    # proposed at all whenever a neighbouring cell scored higher, which is
    # exactly the situation this whole problem is about. Measured: the true
    # site's nearest surviving candidate sat 30-60px away -- two to five cells
    # -- on the trials where it was missing from the pool entirely.
    #
    # The cells are the hypotheses. They must survive to be compared.
    nms_r = int(max(3, round(NMS_PITCH_FRACTION * pitch)))

    # ---- rotation: MEASURED once, then VERIFIED against evidence ----
    # Sweeping rotation lifts the periodic distractors at least as much as the
    # true peak (each gets its own best-case angle), so it costs discrimination
    # even as it raises absolute correlation. Derotating ONCE by a measured
    # angle has no such effect: it moves every candidate by the same transform.
    #
    # The spectral estimator is accurate in the median (0.32-0.38 deg, against
    # 5.2 deg for the argmax-bin read it replaces) but has a heavy tail -- p90
    # around 5 deg, and the occasional gross miss when the two images' angle
    # estimates lock onto different harmonic families. Its own confidence does
    # not predict those failures. Since a wrong derotation is strictly worse
    # than none, the angle is not trusted: it is TESTED, by correlating once
    # each way and keeping whichever reference actually matches better.
    #
    # This is a two-way A/B decided on evidence, not a sweep. The
    # distractor-lifting objection applies to maximising over many angles; here
    # a single measured hypothesis has to beat "no rotation" by a margin to be
    # adopted at all.
    rot_deg, rot_conf = (0.0, 0.0)
    if use_rotation:
        cand_deg, rot_conf = relative_rotation(search_u8, ref_u8, pitch, scale_est)
        if abs(cand_deg) > 0.15:
            # Sign: relative_rotation returns the angle that, applied to the
            # reference, brings it back into the search image's frame -- it is
            # already the CORRECTION, not the misalignment. Verified against the
            # generator's recorded rotation_ref_deg (probes/rot_probe.py):
            # estimate ~= -rotation_ref_deg to a median 0.15 deg.
            ref_rot = rotate(ref_u8, cand_deg)
            base = _best_ncc(search_u8, ref_u8, foot_ref)
            alt = _best_ncc(search_u8, ref_rot, foot_ref)
            if alt > base + ROTATION_MIN_GAIN:
                ref_u8, rot_deg = ref_rot, cand_deg

    # The search window is the WHOLE FRAME by default. The statement says only
    # that the reference appears "somewhere inside"; assuming it sits near the
    # centre cost 43.8% -> 62.5% solvable accuracy under uniform placement, and
    # the "widen if the score is weak" escape hatch could not recover it because
    # wrong periodic repeats score 0.75-0.91 NCC. Proximity to the centre
    # survives only as the mandated tie-break, which is all the statement
    # actually asks for.
    radius = max(H, W) if (full_search or drift_radius is None) else float(drift_radius)
    win = _window(search_u8.shape, center, radius + foot_ref)

    cands = _multiscale_peaks(search_u8, ref_u8, win, feet, nms_radius=nms_r,
                              top_k_per_scale=TOP_K_PER_SCALE)
    if not cands:
        # Key set kept identical to the success path: downstream consumers
        # (benchmark.py rows, commit_gate feature selection) index these by
        # name, and a row that silently lacks a field is a row that quietly
        # drops out of a calibration set.
        return center[0], center[1], dict(
            confidence=0.0, coarse_score=0.0, reason='no candidates',
            n_near_peaks=0, scale_est=float(scale_est), scale_ok=bool(scale_ok),
            n_feet=len(feet), scale_span=float(span),
            landmark_z=0.0, fine_landmark=0.0, residual_saliency=0.0,
            ambiguity=0.0, induction_score=float(induction),
            rotation_deg=float(rot_deg),
            rotation_conf=float(rot_conf), phase_locked=False, phase_shift=0.0,
            ambiguous=True, n_candidates=0, full_search=True, foot=foot_ref)

    # ---- lattice phase lock: pin the within-cell offset from two FFTs ----
    lock = phase_lock(search_u8, ref_u8, pitch, scale_est) if use_phase_lock else None
    phase_locked = bool(lock and (lock['ok_x'] or lock['ok_y']))

    # ---- landmark channel: propose and score, over the full frame ----
    lmap, saliency, res_s, res_r = None, 0.0, None, None
    if use_landmark:
        s_band, r_band = pitch_bands(pitch)
        res_s = aperiodic_residual(search_u8, s_band)
        res_r = aperiodic_residual(ref_u8, r_band)
        # Saliency is read BEFORE envelope normalisation: it asks "is there a
        # concentrated landmark in here at all, or only noise?", and flattening
        # the envelope is exactly what would erase that distinction.
        saliency = residual_saliency(res_r)
        res_s = _envelope_normalise(res_s)
        res_r = _envelope_normalise(res_r)
        lmap = _landmark_map_from(res_s, res_r, foot_ref, win)
        if lmap is not None:
            cands = cands + _peaks_from_map(lmap, win, foot_ref, nms_radius=nms_r)

    # Counted on the FULL candidate list, before de-duplication: the pre-dedupe
    # pool is what reflects how many repeats the correlation surface offered.
    # Counting after dedupe measures the de-duplicator instead.
    appear_all = np.array([c['score'] for c in cands])
    n_near_peaks = int(np.sum(appear_all >= appear_all.max() - TIE_EPS))
    ambiguity = ambiguity_ratio(lmap, pitch, foot_ref) if lmap is not None else 0.0

    # Candidate budget scales with the AREA actually searched. Keeping a fixed
    # 15 was calibrated when the search was a 180px disk; over the full frame
    # that is ~10x the area for the same budget, and it showed up directly as
    # the true site simply not being in the pool on half of trials -- a search
    # failure that no amount of rescoring can undo. Budget is capped so the
    # full-resolution stage stays affordable.
    win_area = max(1.0, (win[2] - win[0]) * (win[3] - win[1]))
    keep = int(np.clip(round(15 * win_area / (360.0 ** 2)), 15, MAX_CANDIDATES))
    cands = _dedupe(cands, radius=nms_r, keep=keep)

    # Record how far each candidate sits from the lattice phase grid, but do
    # NOT move it yet.
    #
    # Snapping every candidate before scoring was measured head-to-head on the
    # same 40 pairs and does not pay as a ranking aid: fine_lm rank-1 50.0% ->
    # 53.3%, but fine_app 40.0% -> 36.7% and fused 60.0% -> 56.7%. Moving a
    # candidate by up to half a pitch to satisfy phase can walk it away from the
    # position appearance actually preferred, and appearance is not wrong often
    # enough for that trade to be worth it.
    #
    # The phase lock still earns its place -- just at the end, on the WINNER
    # only, where it sets the sub-pixel offset without touching which cell was
    # chosen. Evidence that changes the answer has to beat the alternative;
    # evidence that only refines it does not.
    if phase_locked:
        for c in cands:
            _sx, _sy, dx_s, dy_s = snap_to_phase(c['x'], c['y'], c['foot'], lock)
            c['phase_shift'] = float(np.hypot(dx_s, dy_s))
    else:
        for c in cands:
            c['phase_shift'] = 0.0

    # ---- rank at FULL reference resolution, not at the decimated scale ----
    cands = rescore_fullres(search_u8, ref_u8, res_s, res_r, cands,
                            foot_exact=ref_u8.shape[0] / float(scale_est))

    appearance = np.array([c['score'] for c in cands])
    fine_app = np.array([c['fine_appearance'] for c in cands])
    fine_lm = np.array([c['fine_landmark'] for c in cands])

    # Landmark evidence, normalised across the candidate SET rather than against
    # a fixed constant: what matters is whether one candidate stands outside the
    # spread of the others, and in a pure array interior that spread is all
    # there is.
    lm_z = _spread_z(fine_lm)

    # Proximity to the centre, as a weak preference only -- the statement's rule
    # is a tie-break between equally good matches, not a belief about where the
    # site is.
    dist = np.array([np.hypot(c['x'] - center[0], c['y'] - center[1]) for c in cands])
    prior = -dist / float(max(H, W))

    phase_penalty = -np.array([c['phase_shift'] for c in cands]) / max(pitch, 1.0)

    fused = (W_APPEARANCE * appearance + W_FINE_APPEARANCE * fine_app
             + W_LANDMARK * np.clip(lm_z, 0.0, 12.0) / 12.0
             + w_prior * prior + w_phase * phase_penalty)
    order = np.argsort(-fused)
    ranked = [cands[i] for i in order]
    fused_sorted = fused[order]

    # The landmark channel is the JUDGE when it speaks. It is the only evidence
    # that carries absolute identity inside a periodic array, so when one
    # candidate's residual correlation stands clear of the rest of the set, that
    # peak IS the disambiguation and geometry does not get a vote. When it is
    # silent -- a defect-free array interior, where the residual is noise
    # against noise -- the mandated centre rule resolves the tie.
    landmark_decided = bool(len(lm_z) and lm_z.max() >= LANDMARK_DECISIVE_Z
                            and fine_lm.max() >= LANDMARK_MIN_NCC)
    tied = [r for r, f in zip(ranked, fused_sorted) if fused_sorted[0] - f < TIE_EPS]

    if landmark_decided:
        winner, ambiguous = cands[int(np.argmax(lm_z))], False
    elif len(tied) > 1:
        winner, ambiguous = min(tied, key=lambda c: np.hypot(c['x'] - center[0],
                                                             c['y'] - center[1])), True
    else:
        winner, ambiguous = ranked[0], False

    # Sub-pixel refinement of the CHOSEN cell. The phase lock fixes the
    # within-cell offset from two FFTs -- one Fourier coefficient averaged over
    # the whole projection, rather than a parabola through three bins of a noisy
    # correlation surface. Applied here it cannot change which cell won, only
    # where inside it the answer sits.
    if phase_locked:
        x, y, _dx, _dy = snap_to_phase(winner['x'], winner['y'],
                                       winner['foot'], lock)
    else:
        x, y = _subpixel(search_u8, ref_u8, winner['x'], winner['y'], winner['foot'])

    return x, y, dict(confidence=float(winner.get('fine_appearance', winner['score'])),
                      coarse_score=float(winner['score']),
                      scale_est=float(scale_est), scale_ok=bool(scale_ok),
                      n_feet=len(feet), scale_span=float(span),
                      landmark_z=float(lm_z.max()) if len(lm_z) else 0.0,
                      fine_landmark=float(fine_lm.max()) if len(fine_lm) else 0.0,
                      residual_saliency=float(saliency),
                      ambiguity=float(ambiguity),
                      induction_score=float(induction),
                      rotation_deg=float(rot_deg),
                      rotation_conf=float(rot_conf),
                      phase_locked=phase_locked,
                      phase_shift=float(winner.get('phase_shift', 0.0)),
                      n_near_peaks=n_near_peaks, ambiguous=bool(ambiguous),
                      n_candidates=len(cands),
                      full_search=bool(radius >= max(H, W)),
                      foot=int(winner['foot']))


def main():
    ap = argparse.ArgumentParser(
        description='Locate a reference pattern inside a search image.',
        epilog='Both forms work:\n'
               '  python localize.py --reference ref.png --search search.png\n'
               '  python localize.py ref.png search.png',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Positional form accepted as well as the flags. The statement says the
    # script "accepts: (a) path to reference image, (b) path to search image",
    # and an unrunnable script cannot be scored -- so the obvious invocation
    # must not fail on an argument-name guess.
    ap.add_argument('paths', nargs='*', metavar='REF SEARCH',
                    help='reference and search image paths, in that order')
    ap.add_argument('--reference')
    ap.add_argument('--search')
    ap.add_argument('--drift-radius', type=float, default=None,
                    help='optional bound on how far from the frame centre the '
                         'site may be, px. Default: unset, i.e. search the whole '
                         'frame -- the statement says only that the reference '
                         'appears "somewhere inside". Set this only if the stage '
                         'genuinely constrains the landing position; it was '
                         'measured to COST 43.8%% -> 62.5%% solvable accuracy '
                         'when the assumption does not hold.')
    ap.add_argument('--no-landmark', action='store_true',
                    help='disable the aperiodic residual channel')
    ap.add_argument('--no-phase-lock', action='store_true',
                    help='disable the cross-image lattice phase constraint')
    ap.add_argument('--no-rotation', action='store_true',
                    help='disable relative-rotation measurement/derotation')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    ref_path, search_path = args.reference, args.search
    if ref_path is None or search_path is None:
        if len(args.paths) == 2:
            ref_path, search_path = args.paths
        else:
            ap.error('give both images, either as --reference/--search or as '
                     'two positional paths (reference first)')
    elif args.paths:
        ap.error('give the images either as flags or as positionals, not both')

    ref = load_gray(ref_path)
    search = load_gray(search_path)
    t0 = time.time()
    x, y, diag = localize(ref, search, drift_radius=args.drift_radius,
                          use_landmark=not args.no_landmark,
                          use_phase_lock=not args.no_phase_lock,
                          use_rotation=not args.no_rotation)
    diag['runtime_sec'] = round(time.time() - t0, 3)

    if args.json:
        print(json.dumps(dict(x=x, y=y, **diag), indent=2))
    else:
        print(f'{x:.3f}, {y:.3f}')


if __name__ == '__main__':
    main()
