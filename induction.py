"""
Spatial induction: force the lattice to prove its own geometry.

The lattice pitch is measured from the first autocorrelation peak of a
projection profile (`lattice.estimate_lattice`). That measurement is accurate
in the median -- 0.04% against the generator's recorded pitch -- but it fails
SILENTLY on a minority of frames, and nothing downstream can tell:

    search pitch within 15% of truth      87/100
    locked onto a WRONG lag               13/100   (measured 4.0 where the
                                                    truth is 8-10, etc.)

`estimate_scale`'s own `ok` flag catches five of those thirteen. The other
eight propagate: a wrong search pitch drags the *conditioned* reference band
to the wrong place, so `scale = pitch_ref / pitch_search` lands on the 8.5
clip floor, and every stage downstream inherits a footprint that is ~9% wrong.

THE TEST. Proof by induction, run over space rather than over the integers:

    base case      the profile autocorrelation has a peak at lag 0
    inductive step if pitch p is the true period, lag n*p must ALSO be a
                   peak, for every n
    failure        if any n*p lands in a valley, reject the hypothesis

Measured separation on the 100-pair primary split, at the shipped N_STEPS=2:

    correct pitch  (n=87)   median score  +1.873   min  +0.299
    wrong pitch    (n=13)   median score  -0.007   max  -0.001

The two groups do not overlap: the worst correct pitch scores 0.300 above the
best wrong one. So the SIGN of the score is the decision -- 13/13 caught at
zero false alarms -- and no threshold is fitted, which is the only reason this
is quotable on a set this small.

WHY ONLY TWO STEPS. The intuition for this test is usually that a small pitch
error accumulates -- 2% wrong compounds to a whole cell by step 10. That is not
the failure mode that actually occurs here. The observed errors are gross
(true/measured ratios of 1.5, 1.75, 2.0, 2.25, 2.5), so the FIRST stepping
stone already lands in a valley and later steps only re-measure what step 1
established. Adding them is not free -- the autocorrelation envelope decays
with lag, so each extra step gives a correct pitch another chance to dip below
zero:

    n_steps   caught   false alarms   margin (worst correct - best wrong)
       2       13/13        0             +0.300
       3       13/13        1             -0.015
       4       13/13        1             -0.296
       6       13/13        2             -0.508
       8       13/13        2             -0.584

Two is the only setting where the groups separate cleanly. It also keeps the
check at two 1-D autocorrelations -- under 10ms per frame.

WHY IT IS NOT WIRED TO THE SCALE BRACKET. The obvious use is "induction fails
-> widen the magnification bracket." That was implemented and measured, and it
does not pay: gating recovers much of what tight bracketing costs (solvable
<=1px 26.2% -> 37.5%) but never beats simply not bracketing tightly at all
(42.5%), on either an 8.5-11.5 prior or a deliberately wide 6-16 one. Repair
by integer multiples is worse still -- the observed ratios are not integers,
so the true pitch is not in {p, 2p, 3p, 4p}, and the repair nets +1 pair.

WHAT IT IS FOR. The score predicts LOCALIZATION outcome, with the sign
inverted from intuition (n=100, point-biserial r = -0.455, p = 2e-6):

    induction FAILS  (n=13)   localization accuracy  84.6%
    induction PASSES (n=87)   localization accuracy  32.2%
                              base rate              39.0%

A lattice that fails this test is an IRREGULAR lattice -- it contains array
boundaries, dropped or doubled vias, broken periodicity. That aperiodic
content is precisely what carries absolute identity inside a repeating array,
so a frame that fails induction is a frame that can be localized. A perfectly
inductive lattice is a perfectly ambiguous one.

That makes this a solvability signal computable from the search image ALONE,
before any correlation is run, and it is partly independent of the signals the
commit gate already has (n=100, success = within 15px):

    n_near_peaks <= 5   (existing)   30 sites   86.7%
    induction < 0       (this)       13 sites   84.6%
    either                           35 sites   82.9%
    both                              8 sites  100.0%

Only eight of the thirteen overlap, so this is added to CANDIDATE_FEATURES in
commit_gate.py as evidence in its own right rather than as a replacement. Note
what the union does and does not buy: coverage 30 -> 35 sites for 3.8 points of
precision, and an eight-site intersection that is perfect on this split. The
intersection is the interesting tier and the one most likely to be sample
noise at n=8; treat it as a hypothesis to re-measure, not a result.

Reproduce all four claims:

    python probes/induction_probe.py --dataset dataset_primary \
                                     --results primary_results.json

KNOWN BLIND SPOT. Induction is one-sided: it rejects pitches SHORTER than the
truth and cannot reject longer ones. Every integer multiple of the true pitch
also lands on peaks, so a hypothesis of 2p passes as cleanly as p (verified on
a synthetic lattice: true 12, hypothesis 24 scores +0.0001, hypothesis 6 scores
-0.93). That is acceptable here only because the observed failures are all in
the rejectable direction -- the estimator locks SHORT, onto lags of 4-6 where
the truth is 8-14. A test set whose estimator overshoots would need the
complementary check (does a peak exist at p/2?) and this module would not see
the failure.

SMALL ERRORS ARE CAUGHT BY SLOPE, NOT SIGN. For a pitch wrong by a few percent
the stepping stones decay rather than invert -- true 12 against a hypothesis of
13 gives steps [1.528, 0.861, 0.033] where a correct pitch gives a flat
[1.854, 1.832, 1.809]. The minimum is still positive at two steps, so such
errors PASS. Extending N_STEPS or scoring the slope across steps would catch
them; neither is done here because errors of that size are not what the
estimator actually makes on this distribution, and an unmeasured refinement is
not worth the coefficients it would add.
"""
import numpy as np
import cv2

from lattice import lattice_angle, rotate

# Steps of the induction. Two, chosen on the measured separation margin -- see
# module docstring.
N_STEPS = 2

# A frame whose lattice is weaker than this has no periodic model worth
# testing; the score is returned as NaN rather than as a confident pass.
MIN_PROFILE_ENERGY = 1e-6

# De-rotation is OFF by default, and that is a measured decision rather than an
# oversight. A lattice tilted by t degrees smears the projection profile by
# H*sin(t), which at 1000px and 2 deg is 35px -- several pitches, enough to
# erase the peaks this test reads -- so aligning first looks obviously correct.
# It is not, on this data:
#
#   derotate=False   13/13 wrong pitches caught, 0 false alarms
#   derotate=True    11/13 caught, and the wrong-pitch group stops separating
#
# The generator never rotates the search canvas, yet lattice_angle reports
# tilts up to 3.1 deg at 0.88 strength on those same frames, and 45% of them
# exceed DEROTATE_ABOVE_DEG. Every one of those rotations is spurious, and each
# costs an INTER_LINEAR resample that damps exactly the high-frequency lattice
# content the autocorrelation peaks are made of. The correction was buying
# noise and paying for it in resolution.
#
# Enable it only for a test set that genuinely rotates the search canvas, and
# re-measure the separation above before trusting the score if you do.
DEROTATE = False
DEROTATE_ABOVE_DEG = 0.25

# Frequency band for the orientation estimate. Fixed and wide -- it must not be
# derived from the pitch hypothesis under test; see align_to_lattice.
ANGLE_BAND = (5.0, 40.0)


def _profile(img_u8, axis):
    """1-D projection profile across the lattice, low-frequency trend removed.

    Field shading and beam charging put a slow ramp under the profile that is
    much larger than the lattice modulation; subtracting a heavily blurred copy
    removes it without touching the pitch-scale structure the test reads.
    """
    prof = img_u8.astype(np.float64).mean(axis=axis)
    trend = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 25).ravel()
    return prof - trend


def _acf(prof, max_lag):
    """Normalised autocorrelation of the profile, lags 0..max_lag."""
    p = prof - prof.mean()
    n = len(p)
    ac = np.correlate(p, p, mode='full')[n - 1:]
    if not np.isfinite(ac[0]) or ac[0] <= MIN_PROFILE_ENERGY:
        return None
    return (ac / ac[0])[:max_lag + 1]


def align_to_lattice(img_u8, band=ANGLE_BAND, min_strength=0.30):
    """De-rotate so the lattice runs along the image axes. Returns the input
    unchanged if the tilt is negligible, weak, or unmeasurable.

    `band` is deliberately NOT derived from the pitch hypothesis under test.
    Conditioning it on that pitch couples the alignment to the very error this
    module exists to detect: on the thirteen frames whose pitch is wrong, a
    pitch-derived band hands lattice_angle the wrong frequencies, it returns a
    spurious tilt, and de-rotating by it smears the profile enough to hide the
    failure. Measured -- with a pitch-derived band the wrong-pitch group's
    median score rises from -0.68 to -0.012 and detection falls from 13/13 to
    11/13. A fixed wide band keeps the alignment independent of the hypothesis.

    `min_strength` exists for the same reason: an unconfident angle is worse
    than no angle, because rotating by it costs a resample and buys noise.
    """
    try:
        ang, strength = lattice_angle(img_u8, band)
    except Exception:
        return img_u8  # orientation is an optimisation here, not a precondition
    if not np.isfinite(strength) or strength < min_strength:
        return img_u8
    # The wave-vector angle of a square lattice is only defined modulo 90 deg;
    # fold into (-45, 45] so we correct the small residual tilt rather than
    # rotating by a whole quadrant.
    ang = (ang + 45.0) % 90.0 - 45.0
    if not np.isfinite(ang) or abs(ang) <= DEROTATE_ABOVE_DEG:
        return img_u8
    return rotate(img_u8, ang)


def _peakiness(ac, lag, pitch):
    """How far the autocorrelation at `lag` stands above its own neighbourhood.

    Positive iff `lag` is a local maximum relative to the two half-pitch
    offsets either side of it -- i.e. iff the stepping stone landed ON a cell
    rather than in the valley between two. Compared against its immediate
    neighbourhood rather than against a fixed constant because the
    autocorrelation envelope decays with lag, so an absolute threshold would
    reject late steps purely for being late.
    """
    idx = np.arange(len(ac))
    here = float(np.interp(lag, idx, ac))
    half = pitch / 2.0
    left = float(np.interp(max(1.0, lag - half), idx, ac))
    right = float(np.interp(min(len(ac) - 1.0, lag + half), idx, ac))
    return here - 0.5 * (left + right)


def induction_score(img_u8, pitch, n_steps=N_STEPS, axis=0, derotate=DEROTATE):
    """Test pitch hypothesis `pitch` on one axis. -> (score, per_step_scores).

    score > 0  every stepping stone landed on a lattice peak: the pitch is
               self-consistent across the frame.
    score < 0  at least one landed in a valley: reject the hypothesis.
    NaN        no periodic model to test (flat or featureless frame).

    The score is the MINIMUM over steps, not the mean: induction is a
    conjunction, and one failed step falsifies the hypothesis no matter how
    well the others did. Averaging would let the passing steps outvote the one
    that carries the refutation.
    """
    if not np.isfinite(pitch) or pitch <= 1.0:
        return np.nan, []

    if derotate:
        img_u8 = align_to_lattice(img_u8)

    prof = _profile(img_u8, axis)
    max_lag = int(pitch * (n_steps + 1)) + 4
    if max_lag >= len(prof) - 2:
        return np.nan, []
    ac = _acf(prof, max_lag)
    if ac is None:
        return np.nan, []

    steps = [_peakiness(ac, pitch * n, pitch) for n in range(1, n_steps + 1)]
    if not steps or not np.all(np.isfinite(steps)):
        return np.nan, []
    return float(np.min(steps)), [float(s) for s in steps]


def induction_evidence(search_u8, pitch, n_steps=N_STEPS, derotate=DEROTATE):
    """Best-of-both-axes induction score for a frame. -> float or NaN.

    Taken as the MAXIMUM over the two axes: a DRAM grid is periodic on both,
    but a FinFET fin field is periodic along the fins and nearly featureless
    across them, so requiring both axes to pass would reject every FinFET frame
    on the strength of the axis that carries no lattice. The question this
    evidence answers is "does a self-consistent periodic model fit this frame at
    all", and one axis suffices to answer yes.
    """
    # De-rotate at most once, not once per axis: the alignment is a property of
    # the lattice, not of the axis being projected, and the spectral angle
    # estimate is the expensive half of this check.
    aligned = align_to_lattice(search_u8) if derotate else search_u8
    scores = []
    for axis in (0, 1):
        s, _ = induction_score(aligned, pitch, n_steps=n_steps, axis=axis,
                               derotate=False)
        if np.isfinite(s):
            scores.append(s)
    return float(max(scores)) if scores else float('nan')
