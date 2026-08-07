"""
Lattice geometry sensor.

The core insight this module encodes: in a periodic array, the pixels carry
two *separable* kinds of information.

  1. The PERIODIC component (pitch, orientation, phase). It is identical at
     every lattice cell, so it can never identify *which* cell you are in --
     but it does pin down the magnification exactly (pitch_ref / pitch_search
     IS the unknown ~10x scale factor) and it constrains position modulo the
     pitch.

  2. The APERIODIC RESIDUAL (array boundaries, dropped/doubled vias, gate
     crossings, i.e. every actual landmark). This is the *only* channel that
     carries absolute identity.

A raw NCC or a CNN embedding over the whole patch mixes the two, and since
the periodic component dominates the energy by an order of magnitude, the
identity-bearing residual is drowned out. Separating them with a spectral
notch turns "which of 400 identical cells is this?" into a question that has
visible evidence.

Everything here is deterministic signal processing -- no learned weights.
"""
import numpy as np
import cv2


# ---------------------------------------------------------------- orientation

def _spectrum(img_f):
    """Centered magnitude spectrum of a Hann-windowed, mean-removed image."""
    n = img_f.shape[0]
    w = np.outer(np.hanning(img_f.shape[0]), np.hanning(img_f.shape[1]))
    F = np.fft.fftshift(np.fft.fft2((img_f - img_f.mean()) * w))
    return np.abs(F)


def _band_mask(shape, pitch_range):
    """Bins whose spatial period falls inside pitch_range (px).

    Returns (mask, fy, fx) where fy/fx are spatial FREQUENCIES in cycles/px,
    not raw bin offsets. The distinction matters as soon as the image is not
    square: bin offset k along an axis of length N is k/N cycles/px, so a
    non-square image has different cycles-per-bin on the two axes. Treating
    bins as isotropic (the previous `period = h / r`) silently corrupts the
    band mask, and with it every pitch, scale and orientation estimate
    downstream. Test images are not guaranteed square.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    # Built by BROADCASTING two 1-D axes rather than materialising np.mgrid.
    # mgrid allocates two full int64 planes and every derived expression another
    # float64 plane; on the zero-padded spectra used for angle estimation
    # (4000x4000) that is ~750MB of traffic per call and dominated the whole
    # pipeline's runtime -- 12s of a 14s pair. The broadcast form computes the
    # same thing in float32 with two real planes.
    fy = ((np.arange(h, dtype=np.float32) - cy) / np.float32(h))[:, None]
    fx = ((np.arange(w, dtype=np.float32) - cx) / np.float32(w))[None, :]
    f = np.hypot(fy, fx)
    with np.errstate(divide='ignore'):
        period = 1.0 / np.maximum(f, 1e-12)
    mask = (period >= pitch_range[0]) & (period <= pitch_range[1])
    # Broadcast to full shape as VIEWS so callers can index them elementwise
    # without paying for the materialisation.
    return mask, np.broadcast_to(fy, (h, w)), np.broadcast_to(fx, (h, w))


def dominant_orientation(img_u8, pitch_range):
    """Angle (deg) of the dominant lattice wave-vector. Rotation-robust:
    read straight off the 2-D power spectrum, so the reference image's ~1.3
    deg capture rotation is measured rather than ignored."""
    img_f = img_u8.astype(np.float32)
    mag = _spectrum(img_f)
    band, fy, fx = _band_mask(mag.shape, pitch_range)
    m = np.where(band, mag, 0.0)
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    return float(np.degrees(np.arctan2(fy[iy, ix], fx[iy, ix])))


def _autocorr_pitch(prof, lo, hi):
    """First autocorrelation peak of a 1-D projection profile, parabolically
    refined to sub-pixel. Sub-pixel matters: at ref scale the pitch is ~120px
    and a 1px error is a 1% magnification error, which is what makes the
    scale factor recoverable to a fraction of a percent."""
    p = prof - prof.mean()
    if np.allclose(p, 0):
        return None, 0.0
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(max(2, lo)), int(min(len(ac) - 2, hi))
    if hi <= lo:
        return None, 0.0
    k = lo + int(np.argmax(ac[lo:hi]))
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    denom = a - 2 * b + c
    delta = 0.0 if abs(denom) < 1e-9 else 0.5 * (a - c) / denom
    return float(k + np.clip(delta, -1, 1)), float(b)


def estimate_lattice(img_u8, pitch_range):
    """-> dict(pitch, theta_deg, quality). Rotates the image so the lattice
    lines are axis-aligned, then measures the pitch from the projection
    profile's autocorrelation (far more precise than the FFT bin grid)."""
    theta = dominant_orientation(img_u8, pitch_range)
    h, w = img_u8.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    rot = cv2.warpAffine(img_u8, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    # crop the valid centre so rotation fill does not pollute the profile
    m = int(0.15 * h)
    core = rot[m:h - m, m:w - m].astype(np.float32)

    best = (None, -1.0)
    for axis in (0, 1):                      # try both projections; the lattice
        prof = core.mean(axis=axis)          # family lands on one of them
        pitch, q = _autocorr_pitch(prof, pitch_range[0], pitch_range[1])
        if pitch is not None and q > best[1]:
            best = (pitch, q)
    pitch, quality = best
    if pitch is None:
        pitch = float(np.mean(pitch_range))
        quality = 0.0
    return dict(pitch=float(pitch), theta_deg=float(theta), quality=float(quality))


def rotate(img_u8, deg):
    """Rotate about the image centre, reflecting at the border."""
    if abs(deg) < 1e-3:
        return img_u8
    h, w = img_u8.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(deg), 1.0)
    return cv2.warpAffine(img_u8, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


# ------------------------------------------------------------ relative rotation
#
# History, because the fix only makes sense against what it replaces:
#
#   `dominant_orientation` reads the ARGMAX BIN of the 2-D spectrum, so its
#   angular precision is set by how far the lattice fundamental sits from DC:
#   ~N/pitch bins, i.e. ~degrees(pitch/N). On the SEARCH image (pitch ~9px in
#   1000px) that is ~111 bins -> ~0.5 deg, fine. On the REFERENCE the pitch is
#   ~10x larger, so the fundamental sits only ~11 bins out and the angle
#   quantises to ~5.2 deg -- four times coarser than the ~1.2 deg capture
#   rotation being measured. Differencing the two angles returned exactly 0.0 on
#   37 of 40 pairs. The matcher therefore SWEPT rotation instead, and the sweep
#   made things worse (40.0% vs 46.0%), for a sound reason: maximising over a
#   nuisance parameter lifts the periodic DISTRACTORS at least as much as the
#   true peak, since each distractor gets its own best-case angle while the true
#   peak already sits near its optimum. So the sweep was pinned at a single
#   angle and the rotation was, in effect, ignored.
#
# That diagnosis was right about the sweep and wrong about the limit. The 5.2
# deg figure is an argmax-bin artefact, not an information bound, and it is
# fixed by not reading the argmax bin:
#
#   1. ZERO-PAD before the FFT. The spectrum of a finite image is not
#      band-limited to its bin grid; padding interpolates it, so the peak can be
#      located between bins.
#   2. USE EVERY HARMONIC, not just the fundamental. The k-th harmonic sits at
#      k x the radius, so the same absolute bin uncertainty is k times finer in
#      ANGLE. Averaging over harmonics weighted by magnitude therefore beats the
#      fundamental alone by roughly the mean harmonic order.
#   3. CENTROID, don't argmax. A magnitude-weighted centroid over each peak's
#      immediate neighbourhood is sub-bin by construction.
#
# The rotation is worth recovering: derotating by the true angle lifts the NCC
# at the true site by +0.077 on average (+11.3%), which is ~3.8x the ~0.02
# window inside which periodic near-ties sit.


def _harmonic_angles(img_u8, pitch_range, pad_factor=4, max_peaks=24,
                     max_padded=1024):
    """Sub-bin (angle, magnitude, radius) of each lattice harmonic.

    `max_padded` caps the padded transform size. Padding and the local centroid
    below buy the same thing -- sub-bin peak localisation -- and measured against
    the recorded truth they turn out to be redundant, with padding actively
    worse past a point:

        max_padded   median |err|   p90     cost
        1024 (=none)   0.38 deg     5.02    0.77 s/pair
        2048           0.32 deg     8.75    2.41 s/pair
        3072           0.34 deg     8.45    5.85 s/pair
        4096           1.21 deg    30.19   10.73 s/pair

    So the centroid was doing the work all along, and heavy padding broadens the
    peaks enough that the 5x5 local-max test starts selecting sidelobes. Capped
    at 1024, which for a 1000px image means no padding at all.
    """
    img_f = img_u8.astype(np.float32)
    h, w = img_f.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    a = (img_f - img_f.mean()) * win
    pad = max(1, min(pad_factor, max_padded // max(h, w)))
    H, W = h * pad, w * pad
    F = np.fft.fftshift(np.fft.fft2(a, s=(H, W)))
    mag = np.abs(F).astype(np.float32)

    band, fy, fx = _band_mask(mag.shape, pitch_range)
    bg = cv2.blur(mag, (11, 11)) + 1e-6
    local_max = (mag >= cv2.dilate(mag, np.ones((5, 5), np.uint8)))
    cand = local_max & band & ((mag / bg) > 2.0)
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        return []
    order = np.argsort(-mag[ys, xs])[:max_peaks]

    out = []
    r = 2
    for iy, ix in zip(ys[order], xs[order]):
        y0, y1 = max(0, iy - r), min(mag.shape[0], iy + r + 1)
        x0, x1 = max(0, ix - r), min(mag.shape[1], ix + r + 1)
        patch = mag[y0:y1, x0:x1].astype(np.float64)
        tot = patch.sum()
        if tot <= 0:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        cy_c = float((patch * gy).sum() / tot)
        cx_c = float((patch * gx).sum() / tot)
        # convert the sub-bin centroid to a frequency, then to an angle
        fyy = (cy_c - mag.shape[0] // 2) / float(mag.shape[0])
        fxx = (cx_c - mag.shape[1] // 2) / float(mag.shape[1])
        rad = float(np.hypot(fyy, fxx))
        if rad <= 0:
            continue
        out.append((float(np.degrees(np.arctan2(fyy, fxx))),
                    float(mag[iy, ix]), rad))
    return out


def _circular_mean_deg(angles_deg, weights, period=90.0):
    """Weighted circular mean, modulo `period`.

    A lattice is symmetric under 90 deg rotation (and a 1-D fin field under
    180), so raw angles are only defined modulo that symmetry; averaging them
    linearly would put a +44 deg and a -44 deg harmonic at 0 instead of at the
    +-45 they actually agree on. Mapping onto the unit circle at the symmetry
    frequency handles the wraparound correctly.
    """
    a = np.radians(np.asarray(angles_deg, dtype=np.float64) * (360.0 / period))
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return 0.0, 0.0
    c, s = float((w * np.cos(a)).sum()), float((w * np.sin(a)).sum())
    mean = np.degrees(np.arctan2(s, c)) * (period / 360.0)
    strength = float(np.hypot(c, s) / w.sum())
    return float(mean), strength


def lattice_angle(img_u8, pitch_range, period=90.0):
    """Orientation of the lattice, to sub-bin precision. -> (deg, strength)."""
    peaks = _harmonic_angles(img_u8, pitch_range)
    if not peaks:
        return 0.0, 0.0
    angles = [p[0] for p in peaks]
    # Weight by magnitude AND by radius: a harmonic further from DC pins the
    # angle more tightly, in direct proportion to its radius.
    weights = [p[1] * p[2] for p in peaks]
    return _circular_mean_deg(angles, weights, period=period)


def relative_rotation(search_u8, ref_u8, pitch_search, scale,
                      period=90.0, max_deg=15.0):
    """Rotation of the REFERENCE relative to the SEARCH image, in degrees.

    Both angles are measured with the sub-bin, multi-harmonic estimator, so the
    reference's ~11-period field is no longer the accuracy bottleneck it was for
    an argmax-bin read. -> (deg, confidence in [0, 1]).
    """
    p = float(pitch_search)
    s_band = (max(2.0, p * 0.45), p * 3.0)
    r_band = (p * scale * 0.45, p * scale * 3.0)
    a_s, k_s = lattice_angle(search_u8, s_band, period=period)
    a_r, k_r = lattice_angle(ref_u8, r_band, period=period)
    conf = float(min(k_s, k_r))
    d = (a_r - a_s + period / 2.0) % period - period / 2.0
    if abs(d) > max_deg:
        # Beyond the plausible capture misalignment: more likely the two
        # estimates locked onto different harmonic families than that the stage
        # actually rotated this far. Report no rotation and low confidence
        # rather than derotating by a number we do not believe.
        return 0.0, 0.0
    return float(d), conf


def estimate_scale(search_u8, ref_u8, fallback=10.0, bounds=(8.5, 11.5)):
    """The magnification is not searched for -- it is *measured*, as the ratio
    of the two lattice pitches.

    The search-image pitch is measured first and unconstrained; it then
    *conditions* the reference measurement -- the reference pitch is only
    looked for in the lag window [8.5, 11.5] x pitch_search implied by the
    nominal ~10x optics. Without that conditioning the reference
    autocorrelation occasionally locks onto the 2nd harmonic; with it, the
    physical prior and the measurement disambiguate each other. This is the
    same 'evidence constrains evidence' move the whole reranker is built on,
    applied one level down.
    """
    lat_s = estimate_lattice(search_u8, (5.0, 40.0))
    ps = lat_s['pitch']
    if not (2.0 < ps < 45.0):
        return float(fallback), lat_s, None, False
    lat_r = estimate_lattice(ref_u8, (ps * bounds[0], ps * bounds[1]))
    s = lat_r['pitch'] / ps
    ok = (bounds[0] + 0.05) <= s <= (bounds[1] - 0.05) and lat_r['quality'] > 0.05
    return float(np.clip(s, *bounds)), lat_s, lat_r, bool(ok)


def scale_uncertainty(lat_s, lat_r, scale, floor=0.04, ceil=0.40):
    """How far the magnification search actually needs to bracket the estimate.

    The pitch ratio is measured to a median 0.14% on this generator, but the
    consumer used to sweep +/-3.5% around it -- twenty-five times wider than
    the measurement error. That is not merely wasted compute: every extra scale
    injects another six near-tied candidates into the pool, inflating the very
    tie-count the commit gate is calibrated against. The sweep was manufacturing
    the ambiguity the gate then measured.

    Both pitches are parabolically-refined autocorrelation peaks, so their
    precision tracks how sharp the peak was; `quality` is that peak height,
    normalised. Relative errors add in quadrature through the ratio.
    """
    def _sigma_rel(lat):
        if lat is None:
            return 0.02
        q = float(np.clip(lat.get('quality', 0.0), 0.0, 1.0))
        # ~0.1px on a clean peak, degrading as the peak flattens
        sigma_px = 0.1 + 2.0 * (1.0 - q) ** 2
        return sigma_px / max(float(lat.get('pitch', 1.0)), 1e-6)

    rel = float(np.hypot(_sigma_rel(lat_s), _sigma_rel(lat_r)))
    return float(np.clip(3.0 * rel * scale, floor, ceil))


def foot_bracket(ref_size, scale_est, span, min_foot=20):
    """Distinct integer template footprints implied by a scale bracket.

    The consumer computes `foot = round(ref_size / scale)` and then resizes to
    that integer, so two scales that round to the same footprint do *identical*
    work and produce duplicate candidates. Sweeping float scales therefore
    over-samples the centre of the bracket and leaves its edges to chance.
    Enumerating the integer footprints instead is exactly non-redundant, and it
    makes the sweep width self-describing: len() IS the number of real hypotheses.
    """
    lo_s, hi_s = max(scale_est - span, 1e-3), scale_est + span
    lo_f = max(min_foot, int(np.floor(ref_size / hi_s)))
    hi_f = max(min_foot, int(np.ceil(ref_size / lo_s)))
    return list(range(lo_f, hi_f + 1))


# ------------------------------------------------------------ aperiodic split

def _find_spectral_peaks(mag, band, max_peaks=90, prominence=3.0):
    """Sharp, prominent local maxima inside the band = lattice harmonics.
    Data-driven rather than analytic, so it works for the 1-D fin lattice and
    the 2-D DRAM lattice without being told which one it is looking at."""
    bg = cv2.blur(mag, (11, 11)) + 1e-6
    prom = mag / bg
    local_max = (mag >= cv2.dilate(mag, np.ones((5, 5), np.uint8)))
    cand = local_max & band & (prom > prominence)
    ys, xs = np.nonzero(cand)
    if len(ys) == 0:
        # Must match the two-value contract below: the sole caller unpacks this
        # into (peaks, n_found), so returning a bare [] raises at the unpack --
        # on exactly the flat-spectrum inputs where the caller most needs to
        # degrade gracefully rather than crash.
        return [], 0
    order = np.argsort(-mag[ys, xs])[:max_peaks]
    return list(zip(ys[order], xs[order])), int(len(ys))


def aperiodic_residual(img_u8, pitch_range, notch_radius=3.0, blur_sigma=1.2,
                       diagnostics=None):
    """Spectrally subtract the periodic lattice; what survives is landmark.

    Notches the lattice harmonics (and DC) out of the spectrum and inverse
    transforms. Repetitive grid -> ~0 everywhere. Array boundary, dropped or
    doubled via, gate crossing -> survives with full contrast. Sensor noise
    also survives, but noise is independent between the two captures, so it
    contributes ~0 to a normalised cross-correlation between residuals.
    """
    img_f = img_u8.astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(img_f - img_f.mean()))
    mag = np.abs(F)
    band, _fy, _fx = _band_mask(mag.shape, pitch_range)
    peaks, n_found = _find_spectral_peaks(mag, band)
    if diagnostics is not None:
        # A cap-hit means harmonics were silently left un-notched, so lattice
        # energy leaks into the "landmark only" residual. Surfacing the count
        # is what separates "clean residual" from "we ran out of notches".
        diagnostics['n_harmonics_found'] = n_found
        diagnostics['n_harmonics_notched'] = len(peaks)
        diagnostics['harmonic_cap_hit'] = bool(n_found > len(peaks))

    h, w = mag.shape
    keep = np.ones((h, w), dtype=np.float32)
    rad = int(np.ceil(4 * notch_radius))
    oy, ox = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float32)

    def _notch(py, px, sigma):
        """Apply a Gaussian hole locally -- the notch is negligible beyond a
        few sigma, so touching the whole plane per peak is wasted work."""
        y0, y1 = max(0, py - rad), min(h, py + rad + 1)
        x0, x1 = max(0, px - rad), min(w, px + rad + 1)
        gy0, gx0 = y0 - (py - rad), x0 - (px - rad)
        d2 = (oy[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)] ** 2
              + ox[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)] ** 2)
        keep[y0:y1, x0:x1] *= 1.0 - np.exp(-d2 / (2.0 * sigma ** 2))

    for (py, px) in peaks:
        _notch(int(py), int(px), notch_radius)
    _notch(h // 2, w // 2, 2.0)   # DC / illumination gradient

    res = np.real(np.fft.ifft2(np.fft.ifftshift(F * keep))).astype(np.float32)
    if blur_sigma > 0:
        res = cv2.GaussianBlur(res, (0, 0), blur_sigma)
    return res


# --------------------------------------------------------------- phase lock

def fundamental_phase(img_u8, pitch, axis):
    """Absolute phase (px) of the periodic line family along `axis`, plus a
    coherence in [0, 1] saying how line-like that axis really is.

    Projects the image onto the axis, then reads the argument of the SINGLE DFT
    coefficient at the known pitch. One coefficient, so the estimate averages
    over the whole 1000-sample projection instead of reading three adjacent
    bins of a noisy correlation surface -- which is what makes it hold up under
    the heavier noise the test set is promised to carry.

    axis=0 collapses rows, giving the phase of the family that VARIES IN X.
    """
    prof = img_u8.astype(np.float64).mean(axis=axis)
    n = len(prof)
    if n < 8 or pitch <= 1.0 or pitch >= n:
        return 0.0, 0.0
    p = prof - prof.mean()
    t = np.arange(n)
    c = np.sum(p * np.exp(-2j * np.pi * (n / pitch) * t / n))
    # coherence: how much of the projection's energy this one line frequency
    # explains. A fin/word-line family scores high; an axis with no periodic
    # family at all (the along-fin direction of a FinFET field) scores ~0.
    denom = np.sqrt(np.sum(p ** 2) * n / 2.0) + 1e-12
    coherence = float(np.clip(np.abs(c) / denom, 0.0, 1.0))
    phase = float((-np.angle(c) * pitch / (2 * np.pi)) % pitch)
    return phase, coherence


def phase_lock(search_u8, ref_u8, pitch_search, scale, min_coherence=0.10):
    """Pin the reference crop's ORIGIN, modulo the pitch, from two FFTs.

    Both images depict one globally coherent lattice, so each carries an
    absolute lattice phase and the difference between them fixes where the
    reference sits -- modulo one pitch -- with no correlation at all.

        origin_x  ==  phase_search_x - phase_ref_x / scale   (mod pitch)

    Measured on this generator: median residual 0.42px against the recorded
    truth, p90 0.88px. That is a sub-pixel position estimate obtained before a
    single template has been slid, and it is evidence of a completely different
    kind from appearance -- it cannot be fooled by a well-correlating wrong
    repeat, because every repeat satisfies it equally and it is the WITHIN-cell
    offset it determines.

    This is not an artefact of the simulator. A die's lattice is stepper-printed
    from one reticle, so it is globally phase-coherent in reality too; placement
    error is nanometres against a pitch of tens of nanometres. See Mack,
    'Fundamental Principles of Optical Lithography', ch. on overlay budget.

    -> dict(origin_x, origin_y, ok_x, ok_y, pitch, coherence_x, coherence_y)
       origin_* are residues in [0, pitch).
    """
    p = float(pitch_search)
    pr = p * float(scale)
    out = dict(pitch=p, origin_x=None, origin_y=None, ok_x=False, ok_y=False,
               coherence_x=0.0, coherence_y=0.0)
    if not (2.0 < p < 200.0) or not np.isfinite(pr) or pr >= min(ref_u8.shape):
        return out

    for name, axis in (('x', 0), ('y', 1)):
        ph_s, co_s = fundamental_phase(search_u8, p, axis)
        ph_r, co_r = fundamental_phase(ref_u8, pr, axis)
        co = float(min(co_s, co_r))
        out[f'coherence_{name}'] = co
        if co < min_coherence:
            continue
        out[f'origin_{name}'] = float((ph_s - ph_r / float(scale)) % p)
        out[f'ok_{name}'] = True
    return out


def snap_to_phase(x, y, foot, lock):
    """Move a candidate centre onto the nearest phase-consistent position.

    Only the axes whose lattice family was actually detected are moved, and the
    shift is never more than half a pitch -- this refines a candidate, it never
    relocates it to a different lattice cell. Returns (x, y, dx, dy) so the
    caller can see how far each axis had to move; a large required shift means
    the candidate was sitting between cells, which is the signature of a
    correlation artefact rather than a real site.
    """
    p = lock['pitch']
    dx = dy = 0.0
    if lock['ok_x']:
        target = lock['origin_x'] + foot / 2.0
        dx = ((target - x + p / 2.0) % p) - p / 2.0
    if lock['ok_y']:
        target = lock['origin_y'] + foot / 2.0
        dy = ((target - y + p / 2.0) % p) - p / 2.0
    return x + dx, y + dy, dx, dy


def residual_saliency(res):
    """Is there actually a landmark in here, or only noise?

    Robust peak-to-noise ratio of the residual. This is the system's own
    answer to 'is this trial information-theoretically solvable?', computed
    WITHOUT ground truth -- a pure periodic-interior reference has residual
    energy that is spatially unstructured (noise), a landmark reference has a
    concentrated high-amplitude structure.
    """
    a = np.abs(res)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) + 1e-6
    peak = float(np.percentile(a, 99.9))
    return (peak - med) / (1.4826 * mad)
