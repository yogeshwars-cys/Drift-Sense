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
    """Bins whose spatial period falls inside pitch_range (px)."""
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    # period (px) = N / r  (square images)
    with np.errstate(divide='ignore'):
        period = h / np.maximum(r, 1e-6)
    return (period >= pitch_range[0]) & (period <= pitch_range[1]), (yy - cy), (xx - cx)


def dominant_orientation(img_u8, pitch_range):
    """Angle (deg) of the dominant lattice wave-vector. Rotation-robust:
    read straight off the 2-D power spectrum, so the reference image's ~1.3
    deg capture rotation is measured rather than ignored."""
    img_f = img_u8.astype(np.float32)
    mag = _spectrum(img_f)
    band, dy, dx = _band_mask(mag.shape, pitch_range)
    m = np.where(band, mag, 0.0)
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    return float(np.degrees(np.arctan2(dy[iy, ix], dx[iy, ix])))


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
        return []
    order = np.argsort(-mag[ys, xs])[:max_peaks]
    return list(zip(ys[order], xs[order]))


def aperiodic_residual(img_u8, pitch_range, notch_radius=3.0, blur_sigma=1.2):
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
    band, dy, dx = _band_mask(mag.shape, pitch_range)
    peaks = _find_spectral_peaks(mag, band)

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
