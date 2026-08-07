"""
Shared rendering primitives for the Drift-Sense synthetic dataset generator.

Design principle: reference and search images depict the SAME physical
location at two magnifications (search = 1x baseline, reference = 10x).
Both are generated from the same absolute periodic-lattice parameters
(pitch, phase) so that a reference window rendered at 10x resolution is
mathematically guaranteed to match the corresponding search-image crop,
up to independently applied noise/blur/rotation/edge-brightening.
"""
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw

INSET_SIZE = 100     # footprint (px) the reference occupies inside the search image
SEARCH_SIZE = 1000    # search image side length (px)
REF_SIZE = 1000       # reference image side length (px) -- same physical footprint,
                       # rendered at 10x the linear pixel density of the search image
SCALE = REF_SIZE / INSET_SIZE  # = 10

# Supersampling factor for rasterisation. Drawing lines directly at the output
# resolution snaps every feature edge to an integer pixel boundary, which is
# wrong twice over:
#
#   1. Physically. A 10x demagnified image is formed by AREA-AVERAGING the
#      scene through the instrument's PSF, not by re-rasterising it at a coarser
#      grid. Feature edges land on fractional pixels and partially fill them.
#   2. Statistically. Integer rounding makes the sub-pixel phase of each line
#      quantise differently across the frame, stamping a per-cell aliasing
#      FINGERPRINT into the search image. A matcher can learn that fingerprint
#      and tell lattice cells apart by it -- scoring beautifully here and
#      collapsing on any other generator. Supersample-then-area-average removes
#      the artefact rather than leaving a trap in the data.
#
# 4x is where the residual edge-position error falls below the noise floor for
# the pitches used here; higher costs render time for nothing.
SUPERSAMPLE = 4


def _line_positions(size, pitch, phase):
    """Absolute pixel positions of periodic lines within [0, size), anchored
    at an absolute phase offset so periodicity is well-defined canvas-wide."""
    start = phase % pitch
    return np.arange(start, size, pitch)


def in_bbox_mask(canvas_size, bbox, scale=1.0, offset=(0.0, 0.0)):
    """Boolean mask, True where the *search-space* coordinate implied by a
    (possibly higher-resolution, offset) canvas pixel falls inside bbox.
    scale/offset let the same bbox be applied consistently to both the
    search canvas (scale=1, offset=0) and a local high-res reference crop
    (scale=SCALE, offset=(cx-INSET_SIZE/2, cy-INSET_SIZE/2))."""
    x0, y0, x1, y1 = bbox
    ys, xs = np.mgrid[0:canvas_size, 0:canvas_size]
    abs_x = xs / scale + offset[0]
    abs_y = ys / scale + offset[1]
    return (abs_x >= x0) & (abs_x < x1) & (abs_y >= y0) & (abs_y < y1)


def _downsample(img, factor):
    """Area-average an ss-times oversampled raster back to output resolution.
    This is the operation a demagnifying optic actually performs."""
    if factor == 1:
        return img
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def render_dram(canvas_size, pitch, line_width, via_radius, phase,
                 defects, bg, fg, scale=1.0, offset=(0.0, 0.0),
                 supersample=SUPERSAMPLE):
    """DRAM-style word-line/bit-line grid with via contacts at intersections.
    defects: list of (x, y, mode) in SEARCH-space absolute coords, mode in
    {'drop', 'double'} -- rare non-periodic landmarks (missing/doubled via).

    Rasterised at `supersample`x and area-averaged down, so feature edges land
    on fractional pixels the way a real demagnified image's do. See SUPERSAMPLE.
    """
    ss = max(1, int(supersample))
    n = canvas_size * ss
    eff = scale * ss                       # px per unit of SEARCH space
    img = np.full((n, n), bg, dtype=np.float32)
    px, py = phase
    xs = _line_positions(n, pitch * eff, px * eff - offset[0] * eff)
    ys = _line_positions(n, pitch * eff, py * eff - offset[1] * eff)
    lw = max(1, int(round(line_width * eff)))
    for y in ys:
        y = int(round(y))
        img[max(0, y):min(n, y + lw), :] = fg
    for x in xs:
        x = int(round(x))
        img[:, max(0, x):min(n, x + lw)] = fg

    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    r = max(1, via_radius * eff)
    defect_map = {}
    for dx, dy, mode in defects:
        defect_map[(round((dx - px) / pitch), round((dy - py) / pitch))] = mode
    for y in ys:
        for x in xs:
            gx = round((x / eff + offset[0] - px) / pitch)
            gy = round((y / eff + offset[1] - py) / pitch)
            mode = defect_map.get((gx, gy))
            if mode == 'drop':
                continue
            rr = r * 2 if mode == 'double' else r
            cx, cy = float(x), float(y)
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=int(fg))
    return _downsample(np.array(pil, dtype=np.float32), ss)


def render_finfet(canvas_size, pitch, fin_width, gates, phase, bg, fg,
                   scale=1.0, offset=(0.0, 0.0), supersample=SUPERSAMPLE):
    """FinFET-style dense parallel vertical fins crossed by local rectangular
    gate structures. gates: list of (gx, gy, len_x, width_y) in SEARCH-space
    absolute coords -- the non-periodic 2D landmarks.

    Rasterised at `supersample`x and area-averaged down. See SUPERSAMPLE.
    """
    ss = max(1, int(supersample))
    n = canvas_size * ss
    eff = scale * ss
    img = np.full((n, n), bg, dtype=np.float32)
    xs = _line_positions(n, pitch * eff, phase * eff - offset[0] * eff)
    fw = max(1, int(round(fin_width * eff)))
    for x in xs:
        x = int(round(x))
        img[:, max(0, x):min(n, x + fw)] = fg

    gate_fg = min(255, fg + 15)
    for gx, gy, glen, gw in gates:
        cx = (gx - offset[0]) * eff
        cy = (gy - offset[1]) * eff
        hlen, hw = glen * eff / 2, gw * eff / 2
        x0, x1 = int(cx - hlen), int(cx + hlen)
        y0, y1 = int(cy - hw), int(cy + hw)
        x0c, x1c = max(0, x0), min(n, x1)
        y0c, y1c = max(0, y0), min(n, y1)
        if x1c > x0c and y1c > y0c:
            img[y0c:y1c, x0c:x1c] = gate_fg
    return _downsample(img, ss)


def apply_bbox_mask(img, bbox, bg, scale=1.0, offset=(0.0, 0.0)):
    mask = in_bbox_mask(img.shape[0], bbox, scale, offset)
    out = img.copy()
    out[~mask] = bg
    return out


def edge_brighten(img, amount, detector_deg=None, asymmetry=0.6, tail_sigma=1.5):
    """Physical SEM edge-brightening.

    Secondary-electron escape probability rises near topographic edges, so
    intensity is boosted with the local gradient. Two refinements over a plain
    isotropic gradient-magnitude boost, both of which matter to the matcher:

      * DIRECTIONAL. The SE signal depends on the edge's orientation relative to
        the detector: a facet tilted toward the detector is brighter than the
        opposite facet of the same feature. An isotropic operator brightens both
        sides of every line equally, which is (a) not what an SEM does and
        (b) partially cancels under NCC, understating the difficulty. The
        detector azimuth is sampled per image, so reference and search see the
        same structure lit from the same side but with independent noise.

      * DECAYING TAIL. The escape volume extends beyond the geometric edge, so
        the bright band is not one pixel wide -- it falls off over a short
        distance. Modelled by blurring the edge term.

    References: Goldstein et al., 'Scanning Electron Microscopy and X-Ray
    Microanalysis' (SE yield vs. surface tilt, the eta(theta) ~ sec(theta)
    relation); Reimer, 'Scanning Electron Microscopy' (edge/topographic
    contrast and escape depth).
    """
    gy, gx = np.gradient(img.astype(np.float32))
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    peak = float(grad_mag.max()) + 1e-6

    if detector_deg is None:
        edge = grad_mag / peak
    else:
        th = np.radians(float(detector_deg))
        # Signed projection of the surface normal onto the detector direction.
        proj = (gx * np.cos(th) + gy * np.sin(th)) / peak
        # asymmetry=0 -> isotropic; 1 -> only detector-facing edges brighten.
        edge = (1.0 - asymmetry) * (grad_mag / peak) + asymmetry * np.clip(proj, 0.0, None)

    if tail_sigma > 0:
        edge = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), tail_sigma)
    return np.clip(img + amount * edge * 255.0, 0, 255)


def add_scan_distortion(img, jitter_px, drift_px, rng):
    """Raster-scan geometry error.

    An SEM builds the image one scan line at a time. Two things go wrong:

      * LINE JITTER: each line's horizontal start position is perturbed
        independently by beam-deflection noise and mains pickup, shearing rows
        by a fraction of a pixel relative to each other.
      * SLOW DRIFT: stage/beam drift over the seconds-long frame acquisition
        adds a smooth, monotonic shift that accumulates down the frame.

    Both are per-capture, so reference and search receive independent draws --
    which is exactly why a matcher must not assume the two images are related by
    a pure rigid transform.

    References: Reimer, 'Scanning Electron Microscopy' (scan-system linearity
    and drift); Goldstein et al. (image-distortion sources in raster scanning).
    """
    h, w = img.shape
    if jitter_px <= 0 and drift_px <= 0:
        return img
    jitter = rng.normal(0, jitter_px, h) if jitter_px > 0 else np.zeros(h)
    # smooth the jitter slightly: consecutive lines are correlated in practice
    if jitter_px > 0:
        jitter = np.convolve(jitter, np.ones(3) / 3.0, mode='same')
    drift = np.linspace(0.0, drift_px, h) if drift_px > 0 else np.zeros(h)
    shift = (jitter + drift).astype(np.float32)

    xs = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    return cv2.remap(img.astype(np.float32), xs, ys, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def add_charging(img, amount, streak_len, rng):
    """Electron-beam charging: local potential build-up on poorly-conducting
    regions raises the local SE yield and smears it along the fast-scan
    direction, producing the bright horizontal streaks characteristic of
    insulating features in an SEM.

    Reference: Cazaux, 'Charging in scanning electron microscopy from inside
    and outside' (Scanning, 2004); Goldstein et al., ch. on charging artefacts.
    """
    if amount <= 0 or streak_len < 2:
        return img
    h, w = img.shape
    bright = np.clip(img - np.percentile(img, 70), 0, None)
    k = np.zeros((1, int(streak_len)), dtype=np.float32)
    k[0, :] = np.exp(-np.arange(int(streak_len)) / (streak_len / 3.0))
    k /= k.sum()
    # anchor=(0,0) makes the smear TRAIL the feature along the fast-scan
    # direction. Left at the default centre anchor it would smear symmetrically
    # about the feature, which is not what charging does -- the potential builds
    # as the beam passes and decays after it.
    streak = cv2.filter2D(bright.astype(np.float32), -1, k, anchor=(0, 0))
    # patchy rather than uniform: charging accumulates unevenly
    blotch = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32),
                              (0, 0), max(h, w) / 20.0)
    blotch = 0.5 + 0.5 * np.clip(blotch / (blotch.std() + 1e-6), -2, 2) / 2.0
    return np.clip(img + amount * streak * blotch, 0, 255)


def add_shading(img, amount, rng):
    """Non-uniform illumination / detector-collection efficiency across the
    field of view: a smooth low-order multiplicative gradient. Harmless to a
    normalised correlation over the whole patch, and specifically NOT harmless
    to any statistic computed on raw intensity -- which is why it belongs in the
    generator rather than being assumed away.

    Reference: Goldstein et al. (detector collection-efficiency variation
    across the scanned field)."""
    if amount <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w] / float(max(h, w))
    ax, ay = rng.normal(0, 1, 2)
    field = 1.0 + amount * (ax * (xx - 0.5) + ay * (yy - 0.5))
    return np.clip(img * field, 0, 255)


def astigmatic_blur(img, sigma_x, sigma_y, angle_deg):
    """Anisotropic (astigmatic) defocus: a real column is never perfectly
    stigmated, so the PSF is elliptical and its axes are not aligned to the
    raster. Applied by rotating into the PSF frame, blurring separably, and
    rotating back.

    Reference: Reimer, 'Scanning Electron Microscopy' (astigmatism and its
    effect on the probe shape)."""
    if sigma_x <= 0 and sigma_y <= 0:
        return img
    a = img.astype(np.float32)
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    Mi = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    rot = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    rot = cv2.GaussianBlur(rot, (0, 0), sigmaX=max(sigma_x, 1e-3),
                           sigmaY=max(sigma_y, 1e-3))
    return cv2.warpAffine(rot, Mi, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(pil, dtype=np.float32)


def add_sensor_noise(img, gaussian_sigma, poisson_scale, rng):
    """Poisson-Gaussian mixed noise model: Poisson (signal-dependent shot
    noise) approximated via scaled Poisson draw, plus additive Gaussian
    (detector read noise). Independent rng per image call."""
    img_c = np.clip(img, 0, 255)
    if poisson_scale > 0:
        lam = np.clip(img_c / 255.0 * poisson_scale, 1e-3, None)
        shot = rng.poisson(lam).astype(np.float32) / poisson_scale * 255.0
        img_c = 0.5 * img_c + 0.5 * shot
    if gaussian_sigma > 0:
        img_c = img_c + rng.normal(0, gaussian_sigma, img_c.shape)
    return np.clip(img_c, 0, 255)


def rotate_image(img, angle_deg, bg):
    """Rotate about the centre WITHOUT introducing a constant-valued border.

    Filling the corners with `bg` stamps a flat ring into the reference that has
    no counterpart anywhere in the search image. Because NCC is computed over
    the whole template, that ring is a constant bias present at every candidate
    -- it does not merely add noise, it systematically compresses the score
    differences the matcher is trying to resolve, and it does so more at larger
    rotations. Reflecting instead keeps the border statistically continuous with
    the pattern, so the template carries no region that cannot match.
    """
    if abs(angle_deg) < 1e-6:
        return np.asarray(img, dtype=np.float32)
    a = np.clip(img, 0, 255).astype(np.float32)
    h, w = a.shape
    # Pad by enough that no output pixel can sample outside the source, then
    # rotate and crop back -- the corners are filled with real reflected
    # pattern rather than with a flat constant.
    pad = int(np.ceil(0.5 * np.hypot(h, w) - min(h, w) / 2.0)) + 2
    padded = cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape
    M = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(padded, M, (pw, ph), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return rot[pad:pad + h, pad:pad + w].astype(np.float32)


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)
