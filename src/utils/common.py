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
from PIL import Image, ImageFilter, ImageDraw

INSET_SIZE = 100     # footprint (px) the reference occupies inside the search image
SEARCH_SIZE = 1000    # search image side length (px)
REF_SIZE = 1000       # reference image side length (px) -- same physical footprint,
                       # rendered at 10x the linear pixel density of the search image
SCALE = REF_SIZE / INSET_SIZE  # = 10


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


def render_dram(canvas_size, pitch, line_width, via_radius, phase,
                 defects, bg, fg, scale=1.0, offset=(0.0, 0.0)):
    """DRAM-style word-line/bit-line grid with via contacts at intersections.
    defects: list of (x, y, mode) in SEARCH-space absolute coords, mode in
    {'drop', 'double'} -- rare non-periodic landmarks (missing/doubled via)."""
    img = np.full((canvas_size, canvas_size), bg, dtype=np.float32)
    px, py = phase
    xs = _line_positions(canvas_size, pitch * scale, px * scale - offset[0] * scale)
    ys = _line_positions(canvas_size, pitch * scale, py * scale - offset[1] * scale)
    lw = max(1, int(round(line_width * scale)))
    for y in ys:
        y = int(round(y))
        img[max(0, y):min(canvas_size, y + lw), :] = fg
    for x in xs:
        x = int(round(x))
        img[:, max(0, x):min(canvas_size, x + lw)] = fg

    pil = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(pil)
    r = max(1, via_radius * scale)
    defect_map = {}
    for dx, dy, mode in defects:
        defect_map[(round((dx - px) / pitch), round((dy - py) / pitch))] = mode
    for y in ys:
        for x in xs:
            gx = round((x / scale + offset[0] - px) / pitch)
            gy = round((y / scale + offset[1] - py) / pitch)
            mode = defect_map.get((gx, gy))
            if mode == 'drop':
                continue
            rr = r * 2 if mode == 'double' else r
            cx, cy = float(x), float(y)
            draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=int(fg))
    return np.array(pil, dtype=np.float32)


def render_finfet(canvas_size, pitch, fin_width, gates, phase, bg, fg,
                   scale=1.0, offset=(0.0, 0.0)):
    """FinFET-style dense parallel vertical fins crossed by local rectangular
    gate structures. gates: list of (gx, gy, len_x, width_y) in SEARCH-space
    absolute coords -- the non-periodic 2D landmarks."""
    img = np.full((canvas_size, canvas_size), bg, dtype=np.float32)
    xs = _line_positions(canvas_size, pitch * scale, phase * scale - offset[0] * scale)
    fw = max(1, int(round(fin_width * scale)))
    for x in xs:
        x = int(round(x))
        img[:, max(0, x):min(canvas_size, x + fw)] = fg

    gate_fg = min(255, fg + 15)
    for gx, gy, glen, gw in gates:
        cx = (gx - offset[0]) * scale
        cy = (gy - offset[1]) * scale
        hlen, hw = glen * scale / 2, gw * scale / 2
        x0, x1 = int(cx - hlen), int(cx + hlen)
        y0, y1 = int(cy - hw), int(cy + hw)
        x0c, x1c = max(0, x0), min(canvas_size, x1)
        y0c, y1c = max(0, y0), min(canvas_size, y1)
        if x1c > x0c and y1c > y0c:
            img[y0c:y1c, x0c:x1c] = gate_fg
    return img


def apply_bbox_mask(img, bbox, bg, scale=1.0, offset=(0.0, 0.0)):
    mask = in_bbox_mask(img.shape[0], bbox, scale, offset)
    out = img.copy()
    out[~mask] = bg
    return out


def edge_brighten(img, amount):
    """Physical SEM edge-brightening: secondary-electron escape probability
    rises near topographic edges, so intensity is boosted proportional to
    local gradient magnitude (Sobel-style)."""
    gy, gx = np.gradient(img)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = grad_mag / (grad_mag.max() + 1e-6)
    return np.clip(img + amount * grad_mag * 255.0, 0, 255)


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
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    pil = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False,
                      fillcolor=int(bg))
    return np.array(pil, dtype=np.float32)


def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)
