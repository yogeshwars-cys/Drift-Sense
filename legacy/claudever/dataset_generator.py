"""
dataset_generator.py
=====================
Synthetic dataset generator for the Drift-Sense navigation-error-recovery
challenge.

Generates (Reference, Search) image pairs that mimic a wafer inspection
tool re-visiting a die site:

  * Reference image  -> a single site imaged at "100x" magnification.
  * Search image     -> the same physical region imaged at "10x" magnification
                         (i.e. ~10x wider field of view, downsampled), with the
                         reference site appearing at a random location inside
                         it. This is the tool's re-landing image.

Both images are rendered procedurally from a shared, continuous coordinate
space so that the *exact* ground-truth location of the reference pattern
inside the search image is known analytically -- no manual annotation needed.

Two architecture styles are supported:
  * "dram"   -> periodic horizontal word-lines / vertical bit-lines with a
                via dot at every intersection.
  * "finfet" -> dense parallel vertical fins crossed by 1-2 horizontal gate
                bars.

Independent, physically-motivated degradations are applied separately to the
reference and search image (see CITATIONS.md for justification of each
choice):
  * Independent Poisson+Gaussian sensor noise (shot noise + read noise),
    with the search image noisier than the reference image.
  * SEM-style edge brightening (gradient-proportional intensity boost).
  * Gaussian blur (beam spot / defocus), stronger on the search image.
  * Small relative rotation and scale jitter between reference and search,
    modelling stage rotation error and magnification calibration drift.

Usage
-----
    python dataset_generator.py --style dram --num_pairs 30 --out_dir data/
    python dataset_generator.py --style finfet --num_pairs 30 --out_dir data/
    python dataset_generator.py --style both --num_pairs 30 --out_dir data/

Output
------
For every pair `i`, writes:
    <out_dir>/<style>_<i>_reference.png
    <out_dir>/<style>_<i>_search.png
and appends one row to <out_dir>/ground_truth.csv with columns:
    pair_id,style,search_w,search_h,true_x,true_y,scale_factor,
    rotation_deg,ref_w,ref_h
`true_x, true_y` is the ground-truth CENTER of the reference pattern's
footprint inside the search image, in search-image pixel coordinates
(sub-pixel precision, float).
"""

import argparse
import csv
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel


# --------------------------------------------------------------------------
# Procedural pattern functions
# --------------------------------------------------------------------------
# All patterns are defined as smooth (Gaussian-profile) periodic structures
# over a continuous coordinate space, in "native units" (the resolution the
# reference image is rendered at, i.e. 1 native unit = 1 reference pixel).
# Using smooth Gaussian line/blob profiles (rather than hard-edged rectangles)
# is a deliberate anti-aliasing choice: it keeps the pattern band-limited so
# it can be point-sampled at the coarser search-image resolution without
# introducing severe aliasing artifacts, and it is also a reasonable proxy
# for real lithographic edge rounding + SEM beam blur (see CITATIONS.md).

def _periodic_gaussian(coord, pitch, phase, sigma):
    """1D periodic Gaussian 'line' intensity profile, value in [0,1]."""
    d = ((coord - phase + pitch / 2.0) % pitch) - pitch / 2.0
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def dram_pattern(X, Y, params):
    """Grid of horizontal word-lines + vertical bit-lines with via dots."""
    horiz = _periodic_gaussian(Y, params["pitch_y"], params["phase_y"], params["line_sigma"])
    vert = _periodic_gaussian(X, params["pitch_x"], params["phase_x"], params["line_sigma"])
    lines = np.clip(horiz + vert, 0.0, 1.0)
    via = horiz * vert  # peaks only near true intersections
    img = 0.12 + 0.55 * lines + params["via_amp"] * via
    return np.clip(img, 0.0, 1.0)


def finfet_pattern(X, Y, params):
    """Dense vertical fins crossed by 1-2 horizontal gate bars."""
    fins = _periodic_gaussian(X, params["pitch_x"], params["phase_x"], params["fin_sigma"])
    gate1 = _periodic_gaussian(Y, params["gate_pitch"], params["gate_phase"], params["gate_sigma"])
    # A second, fainter gate bar offset from the first for extra local structure.
    gate2 = _periodic_gaussian(
        Y, params["gate_pitch"], params["gate_phase"] + params["gate_pitch"] * 0.5, params["gate_sigma"] * 0.7
    )
    base = 0.10 + 0.6 * fins
    img = base + 0.35 * gate1 + 0.20 * gate2
    return np.clip(img, 0.0, 1.0)


PATTERN_FUNCS = {"dram": dram_pattern, "finfet": finfet_pattern}


def random_params(style, rng):
    # NOTE ON SCALE: patterns are defined in "native units" = reference-image
    # pixels. The search image samples this same space at ~10 native units
    # per pixel. For periodic structure to survive that downsampling without
    # aliasing away to nothing, pitch and line width must be large enough in
    # native units that they still span several SEARCH pixels after the /10
    # downsample (rule of thumb: sigma_native should be a good multiple of
    # the downsample factor, and pitch_native >> sigma_native).
    if style == "dram":
        return {
            "pitch_x": rng.uniform(100, 160),
            "pitch_y": rng.uniform(100, 160),
            "phase_x": rng.uniform(0, 100),
            "phase_y": rng.uniform(0, 100),
            "line_sigma": rng.uniform(10, 18),
            "via_amp": rng.uniform(0.25, 0.4),
        }
    elif style == "finfet":
        return {
            "pitch_x": rng.uniform(55, 90),
            "fin_sigma": rng.uniform(6, 10),
            "phase_x": rng.uniform(0, 100),
            "gate_pitch": rng.uniform(220, 320),
            "gate_phase": rng.uniform(0, 100),
            "gate_sigma": rng.uniform(25, 40),
        }
    else:
        raise ValueError(f"Unknown style: {style}")


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def render_patch(pattern_fn, params, origin, size_px, sample_step, rotation_deg,
                  supersample=1):
    """
    Evaluate a pattern over a square patch.

    origin        : (x0, y0) top-left corner of the patch, in native units.
    size_px       : output resolution (size_px x size_px pixels).
    sample_step   : native units per output pixel (>1 => downsampled/zoomed
                    out, i.e. the "search" image; ==1 => the "reference"
                    image).
    rotation_deg  : rotates the sampling grid about the patch center, to
                    simulate a small relative pose error between the two
                    captures.
    supersample   : anti-aliasing factor for the point-sampling grid.
    """
    n = size_px * supersample
    # pixel-center coordinates in the *output* grid, then scaled to native units
    px = (np.arange(n) + 0.5) / supersample
    gx, gy = np.meshgrid(px, px)
    gx = gx * sample_step
    gy = gy * sample_step

    # rotate about patch center
    cx = cy = (size_px * sample_step) / 2.0
    theta = np.deg2rad(rotation_deg)
    rx = (gx - cx) * np.cos(theta) - (gy - cy) * np.sin(theta) + cx
    ry = (gx - cx) * np.sin(theta) + (gy - cy) * np.cos(theta) + cy

    X = origin[0] + rx
    Y = origin[1] + ry
    img = pattern_fn(X, Y, params)

    if supersample > 1:
        img = img.reshape(size_px, supersample, size_px, supersample).mean(axis=(1, 3))
    return img.astype(np.float32)


# --------------------------------------------------------------------------
# Degradation model (see CITATIONS.md for justification of each choice)
# --------------------------------------------------------------------------

def add_edge_brightening(img, gain, rng):
    """SEM images show brighter contrast along feature edges (increased
    secondary-electron yield near edges/topography). Approximate this with
    a gradient-magnitude-proportional intensity boost."""
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    grad_mag = np.hypot(gx, gy)
    grad_mag = grad_mag / (grad_mag.max() + 1e-8)
    return np.clip(img + gain * grad_mag, 0.0, 1.0)


def add_sensor_noise(img, photon_scale, read_noise_std, rng):
    """Independent, signal-dependent Poisson (shot) noise + additive
    Gaussian (read) noise -- the standard mixed-noise model for
    camera/SEM-type sensors. photon_scale controls SNR: smaller = noisier."""
    scaled = np.clip(img, 0, 1) * photon_scale
    shot = rng.poisson(scaled).astype(np.float32) / photon_scale
    read = rng.normal(0, read_noise_std, size=img.shape).astype(np.float32)
    return np.clip(shot + read, 0.0, 1.0)


def degrade(img, rng, blur_sigma, edge_gain, photon_scale, read_noise_std):
    img = gaussian_filter(img, sigma=blur_sigma)
    img = add_edge_brightening(img, edge_gain, rng)
    img = add_sensor_noise(img, photon_scale, read_noise_std, rng)
    return img


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Main pair generator
# --------------------------------------------------------------------------

def generate_pair(style, rng, ref_size=500, search_size=1000,
                   nominal_downsample=10.0, downsample_jitter=1.0,
                   max_rotation_deg=2.0, max_drift_px=45.0):
    """
    max_drift_px: the true reference-pattern center is placed within this
    many SEARCH-image pixels of the search image's center, in each axis.
    This models bounded navigation/stage drift between visits -- the tool
    re-lands NEAR its intended site, not at an arbitrary point in the frame.
    (This is also why "closest to the search-image center" is a sensible
    disambiguation rule for periodic false positives -- the true answer is
    expected to be near-center by construction of the problem, not because
    of an arbitrary tie-break convention.)

    Returns: ref_img (uint8 HxW), search_img (uint8 HxW),
             true_center_xy (float, float) in search-image pixel coords,
             meta (dict)
    """
    pattern_fn = PATTERN_FUNCS[style]
    params = random_params(style, rng)

    # Random anchor for the reference site in a large native coordinate space.
    anchor = (rng.uniform(2000, 8000), rng.uniform(2000, 8000))

    # --- Reference image: 1:1 sampling, no rotation, mild degradation ---
    ref_clean = render_patch(
        pattern_fn, params, origin=anchor, size_px=ref_size,
        sample_step=1.0, rotation_deg=0.0, supersample=1,
    )
    ref_img = degrade(
        ref_clean, rng,
        blur_sigma=rng.uniform(0.4, 0.8),
        edge_gain=rng.uniform(0.12, 0.22),
        photon_scale=rng.uniform(55, 90),     # higher => less noise
        read_noise_std=rng.uniform(0.01, 0.02),
    )

    # --- Search image: downsampled, with the reference site placed at a
    #     random offset inside it, plus a small relative rotation/scale
    #     jitter to simulate real re-landing error. ---
    downsample = nominal_downsample + rng.uniform(-downsample_jitter, downsample_jitter)
    rotation = rng.uniform(-max_rotation_deg, max_rotation_deg)

    # Reference footprint size inside the search image (native units), after
    # its own small rotation is folded in (bounding box grows slightly).
    ref_footprint_native = ref_size * 1.0

    # Search window covers `search_size * downsample` native units.
    search_extent_native = search_size * downsample
    win_center = search_extent_native / 2.0
    max_drift_native = max_drift_px * downsample

    # Target local position (pre-rotation) of the reference footprint's
    # CENTER, drawn within a bounded-drift disk of the search window center.
    # Drift magnitude itself varies per-visit (not every landing drifts the
    # maximum amount): sample a radius fraction, then an angle.
    drift_frac = rng.uniform(0.1, 1.0)
    drift_angle = rng.uniform(0, 2 * np.pi)
    drift_r = drift_frac * max_drift_native
    target_local_x = win_center + drift_r * np.cos(drift_angle)
    target_local_y = win_center + drift_r * np.sin(drift_angle)
    off_x = target_local_x - ref_footprint_native / 2.0
    off_y = target_local_y - ref_footprint_native / 2.0
    search_origin = (anchor[0] - off_x, anchor[1] - off_y)

    search_clean = render_patch(
        pattern_fn, params, origin=search_origin, size_px=search_size,
        sample_step=downsample, rotation_deg=rotation, supersample=3,
    )
    search_img = degrade(
        search_clean, rng,
        blur_sigma=rng.uniform(0.6, 1.1),
        edge_gain=rng.uniform(0.10, 0.20),
        photon_scale=rng.uniform(20, 45),     # noisier than reference
        read_noise_std=rng.uniform(0.02, 0.035),
    )

    # Ground-truth center of the reference footprint in SEARCH pixel coords.
    #
    # render_patch() maps an output-local coordinate (gx, gy) [physical
    # native units, pre-origin-shift] to a rendered physical location via a
    # rotation about the search patch's own center (cx, cy):
    #   rx = (gx-cx)*cos(t) - (gy-cy)*sin(t) + cx
    #   ry = (gx-cx)*sin(t) + (gy-cy)*cos(t) + cy
    #   physical = origin + (rx, ry)
    # We know the physical location of the reference footprint's center is
    # origin + (target_local_x, target_local_y) where target_local =
    # (off_x, off_y) + half the footprint (i.e. where the un-rotated patch
    # would put it). We need the INVERSE: which output-local (gx, gy) --
    # and therefore which output pixel, since output_pixel = gx/downsample
    # -- rotates onto that physical target. Solving the linear system above
    # for (gx, gy) given (rx, ry) = target_local:
    target_local_x = off_x + ref_footprint_native / 2.0
    target_local_y = off_y + ref_footprint_native / 2.0
    cx = cy = search_extent_native / 2.0
    a = target_local_x - cx
    b = target_local_y - cy
    theta = np.deg2rad(rotation)
    gx = cx + a * np.cos(theta) + b * np.sin(theta)
    gy = cy - a * np.sin(theta) + b * np.cos(theta)
    true_x = gx / downsample
    true_y = gy / downsample

    meta = {
        "scale_factor": downsample,
        "rotation_deg": rotation,
        "ref_w": ref_size, "ref_h": ref_size,
        "search_w": search_size, "search_h": search_size,
    }
    return to_uint8(ref_img), to_uint8(search_img), (true_x, true_y), meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic Drift-Sense image pairs.")
    ap.add_argument("--style", choices=["dram", "finfet", "both"], default="both")
    ap.add_argument("--num_pairs", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--ref_size", type=int, default=500)
    ap.add_argument("--search_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    styles = ["dram", "finfet"] if args.style == "both" else [args.style]
    rng = np.random.default_rng(args.seed)

    gt_path = os.path.join(args.out_dir, "ground_truth.csv")
    write_header = not os.path.exists(gt_path)
    with open(gt_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "pair_id", "style", "search_w", "search_h", "true_x", "true_y",
                "scale_factor", "rotation_deg", "ref_w", "ref_h",
            ])

        t0 = time.time()
        count = 0
        for style in styles:
            n = args.num_pairs if args.style != "both" else args.num_pairs // len(styles)
            for i in range(n):
                ref_img, search_img, (tx, ty), meta = generate_pair(
                    style, rng, ref_size=args.ref_size, search_size=args.search_size,
                )
                pair_id = f"{style}_{i:03d}"
                Image.fromarray(ref_img, mode="L").save(
                    os.path.join(args.out_dir, f"{pair_id}_reference.png"))
                Image.fromarray(search_img, mode="L").save(
                    os.path.join(args.out_dir, f"{pair_id}_search.png"))
                writer.writerow([
                    pair_id, style, meta["search_w"], meta["search_h"],
                    f"{tx:.3f}", f"{ty:.3f}", f"{meta['scale_factor']:.4f}",
                    f"{meta['rotation_deg']:.4f}", meta["ref_w"], meta["ref_h"],
                ])
                count += 1
        dt = time.time() - t0
    print(f"Generated {count} pairs in {dt:.2f}s ({dt / max(count,1):.3f}s/pair) -> {args.out_dir}")


if __name__ == "__main__":
    main()
