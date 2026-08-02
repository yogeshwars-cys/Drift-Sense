"""
Drift-Sense synthetic dataset generator.

Two independent difficulty axes, deliberately kept orthogonal so the
evaluation can attribute error to the right cause:

  1. DRIFT-TIMING axis: how far the true site has wandered from the nominal
     recorded coordinate, driven by an exponential thermal-style model of
     elapsed_time_s. This is what the Digital Twin prior is supposed to
     predict.

  2. PERIODICITY axis: whether a non-periodic landmark (array boundary,
     dropped/doubled via, local gate crossing) sits near the true site.
     'hard' pairs have none nearby -> genuinely ambiguous matches, which is
     the array-interior failure mode Applied Materials explicitly asks for.

Run: python3 dataset_generator.py --n 40 --out dataset
"""
import argparse, json, math, os
import numpy as np
from PIL import Image

from common import (SEARCH_SIZE, REF_SIZE, INSET_SIZE,
                     render_dram, render_finfet, apply_bbox_mask,
                     edge_brighten, gaussian_blur, add_sensor_noise,
                     rotate_image, to_uint8)

MARGIN = 70  # safe for footprints up to ~120px (scale factor jitter 9x-11x)


def sample_scale_factor(rng):
    """True magnification ratio between reference and search image. Nominally
    10x but with realistic jitter -- the matcher is never told this value and
    must recover it via multi-scale search (this is the '10x scale
    difference' requirement, made a genuine unknown rather than a constant)."""
    scale_factor = float(rng.uniform(9.0, 11.0))
    footprint = REF_SIZE / scale_factor
    return scale_factor, footprint


def sample_true_drift(rng):
    """Ties (cx, cy) offset from the nominal recorded site to elapsed time
    via a saturating thermal-style model -- this is the signal the Digital
    Twin's prior is meant to learn."""
    elapsed_s = float(rng.uniform(60, 4 * 3600))
    d_max = float(rng.uniform(120, 220))     # true asymptotic drift, px
    tau = float(rng.uniform(1800, 5400))     # true thermal time constant, s
    mag = d_max * (1 - math.exp(-elapsed_s / tau))
    residual = rng.normal(0, 8)              # unmodeled hysteresis/friction
    theta = rng.uniform(0, 2 * math.pi)
    dx = (mag + residual) * math.cos(theta)
    dy = (mag + residual) * math.sin(theta)
    return elapsed_s, d_max, tau, dx, dy


def clamp_site(dx, dy):
    cx = min(max(500 + dx, MARGIN), SEARCH_SIZE - MARGIN)
    cy = min(max(500 + dy, MARGIN), SEARCH_SIZE - MARGIN)
    return cx, cy


def common_capture_params(rng, noisier_search=True):
    bg = float(rng.integers(15, 45))
    fg = float(rng.integers(200, 245))
    params = dict(
        bg=bg, fg=fg,
        edge_amt_search=float(rng.uniform(0.15, 0.32)),
        edge_amt_ref=float(rng.uniform(0.10, 0.25)),
        blur_search=float(rng.uniform(0.6, 1.6)),
        blur_ref=float(rng.uniform(0.3, 0.9)),
        rot_ref=float(rng.normal(0, 1.3)),
        gauss_noise_search=float(rng.uniform(9, 20)),
        gauss_noise_ref=float(rng.uniform(3, 9)),
        poisson_scale_search=float(rng.uniform(25, 55)),
        poisson_scale_ref=float(rng.uniform(60, 130)),
    )
    return params


def check_landmark_in_fov(cx, cy, footprint, landmark, defects=None, gates=None, array_bbox=None):
    if not landmark:
        return False
    half = footprint / 2.0
    x_lo, x_hi = cx - half, cx + half
    y_lo, y_hi = cy - half, cy + half

    if landmark == 'via_defect' and defects:
        for dx, dy, _ in defects:
            if x_lo + 4 <= dx <= x_hi - 4 and y_lo + 4 <= dy <= y_hi - 4:
                return True
        return False

    if landmark == 'gate_crossing' and gates:
        for gx, gy, glen, gw in gates:
            gx0, gx1 = gx - glen / 2.0, gx + glen / 2.0
            gy0, gy1 = gy - gw / 2.0, gy + gw / 2.0
            # Check for meaningful overlap with reference footprint
            ix0, ix1 = max(x_lo, gx0), min(x_hi, gx1)
            iy0, iy1 = max(y_lo, gy0), min(y_hi, gy1)
            if ix1 > ix0 + 5 and iy1 > iy0 + 5:
                return True
        return False

    if landmark == 'array_corner' and array_bbox:
        x0, y0, x1, y1 = array_bbox
        corners = []
        if x0 > 0:
            if y0 > 0: corners.append((x0, y0))
            if y1 < SEARCH_SIZE: corners.append((x0, y1))
        if x1 < SEARCH_SIZE:
            if y0 > 0: corners.append((x1, y0))
            if y1 < SEARCH_SIZE: corners.append((x1, y1))
        for cx_edge, cy_edge in corners:
            if x_lo <= cx_edge <= x_hi and y_lo <= cy_edge <= y_hi:
                return True
        # Also check if any array edge cuts through the FOV in 2D
        edge_in_x = (x_lo <= x0 <= x_hi) or (x_lo <= x1 <= x_hi)
        edge_in_y = (y_lo <= y0 <= y_hi) or (y_lo <= y1 <= y_hi)
        if edge_in_x and edge_in_y:
            return True
    return False


def generate_dram_pair(rng, pair_id, difficulty):
    pitch = int(rng.integers(9, 15))
    line_w = int(rng.integers(1, 3))
    via_r = int(rng.integers(3, 6))
    phase = (float(rng.uniform(0, pitch)), float(rng.uniform(0, pitch)))
    scale_factor, footprint = sample_scale_factor(rng)

    elapsed_s, d_max, tau, dx, dy = sample_true_drift(rng)
    cx, cy = clamp_site(dx, dy)

    array_bbox = [0, 0, SEARCH_SIZE, SEARCH_SIZE]
    defects = []
    landmark = None

    if difficulty == 'hard':
        if rng.random() < 0.4:
            defects.append((float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN)),
                             float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN)),
                             rng.choice(['drop', 'double'])))
            # keep it far from the true site so it's a distractor, not a cue
            while math.hypot(defects[-1][0] - cx, defects[-1][1] - cy) < 180:
                defects[-1] = (float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN)),
                               float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN)),
                               defects[-1][2])
    else:
        if rng.random() < 0.5:
            side_x = rng.choice(['left', 'right'])
            side_y = rng.choice(['top', 'bottom'])
            max_off = max(15.0, footprint / 2.0 - 8.0)
            off_x = float(rng.uniform(12, max_off))
            off_y = float(rng.uniform(12, max_off))
            x0 = cx - off_x if side_x == 'left' else 0.0
            x1 = SEARCH_SIZE if side_x == 'left' else cx + off_x
            y0 = cy - off_y if side_y == 'top' else 0.0
            y1 = SEARCH_SIZE if side_y == 'top' else cy + off_y
            array_bbox = [float(np.clip(v, 0, SEARCH_SIZE)) for v in (x0, y0, x1, y1)]
            landmark = 'array_corner'
        else:
            ang = rng.uniform(0, 2 * math.pi)
            max_dist = max(12.0, footprint / 2.0 - 10.0)
            ddist = rng.uniform(8, max_dist)
            defects.append((cx + ddist * math.cos(ang), cy + ddist * math.sin(ang),
                             rng.choice(['drop', 'double'])))
            landmark = 'via_defect'

    cp = common_capture_params(rng)

    search = render_dram(SEARCH_SIZE, pitch, line_w, via_r, phase, defects,
                          cp['bg'], cp['fg'], scale=1.0, offset=(0.0, 0.0))
    search = apply_bbox_mask(search, array_bbox, cp['bg'])
    search = edge_brighten(search, cp['edge_amt_search'])
    search = gaussian_blur(search, cp['blur_search'])
    search = add_sensor_noise(search, cp['gauss_noise_search'],
                               cp['poisson_scale_search'], rng)

    ref = render_dram(REF_SIZE, pitch, line_w, via_r, phase, defects,
                       cp['bg'], cp['fg'], scale=scale_factor,
                       offset=(cx - footprint / 2, cy - footprint / 2))
    ref = apply_bbox_mask(ref, array_bbox, cp['bg'], scale=scale_factor,
                           offset=(cx - footprint / 2, cy - footprint / 2))
    ref = rotate_image(ref, cp['rot_ref'], cp['bg'])
    ref = edge_brighten(ref, cp['edge_amt_ref'])
    ref = gaussian_blur(ref, cp['blur_ref'])
    ref = add_sensor_noise(ref, cp['gauss_noise_ref'], cp['poisson_scale_ref'], rng)

    landmark_in_fov = check_landmark_in_fov(cx, cy, footprint, landmark,
                                             defects=defects, array_bbox=array_bbox)

    meta = dict(pair_id=pair_id, style='dram', gt_x=cx, gt_y=cy,
                nominal_x=500.0, nominal_y=500.0, pitch=pitch,
                difficulty=difficulty, landmark=landmark,
                landmark_in_fov=landmark_in_fov,
                elapsed_time_s=elapsed_s, true_d_max=d_max, true_tau=tau,
                true_scale_factor=scale_factor, true_footprint_px=footprint,
                array_bbox=array_bbox, rotation_ref_deg=cp['rot_ref'])
    return to_uint8(search), to_uint8(ref), meta


def generate_finfet_pair(rng, pair_id, difficulty):
    pitch = int(rng.integers(7, 12))
    fin_w = int(rng.integers(1, 3))
    phase = float(rng.uniform(0, pitch))
    scale_factor, footprint = sample_scale_factor(rng)

    elapsed_s, d_max, tau, dx, dy = sample_true_drift(rng)
    cx, cy = clamp_site(dx, dy)

    array_bbox = [0, 0, SEARCH_SIZE, SEARCH_SIZE]
    gates = []
    landmark = None

    if difficulty == 'hard':
        if rng.random() < 0.4:
            gx = float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN))
            gy = float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN))
            while math.hypot(gx - cx, gy - cy) < 180:
                gx = float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN))
                gy = float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN))
            gates.append((gx, gy, float(rng.uniform(150, 300)), float(rng.uniform(15, 30))))
    else:
        if rng.random() < 0.5:
            side_x = rng.choice(['left', 'right'])
            side_y = rng.choice(['top', 'bottom'])
            max_off = max(15.0, footprint / 2.0 - 8.0)
            off_x = float(rng.uniform(12, max_off))
            off_y = float(rng.uniform(12, max_off))
            x0 = cx - off_x if side_x == 'left' else 0.0
            x1 = SEARCH_SIZE if side_x == 'left' else cx + off_x
            y0 = cy - off_y if side_y == 'top' else 0.0
            y1 = SEARCH_SIZE if side_y == 'top' else cy + off_y
            array_bbox = [float(np.clip(v, 0, SEARCH_SIZE)) for v in (x0, y0, x1, y1)]
            landmark = 'array_corner'
        else:
            gates.append((cx, cy + float(rng.uniform(-10, 10)),
                          float(rng.uniform(150, 300)), float(rng.uniform(15, 30))))
            landmark = 'gate_crossing'

    cp = common_capture_params(rng)

    search = render_finfet(SEARCH_SIZE, pitch, fin_w, gates, phase,
                            cp['bg'], cp['fg'], scale=1.0, offset=(0.0, 0.0))
    search = apply_bbox_mask(search, array_bbox, cp['bg'])
    search = edge_brighten(search, cp['edge_amt_search'])
    search = gaussian_blur(search, cp['blur_search'])
    search = add_sensor_noise(search, cp['gauss_noise_search'],
                               cp['poisson_scale_search'], rng)

    ref = render_finfet(REF_SIZE, pitch, fin_w, gates, phase, cp['bg'], cp['fg'],
                         scale=scale_factor, offset=(cx - footprint / 2, cy - footprint / 2))
    ref = apply_bbox_mask(ref, array_bbox, cp['bg'], scale=scale_factor,
                           offset=(cx - footprint / 2, cy - footprint / 2))
    ref = rotate_image(ref, cp['rot_ref'], cp['bg'])
    ref = edge_brighten(ref, cp['edge_amt_ref'])
    ref = gaussian_blur(ref, cp['blur_ref'])
    ref = add_sensor_noise(ref, cp['gauss_noise_ref'], cp['poisson_scale_ref'], rng)

    landmark_in_fov = check_landmark_in_fov(cx, cy, footprint, landmark,
                                             gates=gates, array_bbox=array_bbox)

    meta = dict(pair_id=pair_id, style='finfet', gt_x=cx, gt_y=cy,
                nominal_x=500.0, nominal_y=500.0, pitch=pitch,
                difficulty=difficulty, landmark=landmark,
                landmark_in_fov=landmark_in_fov,
                elapsed_time_s=elapsed_s, true_d_max=d_max, true_tau=tau,
                true_scale_factor=scale_factor, true_footprint_px=footprint,
                array_bbox=array_bbox, rotation_ref_deg=cp['rot_ref'])
    return to_uint8(search), to_uint8(ref), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40, help='total pairs, split evenly dram/finfet')
    ap.add_argument('--out', type=str, default='dataset')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    records = []
    style_counts = {'dram': 0, 'finfet': 0}

    for i in range(args.n):
        style = 'dram' if i % 2 == 0 else 'finfet'
        difficulty = 'easy' if style_counts[style] % 2 == 0 else 'hard'
        style_counts[style] += 1
        gen = generate_dram_pair if style == 'dram' else generate_finfet_pair
        search, ref, meta = gen(rng, i, difficulty)
        search_path = os.path.join(img_dir, f'{i:03d}_{style}_search.png')
        ref_path = os.path.join(img_dir, f'{i:03d}_{style}_ref.png')
        Image.fromarray(search).save(search_path)
        Image.fromarray(ref).save(ref_path)
        meta['search_path'] = search_path
        meta['ref_path'] = ref_path
        records.append(meta)
        print(f'[{i+1}/{args.n}] {style} difficulty={meta["difficulty"]} '
              f'landmark={meta["landmark"]} gt=({meta["gt_x"]:.1f},{meta["gt_y"]:.1f})')

    with open(os.path.join(args.out, 'ground_truth.json'), 'w') as f:
        json.dump(records, f, indent=2)
    print(f'\nWrote {len(records)} pairs to {args.out}/ (images/ + ground_truth.json)')


if __name__ == '__main__':
    main()
