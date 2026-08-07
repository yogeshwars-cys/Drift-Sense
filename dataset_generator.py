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
                     edge_brighten, add_sensor_noise, rotate_image, to_uint8,
                     add_scan_distortion, add_charging, add_shading,
                     astigmatic_blur)

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


def sample_site(rng, dx, dy, placement):
    """True site placement.

    'annulus' ties the site to the thermal drift model: it lands 120-220px from
    the nominal recorded coordinate, which is what a stage that re-lands NEAR
    its target actually does.

    'uniform' places it anywhere in the frame. The problem statement says only
    that the reference appears "somewhere inside" the search image, so a matcher
    that assumes proximity to the centre is assuming a fact not in evidence.
    Measured: the pre-rewrite pipeline scored 44% under 'annulus' and 21.9%
    under 'uniform' -- more than half its accuracy was the placement prior
    rather than the matcher. 'uniform' is the default here for that reason.
    """
    if placement == 'annulus':
        return clamp_site(dx, dy)
    if placement == 'uniform':
        return (float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN)),
                float(rng.uniform(MARGIN, SEARCH_SIZE - MARGIN)))
    raise ValueError(f'unknown placement: {placement!r}')


def common_capture_params(rng, noisier_search=True):
    """Per-pair imaging conditions.

    Structure of this dict: anything named *_search or *_ref is drawn
    INDEPENDENTLY for the two captures, because they are two separate physical
    acquisitions. Anything without a suffix is a property of the instrument or
    the specimen and is therefore shared -- the detector azimuth, for instance,
    is where the detector physically is, so both captures see the same faces lit.

    Search-side degradation is uniformly heavier than reference-side: the test
    set is explicitly promised to be noisier on the search image, and a
    generator that does not reproduce that asymmetry trains for the wrong problem.
    """
    bg = float(rng.integers(15, 45))
    fg = float(rng.integers(200, 245))
    params = dict(
        bg=bg, fg=fg,
        # --- shared instrument geometry ---
        detector_deg=float(rng.uniform(0, 360)),
        edge_asymmetry=float(rng.uniform(0.35, 0.8)),
        # --- independent per capture ---
        edge_amt_search=float(rng.uniform(0.15, 0.32)),
        edge_amt_ref=float(rng.uniform(0.10, 0.25)),
        blur_search=float(rng.uniform(0.6, 1.6)),
        blur_ref=float(rng.uniform(0.3, 0.9)),
        rot_ref=float(rng.normal(0, 1.3)),
        gauss_noise_search=float(rng.uniform(9, 20)),
        gauss_noise_ref=float(rng.uniform(3, 9)),
        poisson_scale_search=float(rng.uniform(25, 55)),
        poisson_scale_ref=float(rng.uniform(60, 130)),
        # --- SEM scan-system and charging artefacts ---
        jitter_search=float(rng.uniform(0.0, 0.6)),
        jitter_ref=float(rng.uniform(0.0, 0.3)),
        scan_drift_search=float(rng.uniform(-2.5, 2.5)),
        scan_drift_ref=float(rng.uniform(-1.0, 1.0)),
        charge_search=float(rng.uniform(0.0, 0.30)),
        charge_ref=float(rng.uniform(0.0, 0.15)),
        streak_len=float(rng.integers(4, 14)),
        shading_search=float(rng.uniform(0.0, 0.18)),
        shading_ref=float(rng.uniform(0.0, 0.10)),
        astig_ratio=float(rng.uniform(1.0, 1.8)),
        astig_angle=float(rng.uniform(0, 180)),
    )
    return params


def _capture(img, cp, side, rng):
    """Apply one physical acquisition to a rendered scene.

    Order matters and follows the imaging chain: topographic contrast is formed
    first (edge brightening), then the optics blur it (astigmatic PSF), then the
    scan system distorts geometry, then charging adds its streaks, then the
    detector's collection efficiency shades the field, and only at the very end
    does the sensor add shot and read noise. Adding noise before blurring, for
    instance, would let the optics smooth the noise -- which is not what happens
    in a real column.
    """
    out = edge_brighten(img, cp[f'edge_amt_{side}'],
                        detector_deg=cp['detector_deg'],
                        asymmetry=cp['edge_asymmetry'])
    sigma = cp[f'blur_{side}']
    out = astigmatic_blur(out, sigma, sigma * cp['astig_ratio'], cp['astig_angle'])
    out = add_scan_distortion(out, cp[f'jitter_{side}'],
                              cp[f'scan_drift_{side}'], rng)
    out = add_charging(out, cp[f'charge_{side}'], cp['streak_len'], rng)
    out = add_shading(out, cp[f'shading_{side}'], rng)
    out = add_sensor_noise(out, cp[f'gauss_noise_{side}'],
                           cp[f'poisson_scale_{side}'], rng)
    return out


def nearest_landmark_distance(cx, cy, defects=None, gates=None, array_bbox=None):
    """Distance (px) from the true site to the nearest non-periodic feature.

    This is the continuous version of the easy/hard switch, and it is a far more
    honest difficulty axis. `landmark_in_fov` is a threshold on this quantity at
    roughly footprint/2, and thresholding it throws away the whole gradient: a
    landmark just outside the field of view is not equivalent to one 400px away,
    but the boolean says they are. Reporting accuracy against this distance
    shows where the algorithm actually degrades instead of asserting a cliff.
    """
    d = []
    for dx, dy, _mode in (defects or []):
        d.append(math.hypot(dx - cx, dy - cy))
    for gx, gy, glen, gw in (gates or []):
        # distance to the gate rectangle, not to its centre
        ddx = max(abs(gx - cx) - glen / 2.0, 0.0)
        ddy = max(abs(gy - cy) - gw / 2.0, 0.0)
        d.append(math.hypot(ddx, ddy))
    if array_bbox:
        x0, y0, x1, y1 = array_bbox
        # distance to any array EDGE that is actually interior to the canvas
        for edge, is_real in ((abs(cx - x0), x0 > 0), (abs(cx - x1), x1 < SEARCH_SIZE),
                              (abs(cy - y0), y0 > 0), (abs(cy - y1), y1 < SEARCH_SIZE)):
            if is_real:
                d.append(edge)
    return float(min(d)) if d else float('inf')


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


def generate_dram_pair(rng, pair_id, difficulty, placement='uniform'):
    pitch = int(rng.integers(9, 15))
    line_w = int(rng.integers(1, 3))
    via_r = int(rng.integers(3, 6))
    phase = (float(rng.uniform(0, pitch)), float(rng.uniform(0, pitch)))
    scale_factor, footprint = sample_scale_factor(rng)

    elapsed_s, d_max, tau, dx, dy = sample_true_drift(rng)
    cx, cy = sample_site(rng, dx, dy, placement)

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
    search = _capture(search, cp, 'search', rng)

    ref = render_dram(REF_SIZE, pitch, line_w, via_r, phase, defects,
                       cp['bg'], cp['fg'], scale=scale_factor,
                       offset=(cx - footprint / 2, cy - footprint / 2))
    ref = apply_bbox_mask(ref, array_bbox, cp['bg'], scale=scale_factor,
                           offset=(cx - footprint / 2, cy - footprint / 2))
    ref = rotate_image(ref, cp['rot_ref'], cp['bg'])
    ref = _capture(ref, cp, 'ref', rng)

    landmark_in_fov = check_landmark_in_fov(cx, cy, footprint, landmark,
                                             defects=defects, array_bbox=array_bbox)
    lm_dist = nearest_landmark_distance(cx, cy, defects=defects,
                                        array_bbox=array_bbox)

    meta = dict(pair_id=pair_id, style='dram', gt_x=cx, gt_y=cy,
                nominal_x=500.0, nominal_y=500.0, pitch=pitch,
                difficulty=difficulty, landmark=landmark,
                landmark_in_fov=landmark_in_fov,
                landmark_distance_px=lm_dist,
                elapsed_time_s=elapsed_s, true_d_max=d_max, true_tau=tau,
                true_scale_factor=scale_factor, true_footprint_px=footprint,
                array_bbox=array_bbox, rotation_ref_deg=cp['rot_ref'])
    return to_uint8(search), to_uint8(ref), meta


def generate_finfet_pair(rng, pair_id, difficulty, placement='uniform'):
    pitch = int(rng.integers(7, 12))
    fin_w = int(rng.integers(1, 3))
    phase = float(rng.uniform(0, pitch))
    scale_factor, footprint = sample_scale_factor(rng)

    elapsed_s, d_max, tau, dx, dy = sample_true_drift(rng)
    cx, cy = sample_site(rng, dx, dy, placement)

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
    search = _capture(search, cp, 'search', rng)

    ref = render_finfet(REF_SIZE, pitch, fin_w, gates, phase, cp['bg'], cp['fg'],
                         scale=scale_factor, offset=(cx - footprint / 2, cy - footprint / 2))
    ref = apply_bbox_mask(ref, array_bbox, cp['bg'], scale=scale_factor,
                           offset=(cx - footprint / 2, cy - footprint / 2))
    ref = rotate_image(ref, cp['rot_ref'], cp['bg'])
    ref = _capture(ref, cp, 'ref', rng)

    landmark_in_fov = check_landmark_in_fov(cx, cy, footprint, landmark,
                                             gates=gates, array_bbox=array_bbox)
    lm_dist = nearest_landmark_distance(cx, cy, gates=gates,
                                        array_bbox=array_bbox)

    meta = dict(pair_id=pair_id, style='finfet', gt_x=cx, gt_y=cy,
                nominal_x=500.0, nominal_y=500.0, pitch=pitch,
                difficulty=difficulty, landmark=landmark,
                landmark_in_fov=landmark_in_fov,
                landmark_distance_px=lm_dist,
                elapsed_time_s=elapsed_s, true_d_max=d_max, true_tau=tau,
                true_scale_factor=scale_factor, true_footprint_px=footprint,
                array_bbox=array_bbox, rotation_ref_deg=cp['rot_ref'])
    return to_uint8(search), to_uint8(ref), meta


def main():
    ap = argparse.ArgumentParser(
        description='Generate synthetic Reference/Search image pairs with '
                    'recorded ground-truth centres.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--style', choices=('dram', 'finfet', 'both'), default='both',
                    help='die architecture to generate')
    ap.add_argument('--n', type=int, default=40, help='total number of pairs')
    ap.add_argument('--out', type=str, default='dataset', help='output directory')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--difficulty-mix', type=float, default=0.2, metavar='FRAC',
                    help='fraction of pairs with NO landmark near the true site, '
                         'i.e. deep array interior. These are unidentifiable by '
                         'construction. The problem statement asks for "at least '
                         'one" such region, so the default is a minority; set 0.5 '
                         'to reproduce the stress split.')
    ap.add_argument('--placement', choices=('uniform', 'annulus'), default='uniform',
                    help='uniform: the reference may appear anywhere inside the '
                         'search image, per the problem statement. annulus: tied '
                         'to the thermal drift model, 120-220px from the nominal '
                         'coordinate (the legacy behaviour).')
    args = ap.parse_args()

    if not 0.0 <= args.difficulty_mix <= 1.0:
        ap.error('--difficulty-mix must be in [0, 1]')

    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)

    styles = ['dram', 'finfet'] if args.style == 'both' else [args.style]
    # Deterministic counts rather than per-pair coin flips, so a 20-pair set
    # actually contains the requested mix instead of a binomial sample of it.
    n_hard = int(round(args.n * args.difficulty_mix))
    difficulties = ['hard'] * n_hard + ['easy'] * (args.n - n_hard)
    rng.shuffle(difficulties)

    records = []
    for i in range(args.n):
        style = styles[i % len(styles)]
        difficulty = difficulties[i]
        gen = generate_dram_pair if style == 'dram' else generate_finfet_pair
        search, ref, meta = gen(rng, i, difficulty, placement=args.placement)
        search_path = os.path.join(img_dir, f'{i:03d}_{style}_search.png')
        ref_path = os.path.join(img_dir, f'{i:03d}_{style}_ref.png')
        Image.fromarray(search).save(search_path)
        Image.fromarray(ref).save(ref_path)
        meta['search_path'] = search_path
        meta['ref_path'] = ref_path
        meta['placement'] = args.placement
        records.append(meta)
        print(f'[{i+1}/{args.n}] {style} difficulty={meta["difficulty"]} '
              f'landmark={meta["landmark"]} gt=({meta["gt_x"]:.1f},{meta["gt_y"]:.1f})')

    with open(os.path.join(args.out, 'ground_truth.json'), 'w') as f:
        json.dump(records, f, indent=2)
    n_solvable = sum(1 for r in records if r['landmark_in_fov'])
    print(f'\nWrote {len(records)} pairs to {args.out}/ (images/ + ground_truth.json)')
    print(f'  style={args.style}  placement={args.placement}  '
          f'difficulty_mix={args.difficulty_mix}')
    print(f'  landmark in FOV (information-theoretically solvable): '
          f'{n_solvable}/{len(records)}')


if __name__ == '__main__':
    main()
