"""Localized full-resolution rescoring: score the LANDMARK, not the frame.

`rescore_fullres` compares each candidate's whole upsampled crop against the
whole reference. In a periodic array that means ~98.7% of the score is
lattice that is identical at every candidate, and the ~1.3% that actually
carries identity -- one dropped via -- is averaged into nothing. Measured
consequence: via-defect sites localize at 19.0% against 92.5% for array
corners.

This finds where the landmark lives in the REFERENCE residual, maps that
offset into each candidate's own frame using the measured magnification, and
compares only there. The lattice-identical neighbours are excluded from the
window by construction, so they cannot win on background agreement.

The mapping is the whole trick. A candidate at (cx, cy) with footprint `foot`
claims the reference centre sits at (cx, cy) and the reference is `n/foot`
times magnified. A reference point (lx, ly) therefore lands at

    cx + (lx - n/2) * foot/n ,  cy + (ly - n/2) * foot/n

so every candidate is asked the same question -- "is the landmark where YOUR
hypothesis says it should be?" -- and a candidate shifted by one lattice
vector looks at a spot where there is no defect.

    python probes/local_rescore.py --dataset dataset_stress  --sweep
    python probes/local_rescore.py --dataset dataset_primary --box 2.0
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2

import localize as L
from lattice import (estimate_scale, aperiodic_residual, scale_uncertainty,
                     foot_bracket, phase_lock, snap_to_phase,
                     relative_rotation, rotate)

TOL = 15.0


# The landmark finder and its border mask now live in localize.py, where they
# ship. Importing rather than copying means this probe measures the code that
# actually runs -- a copy would silently drift from it, and the border mask is
# exactly the kind of "magic constant" a copy loses.
BORDER_FRAC = L.DOG_BORDER_FRAC
landmark_site = L._landmark_site


def local_scores(search_u8, ref_u8, res_search, res_ref, cands, foot_exact,
                 pitch_ref, box_pitches):
    """NCC restricted to the landmark neighbourhood, appearance and residual."""
    n = ref_u8.shape[0]
    lx, ly = landmark_site(res_ref, pitch_ref)
    peak = mean = 0.0
    side = int(round(box_pitches * pitch_ref))
    side = int(np.clip(side, 24, n // 3))
    h = side // 2
    x0, y0 = int(round(lx)) - h, int(round(ly)) - h
    x0 = int(np.clip(x0, 0, n - side))
    y0 = int(np.clip(y0, 0, n - side))
    ref_patch = ref_u8[y0:y0 + side, x0:x0 + side]
    res_patch = res_ref[y0:y0 + side, x0:x0 + side]

    # offset of the window centre from the reference centre, in reference px
    ox = (x0 + h) - n / 2.0
    oy = (y0 + h) - n / 2.0

    out = []
    for c in cands:
        foot = float(foot_exact if foot_exact else c['foot'])
        k = foot / n                       # reference px -> search px
        sx = c['x'] + ox * k
        sy = c['y'] + oy * k
        sub = side * k                     # window side in search px
        a = L._upsampled_crop(search_u8, sx, sy, sub, side)
        r = L._upsampled_crop(res_search, sx, sy, sub, side)
        out.append((0.0 if a is None else L._ncc(a, ref_patch),
                    0.0 if r is None else L._ncc(r, res_patch)))
    arr = np.array(out) if out else np.zeros((0, 2))
    return arr, dict(peak=peak, mean=mean, contrast=peak / max(mean, 1e-9), side=side)


def run(dataset, box_pitches, limit=0):
    recs = json.load(open(f'{dataset}/ground_truth.json'))
    if limit:
        recs = recs[:limit]
    rows = []
    for m in recs:
        if not m.get('landmark_in_fov'):
            continue
        s = cv2.imread(m['search_path'], cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(m['ref_path'], cv2.IMREAD_GRAYSCALE)
        gx, gy = m['gt_x'], m['gt_y']

        scale_est, lat_s, lat_r, ok = estimate_scale(s, r)
        pitch = float(lat_s['pitch'])
        span = scale_uncertainty(lat_s, lat_r if ok else None, scale_est)
        if not ok:
            span = max(span, 1.5)
        feet = foot_bracket(r.shape[0], scale_est, span)
        foot_ref = max(20, int(round(r.shape[0] / scale_est)))
        nms_r = int(max(3, round(L.NMS_PITCH_FRACTION * pitch)))

        cand_deg, _ = relative_rotation(s, r, pitch, scale_est)
        if abs(cand_deg) > 0.15:
            rr = rotate(r, cand_deg)
            if L._best_ncc(s, rr, foot_ref) > L._best_ncc(s, r, foot_ref) + L.ROTATION_MIN_GAIN:
                r = rr

        win = L._window(s.shape, (s.shape[1] / 2.0, s.shape[0] / 2.0),
                        max(s.shape) + foot_ref)
        cands = L._multiscale_peaks(s, r, win, feet, nms_radius=nms_r,
                                    top_k_per_scale=L.TOP_K_PER_SCALE)
        s_band, r_band = L.pitch_bands(pitch)
        res_s = L._envelope_normalise(aperiodic_residual(s, s_band))
        res_r = L._envelope_normalise(aperiodic_residual(r, r_band))
        lmap = L._landmark_map_from(res_s, res_r, foot_ref, win)
        if lmap is not None:
            cands += L._peaks_from_map(lmap, win, foot_ref, nms_radius=nms_r)
        win_area = max(1.0, (win[2] - win[0]) * (win[3] - win[1]))
        keep = int(np.clip(round(15 * win_area / (360.0 ** 2)), 15, L.MAX_CANDIDATES))
        cands = L._dedupe(cands, radius=nms_r, keep=keep)

        lock = phase_lock(s, r, pitch, scale_est)
        if lock['ok_x'] or lock['ok_y']:
            for c in cands:
                sx2, sy2, _, _ = snap_to_phase(c['x'], c['y'], c['foot'], lock)
                c['x'], c['y'] = sx2, sy2
        cands = L.rescore_fullres(s, r, res_s, res_r, cands,
                                  foot_exact=r.shape[0] / float(scale_est))
        d = np.array([np.hypot(c['x'] - gx, c['y'] - gy) for c in cands])
        if not (d <= TOL).any():
            continue

        pitch_ref = pitch * scale_est
        loc, info = local_scores(s, r, res_s, res_r, cands,
                                 r.shape[0] / float(scale_est), pitch_ref,
                                 box_pitches)
        base = (L.W_APPEARANCE * np.array([c['score'] for c in cands])
                + L.W_FINE_APPEARANCE * np.array([c['fine_appearance'] for c in cands])
                + L.W_LANDMARK * np.clip(
                    L._spread_z(np.array([c['fine_landmark'] for c in cands])), 0, 12) / 12.0)
        rows.append(dict(pair_id=m['pair_id'], landmark=m.get('landmark'),
                         y=(d <= TOL).astype(int).tolist(),
                         base=base.tolist(),
                         loc_app=loc[:, 0].tolist(), loc_res=loc[:, 1].tolist(),
                         contrast=info['contrast']))
        print(f'  pair {m["pair_id"]:3d} {m.get("landmark","-"):14s} '
              f'n={len(cands):3d} box={info["side"]}px', flush=True)
    return rows


def r1(rows, fn):
    return np.mean([r['y'][int(np.argmax(fn(r)))] == 1 for r in rows]) if rows else 0.0


def report(rows, w_grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5)):
    print(f'\n  rankable frames: {len(rows)}')
    print(f'  baseline (global rescoring)      rank-1 {100*r1(rows, lambda r: np.array(r["base"])):5.1f}%')
    print(f'  localized appearance ALONE       rank-1 {100*r1(rows, lambda r: np.array(r["loc_app"])):5.1f}%')
    print(f'  localized residual ALONE         rank-1 {100*r1(rows, lambda r: np.array(r["loc_res"])):5.1f}%')
    print()
    best = (0.0, 0.0, 'none')
    for key in ('loc_app', 'loc_res'):
        for w in w_grid:
            a = r1(rows, lambda r, k=key, w=w: np.array(r['base'])
                   + w * np.clip(L._spread_z(np.array(r[k])), 0, 12) / 12.0)
            if w:
                print(f'    base + {w:4.2f}*{key:8s}  rank-1 {100*a:5.1f}%')
            if a > best[0]:
                best = (a, w, key)
    print(f'\n  best: base + {best[1]:.2f}*{best[2]}  ->  {100*best[0]:.1f}%')
    return best


def by_landmark(rows, w, key):
    d = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        b = r['y'][int(np.argmax(np.array(r['base'])))] == 1
        f = np.array(r['base']) + w * np.clip(L._spread_z(np.array(r[key])), 0, 12) / 12.0
        m = r['y'][int(np.argmax(f))] == 1
        k = r['landmark'] or 'none'
        d[k][0] += 1; d[k][1] += int(b); d[k][2] += int(m)
    print(f'\n  {"landmark":15s} {"n":>4s} {"global":>9s} {"localized":>10s}')
    for k in sorted(d, key=lambda x: -d[x][0]):
        n, b, m = d[k]
        print(f'  {k:15s} {n:4d} {100*b/n:8.1f}% {100*m/n:9.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset_primary')
    ap.add_argument('--box', type=float, default=2.0,
                    help='window side, in reference-scale lattice pitches')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--w', type=float, default=None,
                    help="report this fusion weight, fixed in advance on the "
                         "selection split, instead of this split's own best")
    ap.add_argument('--key', default='loc_app', choices=('loc_app', 'loc_res'))
    args = ap.parse_args()

    if args.sweep:
        for b in (1.0, 1.5, 2.0, 3.0, 4.0):
            rows = run(args.dataset, b, args.limit)
            print(f'\n===== box = {b} pitches =====')
            report(rows)
        return
    rows = run(args.dataset, args.box, args.limit)
    print(f'\n===== {args.dataset}, box = {args.box} pitches =====')
    a, w, key = report(rows)
    if args.w is not None:
        # Reporting a setting fixed in advance on the SELECTION split, not the
        # best cell of this split's own grid -- otherwise the number is fitted.
        w, key = args.w, args.key
        held = r1(rows, lambda r: np.array(r['base'])
                  + w * np.clip(L._spread_z(np.array(r[key])), 0, 12) / 12.0)
        base = r1(rows, lambda r: np.array(r['base']))
        n = len(rows)
        print(f'\n  PRE-SELECTED setting: base + {w:.2f}*{key}  (chosen on the '
              f'selection split)')
        print(f'    baseline   {100*base:5.1f}%  ({int(round(base*n))}/{n})')
        print(f'    localized  {100*held:5.1f}%  ({int(round(held*n))}/{n})')
        print(f'    delta      {int(round(held*n)) - int(round(base*n)):+d} pairs')
    by_landmark(rows, w, key)


if __name__ == '__main__':
    main()
