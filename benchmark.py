"""
Common benchmark for the standalone inference path.

Scores localize.py on the 100-pair evaluation set at several tolerances,
because the task specifies "within tolerance of true location" without fixing
the tolerance -- quoting a single number picked by us is how a result gets
flattered. Reports the solvable/unsolvable split too, since half of this set
is unidentifiable by construction and a headline accuracy that ignores that
is not comparable to anything.

    python benchmark.py --variant merged
    python benchmark.py --variant merged --no-landmark
    python benchmark.py --variant merged --drift-radius 900
"""
import argparse, json, time
import numpy as np
import cv2

import localize as L

TOLERANCES = (15.0, 25.0, 50.0, 100.0)


def run(records, **kw):
    rows = []
    for m in records:
        s = cv2.imread(m['search_path'], cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(m['ref_path'], cv2.IMREAD_GRAYSCALE)
        t0 = time.time()
        x, y, diag = L.localize(r, s, **kw)
        dt = time.time() - t0
        err = float(np.hypot(x - m['gt_x'], y - m['gt_y']))
        rows.append(dict(pair_id=m['pair_id'], style=m['style'],
                         difficulty=m['difficulty'],
                         landmark_in_fov=bool(m.get('landmark_in_fov', False)),
                         landmark_distance_px=float(
                             m.get('landmark_distance_px', float('inf'))),
                         error_px=err, runtime_s=dt,
                         # calibrate_gate.py consumes these rows directly, and
                         # needs the outcome it is calibrating against.
                         success=bool(err <= 15.0), **diag))
    return rows


def report(rows, label):
    err = np.array([r['error_px'] for r in rows])
    rt = np.array([r['runtime_s'] for r in rows])
    print(f'\n=== {label}   n={len(rows)} ===')
    print('  accuracy within tolerance:')
    for t in TOLERANCES:
        print(f'     <={t:5.0f}px : {np.mean(err <= t)*100:5.1f}%')
    print(f'  median_err={np.median(err):7.1f}px   mean_err={err.mean():7.1f}px')
    print(f'  runtime/pair: median={np.median(rt):.2f}s  mean={rt.mean():.2f}s')

    for key in ('style', 'difficulty'):
        print(f'  by {key}:')
        for v in sorted(set(r[key] for r in rows)):
            e = np.array([r['error_px'] for r in rows if r[key] == v])
            cells = '  '.join(f'<={t:.0f}px:{np.mean(e <= t)*100:5.1f}%' for t in TOLERANCES)
            print(f'     {v:8s} n={len(e):3d}  {cells}  median={np.median(e):7.1f}px')

    solv = np.array([r['error_px'] for r in rows if r['landmark_in_fov']])
    uns = np.array([r['error_px'] for r in rows if not r['landmark_in_fov']])
    print('  by solvability (landmark in FOV):')
    for name, e in (('solvable', solv), ('unsolvable', uns)):
        if len(e):
            cells = '  '.join(f'<={t:.0f}px:{np.mean(e <= t)*100:5.1f}%' for t in TOLERANCES)
            print(f'     {name:10s} n={len(e):3d}  {cells}  median={np.median(e):7.1f}px')

    # Continuous difficulty: accuracy against distance to the nearest landmark.
    # The solvable/unsolvable split above is a threshold on this axis; showing
    # the axis itself is what distinguishes a real degradation curve from an
    # asserted cliff.
    dists = np.array([r.get('landmark_distance_px', float('inf')) for r in rows])
    if np.isfinite(dists).any():
        print('  by distance to nearest landmark:')
        edges = [(0, 50), (50, 100), (100, 200), (200, np.inf)]
        for lo, hi in edges:
            sel = (dists >= lo) & (dists < hi)
            if sel.sum():
                e = err[sel]
                hi_s = 'inf' if not np.isfinite(hi) else f'{hi:.0f}'
                print(f'     {lo:3.0f}-{hi_s:>3s}px  n={int(sel.sum()):3d}  '
                      f'<=15px:{np.mean(e <= 15)*100:5.1f}%  median={np.median(e):7.1f}px')

    ok = err <= 15.0
    npk = np.array([r.get('n_near_peaks', 0) for r in rows])
    for k in (5,):
        sel = npk <= k
        if sel.sum():
            print(f'  confidence signal: n_near_peaks<={k} selects {int(sel.sum()):3d} sites, '
                  f'{np.mean(ok[sel])*100:.1f}% of them within 15px')
    # The replacement confidence signal: separation, not correlation height.
    amb = np.array([r.get('ambiguity', 0.0) for r in rows])
    for t in (1.2, 1.5):
        sel = amb >= t
        if sel.sum():
            print(f'  confidence signal: ambiguity>={t} selects {int(sel.sum()):3d} sites, '
                  f'{np.mean(ok[sel])*100:.1f}% of them within 15px')
    # Spatial induction. Reported with BOTH signs and against the union with
    # n_near_peaks, because its value is that it fails independently of the
    # candidate-set signals -- coverage gained matters more than its precision
    # in isolation. See induction.py.
    ind = np.array([r.get('induction_score', np.nan) for r in rows],
                   dtype=np.float64)
    if np.isfinite(ind).any():
        fails = np.isfinite(ind) & (ind < 0)
        passes = np.isfinite(ind) & (ind >= 0)
        if fails.any():
            print(f'  confidence signal: induction<0 selects {int(fails.sum()):3d} sites, '
                  f'{np.mean(ok[fails])*100:.1f}% of them within 15px')
        if passes.any():
            print(f'                     induction>=0 covers {int(passes.sum()):3d} sites, '
                  f'{np.mean(ok[passes])*100:.1f}% within 15px  (base {np.mean(ok)*100:.1f}%)')
        union = fails | (npk <= 5)
        both = fails & (npk <= 5)
        if union.any():
            print(f'  union induction<0 OR n_near_peaks<=5: {int(union.sum()):3d} sites, '
                  f'{np.mean(ok[union])*100:.1f}% within 15px')
        if both.any():
            print(f'  both  induction<0 AND n_near_peaks<=5: {int(both.sum()):3d} sites, '
                  f'{np.mean(ok[both])*100:.1f}% within 15px')

    pl = np.array([bool(r.get('phase_locked', False)) for r in rows])
    if pl.any():
        print(f'  phase lock engaged on {int(pl.sum()):3d}/{len(rows)} pairs, '
              f'{np.mean(ok[pl])*100:.1f}% of those within 15px')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    # No default here: an unset flag must inherit localize.py's own default,
    # not shadow it with a second copy that silently drifts out of sync.
    ap.add_argument('--drift-radius', type=float, default=None)
    ap.add_argument('--no-landmark', action='store_true')
    ap.add_argument('--no-dog', action='store_true')
    ap.add_argument('--no-phase-lock', action='store_true')
    ap.add_argument('--no-rotation', action='store_true')
    ap.add_argument('--label', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='score only the first N pairs (ablations)')
    args = ap.parse_args()

    recs = json.load(open(f'{args.dataset}/ground_truth.json'))
    if args.limit:
        recs = recs[:args.limit]
    kw = dict(use_landmark=not args.no_landmark,
              use_phase_lock=not args.no_phase_lock,
              use_rotation=not args.no_rotation,
              use_dog=not args.no_dog)
    if args.drift_radius is not None:
        kw['drift_radius'] = args.drift_radius
    radius = 'full-frame' if args.drift_radius is None else f'{args.drift_radius:.0f}px'
    label = args.label or (f'{args.dataset}  radius={radius}'
                           f'{"  NO-landmark" if args.no_landmark else ""}'
                           f'{"  NO-phase-lock" if args.no_phase_lock else ""}')
    rows = run(recs, **kw)
    report(rows, label)
    if args.out:
        json.dump(rows, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
