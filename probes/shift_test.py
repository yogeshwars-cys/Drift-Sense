"""Distribution-shift probe: does localize.py survive uniform placement of the
true site ("somewhere inside" the search image, per the problem statement)?"""
import sys, os, json, time, math
sys.path.insert(0, r'D:\Downloads\projects\semicon\drift_sense')
os.chdir(r'D:\Downloads\projects\semicon\drift_sense')

import numpy as np
import cv2
import dataset_generator as G
import localize as L

MARGIN = G.MARGIN
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24

# --- override placement: uniform over the frame instead of a 120-220px annulus
def uniform_site(rng):
    return (float(rng.uniform(MARGIN, G.SEARCH_SIZE - MARGIN)),
            float(rng.uniform(MARGIN, G.SEARCH_SIZE - MARGIN)))

_orig_clamp = G.clamp_site
_state = {'rng': None}
def patched_clamp(dx, dy):
    return uniform_site(_state['rng'])
G.clamp_site = patched_clamp

rng = np.random.default_rng(2024)
_state['rng'] = rng

rows = []
counts = {'dram': 0, 'finfet': 0}
for i in range(N):
    style = 'dram' if i % 2 == 0 else 'finfet'
    diff = 'easy' if counts[style] % 2 == 0 else 'hard'
    counts[style] += 1
    gen = G.generate_dram_pair if style == 'dram' else G.generate_finfet_pair
    search, ref, meta = gen(rng, i, diff)
    t0 = time.time()
    x, y, diag = L.localize(ref, search)
    dt = time.time() - t0
    err = float(np.hypot(x - meta['gt_x'], y - meta['gt_y']))
    d_center = float(np.hypot(meta['gt_x'] - 500, meta['gt_y'] - 500))
    rows.append(dict(style=style, difficulty=diff, err=err, dt=dt,
                     d_center=d_center,
                     landmark_in_fov=bool(meta['landmark_in_fov']),
                     full_search=bool(diag['full_search']),
                     conf=float(diag['confidence']),
                     lz=float(diag['landmark_z'])))
    print(f'{i:3d} {style:6s} {diff:4s} d_ctr={d_center:6.1f} err={err:8.1f} '
          f'full={diag["full_search"]} conf={diag["confidence"]:.3f} '
          f'lm_fov={meta["landmark_in_fov"]}')

e = np.array([r['err'] for r in rows])
print('\n--- uniform placement, n=%d ---' % len(rows))
for t in (15, 25, 50, 100):
    print(f'  <={t:3d}px : {np.mean(e <= t)*100:5.1f}%')
print(f'  median={np.median(e):.1f}px  mean={e.mean():.1f}px')
sol = np.array([r['err'] for r in rows if r['landmark_in_fov']])
uns = np.array([r['err'] for r in rows if not r['landmark_in_fov']])
if len(sol):
    print(f'  solvable   n={len(sol)}  <=15px {np.mean(sol<=15)*100:.1f}%  med {np.median(sol):.1f}')
if len(uns):
    print(f'  unsolvable n={len(uns)}  <=15px {np.mean(uns<=15)*100:.1f}%  med {np.median(uns):.1f}')
print(f'  full_search fired on {sum(r["full_search"] for r in rows)}/{len(rows)}')
inside = [r for r in rows if r['d_center'] <= 180]
outside = [r for r in rows if r['d_center'] > 180]
for name, grp in (('gt inside 180px disk', inside), ('gt outside 180px disk', outside)):
    if grp:
        g = np.array([r['err'] for r in grp])
        print(f'  {name:24s} n={len(g):3d}  <=15px {np.mean(g<=15)*100:5.1f}%  med {np.median(g):8.1f}px')
print(f'  runtime median {np.median([r["dt"] for r in rows]):.2f}s')
json.dump(rows, open(r'C:\Users\yoges\.claude\jobs\bc2673f9\tmp\shift_rows.json', 'w'), indent=2)
