"""Same uniform-placement pairs, but search the FULL frame (no disk prior).
Quantifies how much of the loss is the mis-specified prior vs. the matcher."""
import sys, os, json, time
sys.path.insert(0, r'D:\Downloads\projects\semicon\drift_sense')
os.chdir(r'D:\Downloads\projects\semicon\drift_sense')
import numpy as np
import dataset_generator as G
import localize as L

MARGIN = G.MARGIN
_state = {'rng': None}
G.clamp_site = lambda dx, dy: (float(_state['rng'].uniform(MARGIN, G.SEARCH_SIZE - MARGIN)),
                               float(_state['rng'].uniform(MARGIN, G.SEARCH_SIZE - MARGIN)))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
VARIANTS = {
    'disk180 (shipped default)': dict(),
    'full frame, prior off':     dict(drift_radius=900.0, w_prior=0.0),
    'full frame, no landmark':   dict(drift_radius=900.0, w_prior=0.0, use_landmark=False),
}

rng = np.random.default_rng(2024); _state['rng'] = rng
pairs, counts = [], {'dram': 0, 'finfet': 0}
for i in range(N):
    style = 'dram' if i % 2 == 0 else 'finfet'
    diff = 'easy' if counts[style] % 2 == 0 else 'hard'
    counts[style] += 1
    gen = G.generate_dram_pair if style == 'dram' else G.generate_finfet_pair
    pairs.append(gen(rng, i, diff))

for label, kw in VARIANTS.items():
    errs, sol, uns, rts = [], [], [], []
    for search, ref, meta in pairs:
        t0 = time.time()
        x, y, diag = L.localize(ref, search, **kw)
        rts.append(time.time() - t0)
        err = float(np.hypot(x - meta['gt_x'], y - meta['gt_y']))
        errs.append(err)
        (sol if meta['landmark_in_fov'] else uns).append(err)
    e = np.array(errs)
    print(f'\n{label}:  n={len(e)}')
    print('   ' + '  '.join(f'<={t}px {np.mean(e<=t)*100:5.1f}%' for t in (15, 25, 50, 100)))
    print(f'   median={np.median(e):.1f}px  runtime_med={np.median(rts):.2f}s')
    print(f'   solvable n={len(sol)} <=15px {np.mean(np.array(sol)<=15)*100:5.1f}%   '
          f'unsolvable n={len(uns)} <=15px {np.mean(np.array(uns)<=15)*100:5.1f}%')
