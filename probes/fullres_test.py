"""Does matching at FULL reference resolution discriminate the true site from
the best wrong periodic repeat better than the decimated (~100px) match that
localize.py currently uses to rank?

For each pair: build the candidate pool exactly as localize.py does, then score
every candidate three ways and ask whether the TRUE site is ranked #1.

  decimated   : ref resized to foot(~100px), NCC vs raw search crop   [shipped]
  fullres     : search crop upsampled to ref size, NCC vs full ref
  fullres_res : same, but on the aperiodic residuals of both
"""
import sys, os, time
sys.path.insert(0, r'D:\Downloads\projects\semicon\drift_sense')
os.chdir(r'D:\Downloads\projects\semicon\drift_sense')
import numpy as np
import cv2
import dataset_generator as G
import localize as L
from lattice import estimate_scale, aperiodic_residual

MARGIN = G.MARGIN
_state = {'rng': None}
G.clamp_site = lambda dx, dy: (float(_state['rng'].uniform(MARGIN, G.SEARCH_SIZE - MARGIN)),
                               float(_state['rng'].uniform(MARGIN, G.SEARCH_SIZE - MARGIN)))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rng = np.random.default_rng(2024); _state['rng'] = rng
pairs, counts = [], {'dram': 0, 'finfet': 0}
for i in range(N):
    style = 'dram' if i % 2 == 0 else 'finfet'
    diff = 'easy' if counts[style] % 2 == 0 else 'hard'
    counts[style] += 1
    gen = G.generate_dram_pair if style == 'dram' else G.generate_finfet_pair
    pairs.append(gen(rng, i, diff))


def score_at(img_s, tmpl, x, y, foot, upscale_to=None):
    """NCC of the search crop at (x,y) against tmpl, optionally after
    upsampling the crop to tmpl's resolution."""
    half = foot / 2.0
    x0, y0 = int(round(x - half)), int(round(y - half))
    if x0 < 0 or y0 < 0 or x0 + foot > img_s.shape[1] or y0 + foot > img_s.shape[0]:
        return None
    crop = img_s[y0:y0 + foot, x0:x0 + foot]
    if upscale_to is not None:
        crop = cv2.resize(crop, (upscale_to, upscale_to), interpolation=cv2.INTER_CUBIC)
    a = crop.astype(np.float32); b = tmpl.astype(np.float32)
    if a.shape != b.shape:
        b = cv2.resize(b, a.shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float((a * b).sum() / d)


rank1 = dict(decimated=0, fullres=0, fullres_res=0)
n_eval = 0
times = dict(decimated=0.0, fullres=0.0, fullres_res=0.0)

for search, ref, meta in pairs:
    if not meta['landmark_in_fov']:
        continue                      # only solvable trials can be ranked at all
    H, W = search.shape
    scale_est, lat_s, lat_r, ok = estimate_scale(search, ref)
    foot = max(20, int(round(ref.shape[0] / scale_est)))
    win = (0, 0, W, H)
    cands = L._multiscale_peaks(search, ref, win,
                                (scale_est - 0.35, scale_est + 0.35), 5,
                                nms_radius=int(np.clip(round(2 * lat_s['pitch']), 20, 45)))
    lmap, _ = L._landmark_map(search, ref, foot, win)
    if lmap is not None:
        cands += L._peaks_from_map(lmap, win, foot,
                                   nms_radius=int(np.clip(round(2 * lat_s['pitch']), 20, 45)))
    cands = L._dedupe(cands, radius=40, keep=20)
    # inject the true site so ranking is always measurable
    if all(np.hypot(c['x'] - meta['gt_x'], c['y'] - meta['gt_y']) > 12 for c in cands):
        cands.append(dict(score=0.0, x=meta['gt_x'], y=meta['gt_y'], foot=foot))
    n_eval += 1

    res_s = aperiodic_residual(search, L.SEARCH_PITCH_BAND)
    res_r = aperiodic_residual(ref, L.REF_PITCH_BAND)
    ref_small = cv2.resize(ref, (foot, foot), interpolation=cv2.INTER_AREA)
    res_r_small = cv2.resize(res_r, (foot, foot), interpolation=cv2.INTER_AREA)

    for name, tmpl, img, up in (
            ('decimated',   ref_small,  search, None),
            ('fullres',     ref,        search, ref.shape[0]),
            ('fullres_res', res_r,      res_s,  ref.shape[0])):
        t0 = time.time()
        sc = [score_at(img, tmpl, c['x'], c['y'], foot, upscale_to=up) for c in cands]
        times[name] += time.time() - t0
        pairs_sc = [(s, c) for s, c in zip(sc, cands) if s is not None]
        if not pairs_sc:
            continue
        best = max(pairs_sc, key=lambda t: t[0])[1]
        if np.hypot(best['x'] - meta['gt_x'], best['y'] - meta['gt_y']) <= 15:
            rank1[name] += 1

print(f'\nsolvable trials evaluated: {n_eval}   (candidate pool = full-frame, ~20 sites)')
for k in rank1:
    print(f'  {k:12s} picks the TRUE site {rank1[k]:2d}/{n_eval}  '
          f'({rank1[k]/max(n_eval,1)*100:5.1f}%)   rescoring cost {times[k]/max(n_eval,1)*1000:6.1f} ms/pair')
