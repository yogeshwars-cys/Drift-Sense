"""Cache the candidate pool + per-candidate refined position for each pair,
under several front-end variants, so decision rules can be swept offline.

variants:
  base      shipped front end (measured scale -> foot_bracket, rotation A/B)
  fix5      + landmark candidates get a real NCC appearance score
  fix6      fix5, rotation A/B disabled
  fix7      fix5, wider scale bracket (span floor raised)
  ncc       fix5, plain linspace scale sweep instead of foot_bracket
"""
import json, sys, os
import numpy as np
import cv2

sys.path.insert(0, r'D:\Downloads\projects\semicon')
import localize as L
from lattice import (estimate_scale, scale_uncertainty, foot_bracket, phase_lock,
                     aperiodic_residual, relative_rotation, rotate, snap_to_phase)

OUT = r'C:\Users\yoges\AppData\Local\Temp\claude\D--Downloads-projects-semicon\e851f7c8-fdb4-4c14-8b50-23b2f7a17ac2\scratchpad\cands.json'


def build(g, variant):
    ref = L.load_gray(g['ref_path']); sea = L.load_gray(g['search_path'])
    H, W = sea.shape; ctr = (W / 2., H / 2.)
    se, ls, lr, okf = estimate_scale(sea, ref)
    pitch = float(ls['pitch'])
    span = scale_uncertainty(ls, lr if okf else None, se)
    if not okf:
        span = max(span, 1.5)
    if variant == 'fix7':
        span = max(span, 0.35)           # never bracket tighter than +-3.5%
    fr = max(20, int(round(ref.shape[0] / se)))
    if variant == 'ncc':
        feet = sorted({int(round(ref.shape[0] / s)) for s in np.linspace(8.5, 11.5, 13)})
    else:
        feet = foot_bracket(ref.shape[0], se, span)

    ref_c = ref
    if variant != 'fix6':
        cd, _ = relative_rotation(sea, ref, pitch, se)
        if abs(cd) > 0.15:
            rt = rotate(ref, cd)
            if L._best_ncc(sea, rt, fr) > L._best_ncc(sea, ref, fr) + L.ROTATION_MIN_GAIN:
                ref_c = rt

    nms = int(max(3, round(L.NMS_PITCH_FRACTION * pitch)))
    win = L._window(sea.shape, ctr, max(H, W) + fr)
    cands = L._multiscale_peaks(sea, ref_c, win, feet, nms_radius=nms,
                                top_k_per_scale=L.TOP_K_PER_SCALE)
    lock = phase_lock(sea, ref_c, pitch, se)
    sb, rb = L.pitch_bands(pitch)
    rs = L._envelope_normalise(aperiodic_residual(sea, sb))
    rr = L._envelope_normalise(aperiodic_residual(ref_c, rb))
    lmap = L._landmark_map_from(rs, rr, fr, win)
    lmc = L._peaks_from_map(lmap, win, fr, nms_radius=nms) if lmap is not None else []
    if variant != 'base' and lmc:
        tpl = cv2.resize(ref_c, (fr, fr), interpolation=cv2.INTER_AREA)
        for x in lmc:
            crop = L._upsampled_crop(sea, x['x'], x['y'], fr, fr)
            x['score'] = 0.0 if crop is None else L._ncc(crop, tpl)
    cands = cands + lmc

    ka = max(1.0, (win[2] - win[0]) * (win[3] - win[1]))
    keep = int(np.clip(round(15 * ka / 360.0 ** 2), 15, L.MAX_CANDIDATES))
    cands = L._dedupe(cands, radius=nms, keep=keep)
    for x in cands:
        x['phase_shift'] = 0.0
    cands = L.rescore_fullres(sea, ref_c, rs, rr, cands,
                              foot_exact=ref.shape[0] / float(se))

    locked = bool(lock and (lock['ok_x'] or lock['ok_y']))
    out = []
    for c in cands:
        if locked:
            x, y, _, _ = snap_to_phase(c['x'], c['y'], c['foot'], lock)
        else:
            x, y = L._subpixel(sea, ref_c, c['x'], c['y'], c['foot'])
        out.append(dict(x=float(c['x']), y=float(c['y']),
                        rx=float(x), ry=float(y),
                        score=float(c['score']),
                        fa=float(c['fine_appearance']),
                        fl=float(c['fine_landmark'])))
    return dict(pair_id=g['pair_id'], difficulty=g['difficulty'],
                gt_x=g['gt_x'], gt_y=g['gt_y'], cx=ctr[0], cy=ctr[1],
                pitch=pitch, cands=out)


def main():
    gt = json.load(open(r'D:\Downloads\projects\semicon\dataset_primary\ground_truth.json'))
    variants = sys.argv[1:] or ['base', 'fix5', 'fix6', 'fix7', 'ncc']
    all_out = {}
    if os.path.exists(OUT):
        all_out = json.load(open(OUT))
    for v in variants:
        rows = []
        for g in gt:
            rows.append(build(g, v))
            print('.', end='', flush=True)
        all_out[v] = rows
        print(f' {v} done', flush=True)
        json.dump(all_out, open(OUT, 'w'))


if __name__ == '__main__':
    main()
