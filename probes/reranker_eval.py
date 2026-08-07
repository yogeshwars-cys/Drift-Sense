"""Does the learned re-ranker ADD anything to the classical channels?

This is the comparison `train_reranker.py`'s docstring says must be run before
any DL result is quoted, and it is set up so that no split does double duty:

    train the weights on   dataset_train    (seed 77)
    select the fusion weight on  dataset_stress   (seed 22)
    report on              dataset_primary  (seed 11)   <- the headline split

Three questions, in increasing order of what would actually justify shipping:

  1. Is the learned channel better ALONE than the classical matched filter?
     (rank-1 of the true site, per channel)
  2. Is it COMPLEMENTARY -- right on trials where the classical fusion is
     wrong? A weaker-but-decorrelated channel is worth more additively than a
     stronger-but-redundant one, and marginal accuracy alone cannot tell those
     apart. This is the same test that earned `induction.py` its place.
  3. Does adding it at its best weight actually move rank-1 on a split where
     that weight was not chosen?

Two-phase, because re-running the pipeline per weight costs minutes and the
channels do not change when the weight does:

    python probes/reranker_eval.py --dataset dataset_stress  --dump stress.json
    python probes/reranker_eval.py --dataset dataset_primary --dump primary.json
    python probes/reranker_eval.py --replay stress.json primary.json

Requires torch (requirements-train.txt); the shipped inference path does not.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

TOL = 15.0
WEIGHTS_GRID = [round(0.1 * i, 2) for i in range(0, 21)]


# ---------------------------------------------------------------- compute ----
def compute(dataset, weights_path, limit=0):
    import cv2, torch
    import localize as L
    from lattice import (estimate_scale, aperiodic_residual, scale_uncertainty,
                         foot_bracket, phase_lock, snap_to_phase,
                         relative_rotation, rotate)
    from reranker_model import EmbedNet, batch_to_tensor
    from train_reranker import _to_u8, crop

    model = EmbedNet()
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    def cnn_scores(res_s_u8, res_r_u8, cands):
        patches, keep = [], []
        for i, c in enumerate(cands):
            p = crop(res_s_u8, c['x'], c['y'], c['foot'])
            if p is not None:
                patches.append(p)
                keep.append(i)
        out = np.full(len(cands), -1.0)
        if not patches:
            return out
        with torch.no_grad():
            z_ref = model(batch_to_tensor([res_r_u8]))
            z_c = model(batch_to_tensor(patches))
            out[keep] = (z_c @ z_ref.T).squeeze(1).cpu().numpy()
        return out

    recs = json.load(open(f'{dataset}/ground_truth.json'))
    if limit:
        recs = recs[:limit]
    dump = []

    for m in recs:
        if not m.get('landmark_in_fov'):
            continue        # unidentifiable by construction; no channel can win
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
        res_s_raw = aperiodic_residual(s, s_band)
        res_r_raw = aperiodic_residual(r, r_band)
        res_s = L._envelope_normalise(res_s_raw)
        res_r = L._envelope_normalise(res_r_raw)
        lmap = L._landmark_map_from(res_s, res_r, foot_ref, win)
        if lmap is not None:
            cands += L._peaks_from_map(lmap, win, foot_ref, nms_radius=nms_r)
        win_area = max(1.0, (win[2] - win[0]) * (win[3] - win[1]))
        keep = int(np.clip(round(15 * win_area / (360.0 ** 2)), 15, L.MAX_CANDIDATES))
        cands = L._dedupe(cands, radius=nms_r, keep=keep)

        lock = phase_lock(s, r, pitch, scale_est)
        if lock['ok_x'] or lock['ok_y']:
            for c in cands:
                sx, sy, dx, dy = snap_to_phase(c['x'], c['y'], c['foot'], lock)
                c['x'], c['y'] = sx, sy
        cands = L.rescore_fullres(s, r, res_s, res_r, cands,
                                  foot_exact=r.shape[0] / float(scale_est))

        d = np.array([np.hypot(c['x'] - gx, c['y'] - gy) for c in cands])
        if not (d <= TOL).any():
            continue        # never proposed; no ranking channel can recover it

        cnn = cnn_scores(_to_u8(res_s_raw), _to_u8(res_r_raw), cands)
        dump.append(dict(
            pair_id=m['pair_id'], style=m['style'],
            landmark=m.get('landmark'), true_i=int(np.argmin(d)),
            coarse=[float(c['score']) for c in cands],
            fine_app=[float(c['fine_appearance']) for c in cands],
            fine_lm=[float(c['fine_landmark']) for c in cands],
            cnn=cnn.tolist(),
        ))
        print(f'  pair {m["pair_id"]:3d} {m["style"]:6s} '
              f'{len(cands):3d} candidates', flush=True)
    return dump


# ----------------------------------------------------------------- replay ----
def _fused(row, w_cnn, L):
    coarse = np.array(row['coarse'])
    fapp = np.array(row['fine_app'])
    lm_z = L._spread_z(np.array(row['fine_lm']))
    f = (L.W_APPEARANCE * coarse + L.W_FINE_APPEARANCE * fapp
         + L.W_LANDMARK * np.clip(lm_z, 0, 12) / 12.0)
    if w_cnn:
        f = f + w_cnn * np.clip(L._spread_z(np.array(row['cnn'])), 0, 12) / 12.0
    return f


def rank1(rows, w_cnn, L):
    return np.array([int((_fused(r, w_cnn, L) > _fused(r, w_cnn, L)[r['true_i']]).sum()) == 0
                     for r in rows])


def replay(select_path, report_path):
    import localize as L
    sel = json.load(open(select_path))
    rep = json.load(open(report_path))

    print(f'\nweight selected on : {select_path}  (n={len(sel)})')
    print(f'reported on        : {report_path}  (n={len(rep)})\n')

    # --- 1. standalone channels, on the reporting split
    print('  standalone rank-1 of the true site among proposed candidates')
    for key, tag in (('coarse', ''), ('fine_app', ''),
                     ('fine_lm', '  <- shipped judge'), ('cnn', '  <- learned')):
        r1 = np.array([int((np.array(r[key]) > np.array(r[key])[r['true_i']]).sum()) == 0
                       for r in rep])
        print(f'    {key:10s} {100*r1.mean():6.1f}%{tag}')

    # --- 2. complementarity: is it right where the classical fusion is wrong?
    base = rank1(rep, 0.0, L)
    cnn_only = np.array([int((np.array(r['cnn']) > np.array(r['cnn'])[r['true_i']]).sum()) == 0
                         for r in rep])
    both = int((base & cnn_only).sum())
    only_c = int((base & ~cnn_only).sum())
    only_n = int((~base & cnn_only).sum())
    neither = int((~base & ~cnn_only).sum())
    print(f'\n  complementarity (n={len(rep)})')
    print(f'    both correct                  {both:3d}')
    print(f'    classical only                {only_c:3d}')
    print(f'    CNN only  <- the headroom     {only_n:3d}')
    print(f'    neither                       {neither:3d}')
    if only_n == 0:
        print('    -> the CNN is right nowhere the classical fusion is not.')
        print('       Strictly redundant: additive fusion has nothing to add.')

    # --- 3. weight chosen on the selection split, applied to the reporting one
    curve = [(w, rank1(sel, w, L).mean()) for w in WEIGHTS_GRID]
    best_w = max(curve, key=lambda t: (t[1], -t[0]))[0]
    print(f'\n  fusion-weight sweep on the SELECTION split:')
    for w, a in curve:
        if w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0) or w == best_w:
            mark = '  <- best' if w == best_w else ''
            print(f'    w_cnn={w:4.2f}  rank-1 {100*a:5.1f}%{mark}')

    held = rank1(rep, best_w, L)
    print(f'\n  HELD-OUT RESULT at w_cnn={best_w:.2f}')
    print(f'    without CNN  {100*base.mean():5.1f}%  ({base.sum()}/{len(rep)})')
    print(f'    with CNN     {100*held.mean():5.1f}%  ({held.sum()}/{len(rep)})')
    delta = int(held.sum() - base.sum())
    print(f'    delta        {delta:+d} pairs')
    if best_w == 0.0:
        print('\n  VERDICT: the sweep chose weight 0 on a split the CNN never saw.')
        print('           It does not ship.')
    elif abs(delta) <= 2:
        print('\n  VERDICT: within sampling noise at this n. Not evidence to ship.')
    elif delta > 2:
        print('\n  VERDICT: the learned channel adds real signal. Ship it.')
    else:
        print('\n  VERDICT: the learned channel actively hurts. Do not ship.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset')
    ap.add_argument('--weights', default='reranker.pt')
    ap.add_argument('--dump')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--replay', nargs=2, metavar=('SELECT', 'REPORT'))
    args = ap.parse_args()

    if args.replay:
        replay(*args.replay)
        return
    if not (args.dataset and args.dump):
        ap.error('give --dataset and --dump, or --replay SELECT REPORT')
    dump = compute(args.dataset, args.weights, args.limit)
    json.dump(dump, open(args.dump, 'w'))
    print(f'\nwrote {args.dump}  ({len(dump)} scorable trials)')


if __name__ == '__main__':
    main()
