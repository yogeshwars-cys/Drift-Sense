"""Where is the true site lost -- PROPOSAL or RANKING?

For each pair, rebuild localize.py's candidate pool and ask:
  * was the true site proposed at all (within tol of some candidate)?
  * if so, what rank does each scoring channel give it?

A pipeline that proposes the true site but ranks it 5th has a scoring problem;
one that never proposes it has a search problem. They need opposite fixes, and
an end-to-end accuracy number cannot tell them apart.

    python probes/rank_probe.py dataset_primary [n]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import localize as L
from lattice import (estimate_scale, aperiodic_residual, scale_uncertainty,
                     foot_bracket, phase_lock, snap_to_phase, relative_rotation,
                     rotate)

TOL = 15.0
NO_SNAP = '--no-snap' in sys.argv
DUMP = []


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    ds = args[0] if args else 'dataset_primary'
    recs = json.load(open(f'{ds}/ground_truth.json'))
    if len(args) > 1:
        recs = recs[:int(args[1])]

    stat = dict(n=0, proposed=0, solvable=0, solvable_proposed=0, locked=0)
    ranks = {k: [] for k in ('coarse', 'fine_app', 'fine_lm', 'fused')}
    miss_rows = []

    for m in recs:
        s = cv2.imread(m['search_path'], cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(m['ref_path'], cv2.IMREAD_GRAYSCALE)
        gx, gy = m['gt_x'], m['gt_y']
        solvable = bool(m.get('landmark_in_fov'))
        stat['n'] += 1
        stat['solvable'] += solvable

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
        if NO_SNAP:
            lock = dict(lock, ok_x=False, ok_y=False)
        if lock['ok_x'] or lock['ok_y']:
            stat['locked'] += 1
            for c in cands:
                sx, sy, dx, dy = snap_to_phase(c['x'], c['y'], c['foot'], lock)
                c['x'], c['y'] = sx, sy
                c['phase_shift'] = float(np.hypot(dx, dy))
        else:
            for c in cands:
                c['phase_shift'] = 0.0

        cands = L.rescore_fullres(s, r, res_s, res_r, cands,
                                  foot_exact=r.shape[0] / float(scale_est))
        d = np.array([np.hypot(c['x'] - gx, c['y'] - gy) for c in cands])
        hit = d <= TOL
        if not hit.any():
            miss_rows.append((m['pair_id'], m['style'], solvable, float(d.min())))
            continue
        stat['proposed'] += 1
        stat['solvable_proposed'] += solvable
        true_i = int(np.argmin(d))

        coarse = np.array([c['score'] for c in cands])
        fapp = np.array([c['fine_appearance'] for c in cands])
        flm = np.array([c['fine_landmark'] for c in cands])
        lm_z = L._spread_z(flm)
        dist = np.array([np.hypot(c['x'] - s.shape[1] / 2.0,
                                  c['y'] - s.shape[0] / 2.0) for c in cands])
        fused = (L.W_APPEARANCE * coarse + L.W_FINE_APPEARANCE * fapp
                 + L.W_LANDMARK * np.clip(lm_z, 0, 12) / 12
                 + L.W_PRIOR * (-dist / max(s.shape))
                 + L.W_PHASE * (-np.array([c['phase_shift'] for c in cands]) / max(pitch, 1)))
        for key, v in (('coarse', coarse), ('fine_app', fapp),
                       ('fine_lm', flm), ('fused', fused)):
            ranks[key].append(int((v > v[true_i]).sum()) + 1)

        DUMP.append(dict(pair_id=m['pair_id'], style=m['style'],
                         solvable=solvable, true_i=true_i,
                         coarse=coarse.tolist(), fine_app=fapp.tolist(),
                         fine_lm=flm.tolist(), lm_z=lm_z.tolist(),
                         dist=dist.tolist(),
                         phase_shift=[c['phase_shift'] for c in cands],
                         pitch=pitch, diag=float(max(s.shape))))

    n, pr = stat['n'], stat['proposed']
    print(f'\n{ds}  n={n}   snap={"OFF" if NO_SNAP else "ON"}  '
          f'phase_locked={stat["locked"]}/{n}')
    print(f'  true site PROPOSED in pool: {pr}/{n} ({pr/n*100:.1f}%)')
    if stat['solvable']:
        print(f'    of solvable trials:       {stat["solvable_proposed"]}/'
              f'{stat["solvable"]} ({stat["solvable_proposed"]/stat["solvable"]*100:.1f}%)')
    print(f'\n  rank of the true site among proposed candidates (1 = correct):')
    for key, v in ranks.items():
        if not v:
            continue
        v = np.array(v)
        print(f'    {key:9s} rank1={np.mean(v == 1)*100:5.1f}%  '
              f'top3={np.mean(v <= 3)*100:5.1f}%  median_rank={np.median(v):.0f}')
    if miss_rows:
        print(f'\n  NOT PROPOSED ({len(miss_rows)}), nearest candidate distance:')
        for pid, st, sol, dmin in miss_rows[:12]:
            print(f'    pair {pid:3d} {st:6s} solvable={sol}  nearest={dmin:7.1f}px')

    out = 'probes/channel_scores.json'
    json.dump(DUMP, open(out, 'w'))
    print(f'\n  per-candidate channel scores -> {out}  '
          f'({len(DUMP)} trials; sweep weights offline with weight_sweep.py)')


if __name__ == '__main__':
    main()
