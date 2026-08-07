"""Dump per-candidate ranking features for the learned ranker.

The hand-set fusion in localize.py is a linear combination of three channels
with constants chosen by grid search. It cannot express interactions between
channels, cannot calibrate itself per frame, and expresses "candidates compete"
only through a single spread-z. This dumps everything a learned ranker would
need to do better, for one dataset:

  per-candidate   raw channel scores, and each one's z-score, rank and margin
                  WITHIN its own candidate set -- the set-level context that a
                  fixed linear form cannot represent
  per-frame       lattice confidence, ambiguity, induction, rotation confidence;
                  broadcast to every candidate so the model can learn to trust
                  different channels on different frames
  label           1 for the candidate within TOL of the true site, else 0

Every feature is computable at inference time. The label is training-only.

    python probes/rank_features.py --dataset dataset_train --out feats_train.json
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2

import localize as L
from lattice import (estimate_scale, aperiodic_residual, scale_uncertainty,
                     foot_bracket, phase_lock, snap_to_phase, relative_rotation,
                     rotate, residual_saliency)
from induction import induction_score

TOL = 15.0

# Channels that get the full set-level treatment (raw + z + rank + margin).
CHANNELS = ('coarse', 'fine_app', 'fine_lm', 't1_app', 't1_lm')


def _setwise(v):
    """Turn a raw channel into the four things that actually matter for ranking:
    its value, how far it stands out from its own set, where it ranks, and how
    much daylight there is to the next candidate."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    z = (v - med) / mad if mad > 1e-9 else np.zeros(n)
    order = np.argsort(-v)
    rank = np.empty(n)
    rank[order] = np.arange(n)
    srt = np.sort(v)[::-1]
    best, second = (srt[0], srt[1]) if n > 1 else (srt[0], srt[0])
    margin = np.where(v >= best - 1e-12, v - second, v - best)
    spread = (srt[0] - srt[-1]) or 1.0
    return np.stack([v, z, rank / max(1, n - 1), margin / spread], axis=1)


def frame_features(s, r):
    """Everything the pipeline knows about the FRAME before ranking starts."""
    scale_est, lat_s, lat_r, ok = estimate_scale(s, r)
    pitch = float(lat_s['pitch'])
    span = scale_uncertainty(lat_s, lat_r if ok else None, scale_est)
    if not ok:
        span = max(span, 1.5)
    # induction_score returns (score, per_step); the sign of the score is the
    # decision, and NaN means there was no periodic model to test at all.
    ind = induction_score(s, pitch)[0]
    ind = 0.0 if not np.isfinite(ind) else float(ind)
    return dict(scale_est=float(scale_est), pitch=pitch, span=float(span),
                scale_ok=float(bool(ok)), induction=ind), lat_s, lat_r, ok


def build(dataset, limit=0):
    recs = json.load(open(f'{dataset}/ground_truth.json'))
    if limit:
        recs = recs[:limit]
    out = []

    for m in recs:
        s = cv2.imread(m['search_path'], cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(m['ref_path'], cv2.IMREAD_GRAYSCALE)
        gx, gy = m['gt_x'], m['gt_y']

        ff, lat_s, lat_r, ok = frame_features(s, r)
        scale_est, pitch, span = ff['scale_est'], ff['pitch'], ff['span']
        feet = foot_bracket(r.shape[0], scale_est, span)
        foot_ref = max(20, int(round(r.shape[0] / scale_est)))
        nms_r = int(max(3, round(L.NMS_PITCH_FRACTION * pitch)))

        cand_deg, rot_conf = relative_rotation(s, r, pitch, scale_est)
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
        n_from_lmap = 0
        if lmap is not None:
            extra = L._peaks_from_map(lmap, win, foot_ref, nms_radius=nms_r)
            n_from_lmap = len(extra)
            cands += extra
        win_area = max(1.0, (win[2] - win[0]) * (win[3] - win[1]))
        keep = int(np.clip(round(15 * win_area / (360.0 ** 2)), 15, L.MAX_CANDIDATES))
        cands = L._dedupe(cands, radius=nms_r, keep=keep)

        lock = phase_lock(s, r, pitch, scale_est)
        locked = bool(lock['ok_x'] or lock['ok_y'])
        if locked:
            for c in cands:
                sx, sy, dx, dy = snap_to_phase(c['x'], c['y'], c['foot'], lock)
                c['x'], c['y'] = sx, sy
                c['phase_shift'] = float(np.hypot(dx, dy))
        else:
            for c in cands:
                c['phase_shift'] = 0.0

        cands = L.rescore_fullres(s, r, res_s, res_r, cands,
                                  foot_exact=r.shape[0] / float(scale_est))
        if not cands:
            continue

        raw = dict(
            coarse=[float(c['score']) for c in cands],
            fine_app=[float(c['fine_appearance']) for c in cands],
            fine_lm=[float(c['fine_landmark']) for c in cands],
            t1_app=[float(c.get('tier1_appearance', c['fine_appearance'])) for c in cands],
            t1_lm=[float(c.get('tier1_landmark', c['fine_landmark'])) for c in cands],
        )
        H, W = s.shape
        dist = np.array([np.hypot(c['x'] - W / 2.0, c['y'] - H / 2.0) for c in cands])
        d_true = np.array([np.hypot(c['x'] - gx, c['y'] - gy) for c in cands])

        blocks = [_setwise(raw[k]) for k in CHANNELS]
        extra = np.stack([
            dist / max(H, W),
            np.array([c['phase_shift'] for c in cands]) / max(pitch, 1.0),
            np.array([c['foot'] for c in cands]) / max(1.0, foot_ref),
        ], axis=1)
        n = len(cands)
        frame_vec = np.tile(np.array([
            ff['scale_ok'], np.tanh(ff['induction']), float(rot_conf),
            float(locked), span, np.log1p(n) / 5.0,
            n_from_lmap / max(1.0, n),
        ]), (n, 1))
        X = np.concatenate(blocks + [extra, frame_vec], axis=1)

        out.append(dict(
            pair_id=m['pair_id'], style=m['style'], landmark=m.get('landmark'),
            landmark_in_fov=bool(m.get('landmark_in_fov')),
            X=X.tolist(),
            y=(d_true <= TOL).astype(int).tolist(),
            baseline=(L.W_APPEARANCE * np.array(raw['coarse'])
                      + L.W_FINE_APPEARANCE * np.array(raw['fine_app'])
                      + L.W_LANDMARK * np.clip(L._spread_z(np.array(raw['fine_lm'])), 0, 12) / 12.0
                      ).tolist(),
        ))
        print(f'  pair {m["pair_id"]:3d} {m["style"]:6s} n={n:3d} '
              f'has_true={int((d_true <= TOL).any())}', flush=True)
    return out


FEATURE_NAMES = ([f'{c}_{s}' for c in CHANNELS for s in ('raw', 'z', 'rank', 'margin')]
                 + ['dist', 'phase_shift', 'foot_ratio']
                 + ['scale_ok', 'induction', 'rot_conf', 'phase_locked',
                    'span', 'log_n', 'frac_from_landmark'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    data = build(args.dataset, args.limit)
    json.dump(dict(features=FEATURE_NAMES, rows=data), open(args.out, 'w'))
    n_true = sum(1 for r in data if max(r['y']) > 0)
    print(f'\nwrote {args.out}: {len(data)} frames, {len(FEATURE_NAMES)} features, '
          f'true site present in {n_true}')


if __name__ == '__main__':
    main()
