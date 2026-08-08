"""Three point-source mechanisms, measured together.

The measured failure: a dropped via alters one lattice cell of ~78, so it is
1.3% of a whole-footprint NCC. It is lost twice -- 9 of the 10 sites that are
NEVER PROPOSED are via defects, and 9 more are proposed but out-ranked.

Nothing here smooths or pools (the shipped path is FFT + NCC, no morphology,
no pyramid), but the dilution has the same effect a smoother would, so the
same remedies apply:

  1. DoG PROPOSAL.  A Difference-of-Gaussians blob detector on the SEARCH
     residual, at the via's own scale. Each blob votes for a site centre:
     if the reference's landmark sits at offset d from the reference centre,
     a blob at b implies a centre at b - d. That is a Hough vote, and unlike
     NCC it is not diluted by the 98.7% of lattice that agrees everywhere.

  2. FOOTPRINT SALIENCY.  Local energy of the search residual, box-filtered
     over the footprint: "does this candidate's field of view contain any
     aperiodic content at all?" Computed from the SEARCH image alone, so it
     fails independently of every reference-matched channel.

  3. TOPOLOGY-AWARE / SNR SCORING.  For each candidate, look where THAT
     candidate's hypothesis puts the landmark and measure local peak-to-noise
     there. A point is judged on contrast against its immediate background
     instead of on agreement across an extent it does not have. Gated on how
     point-like the reference's landmark actually is, so extended gate bars
     and array edges are not scored by a rule built for dots.

    python probes/point_channel.py --dataset dataset_stress  --dump sel.json
    python probes/point_channel.py --analyse sel.json rep.json
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
from probes.local_rescore import landmark_site, BORDER_FRAC

TOL = 15.0
MAX_BLOBS = 12


def dog_blobs(res, sigma, k=1.6, top=MAX_BLOBS, nms=None):
    """Difference-of-Gaussians blob response, and its strongest peaks.

    Tuned to the via's own radius rather than to the lattice: the lattice is
    already spectrally absent from `res`, so what survives at this scale is
    a dropped or doubled contact."""
    a = np.abs(res.astype(np.float32))
    g1 = cv2.GaussianBlur(a, (0, 0), sigma)
    g2 = cv2.GaussianBlur(a, (0, 0), sigma * k)
    d = g1 - g2
    nms = nms or max(3, int(round(sigma * 3)))
    mx = cv2.dilate(d, np.ones((nms, nms), np.uint8))
    peaks = (d >= mx - 1e-9) & (d > d.mean() + d.std())
    ys, xs = np.nonzero(peaks)
    if not len(xs):
        return d, []
    vals = d[ys, xs]
    order = np.argsort(-vals)[:top]
    return d, [(float(xs[i]), float(ys[i]), float(vals[i])) for i in order]


def local_snr(res, x, y, rad):
    """Peak-to-noise of the residual in a small window -- the point's own
    contrast against its immediate background, not against an extent."""
    h, w = res.shape
    x0, x1 = int(max(0, x - rad)), int(min(w, x + rad + 1))
    y0, y1 = int(max(0, y - rad)), int(min(h, y + rad + 1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    a = np.abs(res[y0:y1, x0:x1]).astype(np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826 + 1e-6
    return (float(a.max()) - med) / mad


def landmark_extent(res_ref, pitch_ref):
    """How point-like is the reference's landmark? Area above half-peak,
    in units of a lattice cell. A dropped via is well under 1; a gate bar or
    an array edge runs to many."""
    n = res_ref.shape[0]
    k = max(3, int(round(pitch_ref / 3.0)) | 1)
    sm = cv2.GaussianBlur(np.abs(res_ref.astype(np.float32)), (k, k), 0)
    b = int(round(BORDER_FRAC * n))
    inner = sm[b:n - b, b:n - b]
    pk = float(inner.max())
    if pk <= 0:
        return 99.0
    area = float((inner >= 0.5 * pk).sum())
    return area / max(1.0, pitch_ref ** 2)


def build(dataset, limit=0):
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
        base_c = L._multiscale_peaks(s, r, win, feet, nms_radius=nms_r,
                                     top_k_per_scale=L.TOP_K_PER_SCALE)
        s_band, r_band = L.pitch_bands(pitch)
        res_s = L._envelope_normalise(aperiodic_residual(s, s_band))
        res_r = L._envelope_normalise(aperiodic_residual(r, r_band))
        lmap = L._landmark_map_from(res_s, res_r, foot_ref, win)
        if lmap is not None:
            base_c += L._peaks_from_map(lmap, win, foot_ref, nms_radius=nms_r)
        win_area = max(1.0, (win[2] - win[0]) * (win[3] - win[1]))
        keep = int(np.clip(round(15 * win_area / (360.0 ** 2)), 15, L.MAX_CANDIDATES))
        base_c = L._dedupe(base_c, radius=nms_r, keep=keep)

        d_base = np.array([np.hypot(c['x'] - gx, c['y'] - gy) for c in base_c])
        proposed_base = bool((d_base <= TOL).any()) if len(d_base) else False

        # ---- 1. DoG proposals: each blob votes for a centre --------------
        pitch_ref = pitch * scale_est
        lx, ly, _, _ = landmark_site(res_r, pitch_ref)
        n = r.shape[0]
        foot_exact = n / float(scale_est)
        kk = foot_exact / n                       # ref px -> search px
        ox, oy = (lx - n / 2.0) * kk, (ly - n / 2.0) * kk
        via_sigma = max(1.5, pitch / 4.0)
        dog, blobs = dog_blobs(res_s, via_sigma)
        voted = [dict(x=float(bx - ox), y=float(by - oy), score=0.0,
                      foot=float(foot_exact), src='dog')
                 for bx, by, _ in blobs]
        voted = [c for c in voted
                 if 0 <= c['x'] < s.shape[1] and 0 <= c['y'] < s.shape[0]]

        merged = L._dedupe(base_c + voted, radius=nms_r, keep=keep + len(voted))
        d = np.array([np.hypot(c['x'] - gx, c['y'] - gy) for c in merged])
        proposed_merged = bool((d <= TOL).any()) if len(d) else False

        merged = L.rescore_fullres(s, r, res_s, res_r, merged,
                                   foot_exact=foot_exact)

        # ---- 2. footprint saliency --------------------------------------
        sal = cv2.GaussianBlur(np.abs(res_s.astype(np.float32)), (0, 0),
                               max(1.0, pitch / 2.0))
        fb = int(max(3, round(foot_exact)) | 1)
        sal_fp = cv2.boxFilter(sal, -1, (fb, fb))

        # ---- 3. local SNR at each candidate's own landmark hypothesis ----
        snr_rad = int(max(4, round(via_sigma * 4)))
        feats = []
        for c in merged:
            cx, cy = c['x'], c['y']
            px, py = cx + ox, cy + oy
            ix = int(np.clip(round(cx), 0, s.shape[1] - 1))
            iy = int(np.clip(round(cy), 0, s.shape[0] - 1))
            feats.append((float(sal_fp[iy, ix]),
                          local_snr(res_s, px, py, snr_rad),
                          float(dog[int(np.clip(round(py), 0, s.shape[0] - 1)),
                                    int(np.clip(round(px), 0, s.shape[1] - 1))])))
        feats = np.array(feats) if feats else np.zeros((0, 3))

        base = (L.W_APPEARANCE * np.array([c['score'] for c in merged])
                + L.W_FINE_APPEARANCE * np.array([c['fine_appearance'] for c in merged])
                + L.W_LANDMARK * np.clip(
                    L._spread_z(np.array([c['fine_landmark'] for c in merged])), 0, 12) / 12.0)

        rows.append(dict(
            pair_id=m['pair_id'], landmark=m.get('landmark'),
            proposed_base=proposed_base, proposed_merged=proposed_merged,
            extent=landmark_extent(res_r, pitch_ref),
            y=(d <= TOL).astype(int).tolist(), base=base.tolist(),
            sal=feats[:, 0].tolist(), snr=feats[:, 1].tolist(),
            dogv=feats[:, 2].tolist(),
            n_dog=len(voted)))
        print(f'  pair {m["pair_id"]:3d} {m.get("landmark","-"):14s} '
              f'base={int(proposed_base)} merged={int(proposed_merged)} '
              f'extent={rows[-1]["extent"]:.2f}', flush=True)
    return rows


# ------------------------------------------------------------- analysis ----
def z(v):
    return np.clip(L._spread_z(np.array(v)), 0, 12) / 12.0


def score(r, w_sal, w_snr, gate):
    f = np.array(r['base'])
    point_like = r['extent'] <= gate if gate else True
    if w_sal:
        f = f + w_sal * z(r['sal'])
    if w_snr and point_like:
        f = f + w_snr * z(r['snr'])
    return f


def r1(rows, **kw):
    ok = [r for r in rows if max(r['y']) > 0]
    if not ok:
        return 0.0
    return np.mean([r['y'][int(np.argmax(score(r, **kw)))] == 1 for r in ok])


def analyse(sel_path, rep_path):
    sel = json.load(open(sel_path))
    rep = json.load(open(rep_path))

    print(f'\n=== 1. PROPOSAL: does the DoG vote recover unproposed sites? ===')
    for name, rows in (('selection', sel), ('report', rep)):
        b = sum(r['proposed_base'] for r in rows)
        m = sum(r['proposed_merged'] for r in rows)
        n = len(rows)
        gained = [r for r in rows if r['proposed_merged'] and not r['proposed_base']]
        lost = [r for r in rows if r['proposed_base'] and not r['proposed_merged']]
        print(f'  {name:9s} n={n:3d}   baseline {100*b/n:5.1f}%  '
              f'+DoG {100*m/n:5.1f}%   gained {len(gained)}, lost {len(lost)}')
        if gained:
            print(f'            recovered: '
                  + ', '.join(f'{r["pair_id"]}({r["landmark"]})' for r in gained[:8]))

    print(f'\n=== 2/3. RANKING: saliency and SNR channels ===')
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    gates = [None, 5.0, 10.0, 15.0, 25.0]   # scaled to observed extents
    best = (-1, None)
    for g in gates:
        for ws in grid:
            for wn in grid:
                a = r1(sel, w_sal=ws, w_snr=wn, gate=g)
                if a > best[0]:
                    best = (a, dict(w_sal=ws, w_snr=wn, gate=g))
    base_sel = r1(sel, w_sal=0, w_snr=0, gate=None)
    print(f'  selection split: baseline {100*base_sel:5.1f}%  ->  '
          f'{100*best[0]:5.1f}% at {best[1]}')

    cfg = best[1]
    base_rep = r1(rep, w_sal=0, w_snr=0, gate=None)
    got_rep = r1(rep, **cfg)
    nr = len([r for r in rep if max(r['y']) > 0])
    print(f'\n  HELD OUT (setting fixed above, n={nr})')
    print(f'    baseline  {100*base_rep:5.1f}%  ({int(round(base_rep*nr))}/{nr})')
    print(f'    +channels {100*got_rep:5.1f}%  ({int(round(got_rep*nr))}/{nr})')
    print(f'    delta     {int(round(got_rep*nr)) - int(round(base_rep*nr)):+d} pairs')

    d = defaultdict(lambda: [0, 0, 0])
    for r in rep:
        if max(r['y']) == 0:
            continue
        k = r['landmark'] or 'none'
        d[k][0] += 1
        d[k][1] += int(r['y'][int(np.argmax(score(r, w_sal=0, w_snr=0, gate=None)))] == 1)
        d[k][2] += int(r['y'][int(np.argmax(score(r, **cfg)))] == 1)
    print(f'\n  {"landmark":15s} {"n":>4s} {"base":>8s} {"+chan":>8s}')
    for k in sorted(d, key=lambda x: -d[x][0]):
        n_, b_, m_ = d[k]
        print(f'  {k:15s} {n_:4d} {100*b_/n_:7.1f}% {100*m_/n_:7.1f}%')

    print('\n  landmark extent by type (point-likeness; <1 is sub-cell)')
    e = defaultdict(list)
    for r in rep:
        e[r['landmark'] or 'none'].append(r['extent'])
    for k, v in e.items():
        print(f'    {k:15s} median {np.median(v):6.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset')
    ap.add_argument('--dump')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--analyse', nargs=2, metavar=('SEL', 'REP'))
    args = ap.parse_args()
    if args.analyse:
        analyse(*args.analyse)
        return
    rows = build(args.dataset, args.limit)
    json.dump(rows, open(args.dump, 'w'))
    print(f'\nwrote {args.dump}  ({len(rows)} frames)')


if __name__ == '__main__':
    main()
