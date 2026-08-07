"""
End-to-end evaluation.

Processes the dataset in shuffled order, simulating a session where the
Digital Twin's radius estimate is refined online (Loop 1) after each site
using the confirmed true drift magnitude -- in a real deployment this
confirmation would come from a high-confidence match or an operator check,
not from ground truth directly; here we use ground truth as the oracle
signal to demonstrate the calibration trend cleanly.

Also produces an Attribution Matrix (Case A-D, mirrored from the original
architecture doc's AI-meta-loop table) classifying each result by whether
the Twin's prior was accurate and whether the final pick was correct --
this is the "failure mode awareness" the problem statement explicitly asks
teams to demonstrate.
"""
import argparse, json
import numpy as np
from PIL import Image
import torch

from common import SEARCH_SIZE
from digital_twin import DriftPrior
from reranker_model import EmbedNet
from matcher import match, NOMINAL_CENTER
import commit_gate


def FEATURE_MSG(g):
    return f'{g["feature"]} {g["sense"]} {g["threshold"]:.0f}'

SUCCESS_PX = 15.0        # sub-pixel-class success tolerance
TWIN_ACCURATE_PX = 40.0  # twin prior considered "accurate" if within this of true magnitude


def true_drift_mag(meta):
    return float(np.hypot(meta['gt_x'] - meta['nominal_x'], meta['gt_y'] - meta['nominal_y']))


def attribute(twin_error_px, match_error_px):
    twin_ok = twin_error_px <= TWIN_ACCURATE_PX
    match_ok = match_error_px <= SUCCESS_PX
    if twin_ok and match_ok:
        return 'A: twin accurate, match correct'
    if (not twin_ok) and match_ok:
        return 'C: twin off, matcher recovered anyway'
    if twin_ok and (not match_ok):
        return 'B: twin accurate, matcher still picked wrong periodic peak'
    return 'D: twin off AND matcher wrong (compounded failure)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--weights', default='reranker.pt')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--out', default='results.json')
    ap.add_argument('--gate', default=None,
                    help='calibrated commit gate from calibrate_gate.py; '
                         'falls back to the hand-set thresholds when absent')
    args = ap.parse_args()

    with open(f'{args.dataset}/ground_truth.json') as f:
        records = json.load(f)

    model = EmbedNet()
    try:
        model.load_state_dict(torch.load(args.weights))
    except FileNotFoundError:
        raise SystemExit(
            f'\nweights not found: {args.weights}\n\n'
            'The previously shipped reranker.pt was trained on RAW patches from\n'
            'the superseded dataset and is archived at\n'
            '  legacy/reranker_raw_patches_stale.pt\n'
            'It is not valid for the current pipeline. Retrain with:\n'
            '  python train_reranker.py --dataset dataset_primary --out reranker.pt\n\n'
            'Note that evaluate.py exercises the RESEARCH path (matcher.py, with\n'
            'the learned re-ranker and the Digital Twin drift prior). The SHIPPED\n'
            'inference path is localize.py, which is torch-free and needs no\n'
            'weights -- score that with benchmark.py.\n')
    model.eval()

    gate = commit_gate.load(args.gate) if args.gate else None
    print(f'commit gate: {args.gate} -> '
          f'{FEATURE_MSG(gate) if gate else "hand-set thresholds"}')

    twin = DriftPrior()
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records))

    results = []
    for idx in order:
        meta = records[int(idx)]
        search = np.array(Image.open(meta['search_path']).convert('L'), dtype=np.uint8)
        ref = np.array(Image.open(meta['ref_path']).convert('L'), dtype=np.uint8)

        pred_radius, _ = twin.predict(meta['elapsed_time_s'])
        true_mag = true_drift_mag(meta)
        twin_error = abs(pred_radius - true_mag)

        r = match(search, ref, meta['elapsed_time_s'], twin, model, gate=gate)
        err = float(np.hypot(r['x'] - meta['gt_x'], r['y'] - meta['gt_y']))
        # error if a flagged (abstained) site were routed to the minimum-
        # expected-error annulus centroid instead of to the argmax guess
        fx = r.get('fallback_x', NOMINAL_CENTER[0]) if r.get('abstain') else r['x']
        fy = r.get('fallback_y', NOMINAL_CENTER[1]) if r.get('abstain') else r['y']
        err_risk = float(np.hypot(fx - meta['gt_x'], fy - meta['gt_y']))
        case = attribute(twin_error, err)

        twin.update(meta['elapsed_time_s'], true_mag)

        lm_in_fov = meta.get('landmark_in_fov', meta.get('landmark') is not None)

        results.append(dict(pair_id=meta['pair_id'], style=meta['style'],
                             difficulty=meta['difficulty'], landmark=meta['landmark'],
                             landmark_in_fov=lm_in_fov,
                             gt_x=meta['gt_x'], gt_y=meta['gt_y'],
                             pred_x=r['x'], pred_y=r['y'], error_px=err,
                             confidence=r['confidence'], twin_pred_radius=pred_radius,
                             twin_true_mag=true_mag, twin_error=twin_error,
                             n_near_peaks=r.get('n_near_peaks', 0),
                             is_periodic_degenerate=r.get('is_periodic_degenerate', False),
                             error_px_risk_managed=err_risk,
                             abstain=bool(r.get('abstain', False)),
                             abstain_reason=r.get('abstain_reason', ''),
                             landmark_z=float(r.get('landmark_z', 0.0)),
                             belief=float(r.get('belief', 0.0)),
                             entropy=float(r.get('entropy', 0.0)),
                             scale_est=r.get('scale_est'), scale_ok=r.get('scale_ok'),
                             attribution=case, success=err <= SUCCESS_PX,
                             true_scale_factor=meta.get('true_scale_factor'),
                             n_twin_updates_so_far=twin.n_updates - 1))

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)

    # ---- report ----
    errs = np.array([r['error_px'] for r in results])
    succ = np.array([r['success'] for r in results])
    print(f'\nOverall: n={len(results)}  success_rate={succ.mean()*100:.1f}%  '
          f'median_err={np.median(errs):.1f}px  mean_err={errs.mean():.1f}px')

    # ---- Information-Theoretic Solvability Audit ----
    solvable = [r for r in results if r['landmark_in_fov']]
    unsolvable = [r for r in results if not r['landmark_in_fov']]

    print('\n=============================================================')
    print('  INFORMATION-THEORETIC SOLVABILITY AUDIT')
    print('=============================================================')
    if solvable:
        e_solv = np.array([r['error_px'] for r in solvable])
        s_solv = np.array([r['success'] for r in solvable])
        print(f' Solvable (Landmark in FOV):     n={len(solvable):2d} ({len(solvable)/len(results)*100:.1f}%)  '
              f'success_rate={s_solv.mean()*100:5.1f}%  median_err={np.median(e_solv):.1f}px')
    else:
        print(' Solvable (Landmark in FOV):     n=0')

    if unsolvable:
        e_unsolv = np.array([r['error_px'] for r in unsolvable])
        s_unsolv = np.array([r['success'] for r in unsolvable])
        print(f' Unsolvable (Pure Periodic Grid): n={len(unsolvable):2d} ({len(unsolvable)/len(results)*100:.1f}%)  '
              f'success_rate={s_unsolv.mean()*100:5.1f}%  median_err={np.median(e_unsolv):.1f}px')
    else:
        print(' Unsolvable (Pure Periodic Grid): n=0')

    solv_rate = len(solvable) / len(results) * 100.0
    alg_solv_rate = (np.array([r['success'] for r in solvable]).mean() * 100.0) if solvable else 0.0
    print(f'\n Summary: Information-theoretic ceiling = {solv_rate:.1f}% solvable trials.')
    print(f' Algorithm solved {alg_solv_rate:.1f}% of the mathematically solvable trials.')
    print('=============================================================\n')

    for key in ['style', 'difficulty']:
        print(f'By {key}:')
        for val in sorted(set(r[key] for r in results)):
            sub = [r for r in results if r[key] == val]
            e = np.array([s['error_px'] for s in sub])
            s = np.array([s['success'] for s in sub])
            print(f'  {val:8s} n={len(sub):3d}  success={s.mean()*100:5.1f}%  '
                  f'median_err={np.median(e):7.1f}px  mean_err={e.mean():7.1f}px')

    # ---- Selective prediction: the system is allowed to say "I don't know" ----
    committed = [r for r in results if not r['abstain']]
    abstained = [r for r in results if r['abstain']]
    print('=============================================================')
    print('  SELECTIVE PREDICTION (belief-gated commitment)')
    print('=============================================================')
    cov = len(committed) / len(results) * 100.0
    if committed:
        acc = np.mean([r['success'] for r in committed]) * 100.0
        med = np.median([r['error_px'] for r in committed])
        print(f' Committed:  n={len(committed):3d}  coverage={cov:5.1f}%  '
              f'precision={acc:5.1f}%  median_err={med:.1f}px')
    if abstained:
        # how many abstentions were genuinely unidentifiable (no landmark in FOV)?
        justified = sum(1 for r in abstained if not r['landmark_in_fov'])
        print(f' Abstained:  n={len(abstained):3d}  of which genuinely unidentifiable '
              f'(no landmark in FOV) = {justified}/{len(abstained)} '
              f'({justified/len(abstained)*100:.0f}%)')
        from collections import Counter as _C
        for reason, k in _C(r['abstain_reason'] for r in abstained).most_common():
            print(f'     - {reason}: {k}')
    # abstention as an unidentifiability detector, scored against ground truth
    tp = sum(1 for r in results if r['abstain'] and not r['landmark_in_fov'])
    fp = sum(1 for r in results if r['abstain'] and r['landmark_in_fov'])
    fn = sum(1 for r in results if not r['abstain'] and not r['landmark_in_fov'])
    tn = sum(1 for r in results if not r['abstain'] and r['landmark_in_fov'])
    errs_risk = np.array([r['error_px_risk_managed'] for r in results])
    print(f'\n Median error, argmax-always:   {np.median(errs):.1f}px')
    print(f' Median error, risk-managed:    {np.median(errs_risk):.1f}px  '
          f'(flagged sites routed to annulus centroid)')

    print(f'\n Unidentifiability detector (abstain vs. landmark-not-in-FOV):')
    print(f'   recall={tp/max(tp+fn,1)*100:.1f}%  precision={tp/max(tp+fp,1)*100:.1f}%  '
          f'(tp={tp} fp={fp} fn={fn} tn={tn})')
    print('=============================================================\n')

    print('\nAttribution matrix (failure-mode audit):')
    from collections import Counter
    counts = Counter(r['attribution'] for r in results)
    for case in sorted(counts):
        print(f'  {case:55s} {counts[case]:3d}')

    # twin calibration trend: error over the course of the session
    print('\nTwin radius error, first 10 vs last 10 sites processed (should shrink):')
    first10 = np.mean([r['twin_error'] for r in results[:10]])
    last10 = np.mean([r['twin_error'] for r in results[-10:]])
    print(f'  first 10 mean twin error: {first10:.1f}px   last 10 mean twin error: {last10:.1f}px')


if __name__ == '__main__':
    main()
