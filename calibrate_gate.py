"""
Fit the commit gate and report what it buys, held out.

    python calibrate_gate.py --results baseline_results.json --out commit_gate.json

The comparison is against the shipped hand-set gate (landmark_z >= 4.0 and
belief >= 0.20) evaluated on the same trials, so the two numbers are directly
comparable. The fitted gate's numbers are K-fold held out; the hand-set gate's
are in-sample and therefore, if anything, flattered.
"""
import argparse, json
import numpy as np
import commit_gate as G


def hand_set(records, lz_thr=4.0, belief_thr=0.20):
    """The gate being replaced. `belief` exists only in the matcher.py path;
    on benchmark rows from localize.py it is absent, so the comparison falls
    back to the landmark threshold alone rather than silently committing
    nothing and flattering the calibrated gate by comparison."""
    m = np.array([(r.get('landmark_z', 0.0) >= lz_thr)
                  and (r.get('belief', 1.0) >= belief_thr)
                  for r in records])
    y = np.array([bool(r['success']) for r in records])
    return m, y


def line(name, n, ncom, ncorrect, prec):
    print(f'  {name:34s} {ncom:4d}/{n:<4d} {ncom/n*100:6.1f}%   '
          f'{prec*100:6.1f}%   {ncorrect/n*100:6.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='primary_results.json',
                    help='output of `benchmark.py --out ...`, or of evaluate.py')
    ap.add_argument('--out', default='commit_gate.json')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--precision-floor', type=float, default=0.90)
    args = ap.parse_args()

    R = json.load(open(args.results))
    n = len(R)

    print(f'\ncalibration set: {args.results}  n={n}')
    print(f'precision floor: {args.precision_floor*100:.0f}%   folds: {args.folds}\n')
    print(f'  {"gate":34s} {"committed":>9s}  {"coverage":>7s}  {"precision":>8s}  '
          f'{"correct/100":>10s}')
    print('  ' + '-' * 76)

    m, y = hand_set(R)
    line('hand-set (lz>=4.0, belief>=0.20)', n, int(m.sum()),
         int(y[m].sum()), float(y[m].mean()) if m.sum() else 0.0)

    cv = G.fit_cv(R, k=args.folds, precision_floor=args.precision_floor)
    line(f'calibrated gate ({args.folds}-fold held out)', n, cv['n_committed'],
         int(round(cv['correct_per_100'] * n / 100.0)), cv['precision'])

    gate = G.fit(R, precision_floor=args.precision_floor)
    with open(args.out, 'w') as f:
        json.dump(gate, f, indent=2)

    print(f'\n  shipped gate: {gate["feature"]} {gate["sense"]} '
          f'{gate["threshold"]:.3g}')
    print(f'  candidates considered: {", ".join(gate["candidates"])}')
    picks = ', '.join(f'{p["feature"]}{p["sense"]}{p["threshold"]:.3g}'
                      for p in cv['fold_picks'])
    print(f'  per-fold picks: {picks}')
    print('    (a stable feature AND threshold across folds means the rule is '
          'not fold-specific; if the folds disagree on the FEATURE, the '
          'selection is noise and the gate should not be trusted)')

    base_correct = float(y[m].sum())
    new_correct = cv['correct_per_100'] * n / 100.0
    if base_correct > 0:
        print(f'\n  usable committed answers per 100 sites: '
              f'{base_correct/n*100:.0f} -> {new_correct/n*100:.0f}  '
              f'({(new_correct/base_correct - 1)*100:+.0f}%)')
    print(f'  written: {args.out}\n')


if __name__ == '__main__':
    main()
