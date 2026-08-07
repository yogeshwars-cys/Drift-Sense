"""Held-out report for the learned ranker.

Reports on a split used for NEITHER training nor early stopping, against the
hand-set linear fusion on exactly the same candidate sets, and breaks the
result down by landmark type -- because the measured gap is not uniform
(array corner 92.5%, gate crossing 42.1%, via defect 19.0%), and a ranker that
only helps where the classical fusion was already winning has not fixed
anything.

    python probes/ranker_report.py --model ranker.npz --report _sweep/feats_primary.json
"""
import argparse, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from train_ranker import load, forward


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ranker.npz')
    ap.add_argument('--report', required=True)
    args = ap.parse_args()

    M = np.load(args.model)
    P = {k: M[k] for k in ('W1', 'b1', 'W2', 'b2')}
    mu, sd = M['mu'], M['sd']

    rows, _ = load(args.report)
    for r in rows:
        r['Xn'] = (r['X'] - mu) / sd

    base_hit, model_hit = [], []
    by_lm = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        b = r['y'][int(np.argmax(r['baseline']))] == 1
        m = r['y'][int(np.argmax(forward(P, r['Xn'])[0]))] == 1
        base_hit.append(b)
        model_hit.append(m)
        k = r['landmark'] or 'none'
        by_lm[k][0] += 1
        by_lm[k][1] += int(b)
        by_lm[k][2] += int(m)

    base_hit = np.array(base_hit)
    model_hit = np.array(model_hit)
    n = len(rows)
    print(f'\nheld-out: {args.report}   model: {args.model}')
    print(f'frames where the true site was proposed (rankable): {n}\n')
    print(f'  hand-set fusion   rank-1  {100*base_hit.mean():5.1f}%  ({base_hit.sum()}/{n})')
    print(f'  learned ranker    rank-1  {100*model_hit.mean():5.1f}%  ({model_hit.sum()}/{n})')
    delta = int(model_hit.sum() - base_hit.sum())
    print(f'  delta                     {delta:+d} pairs')

    # McNemar: only the disagreements carry information about which is better.
    b_only = int((base_hit & ~model_hit).sum())
    m_only = int((~base_hit & model_hit).sum())
    print(f'\n  disagreements: baseline-only {b_only}, model-only {m_only}')
    tot = b_only + m_only
    if tot:
        # exact binomial two-sided p under H0: a disagreement is a coin flip
        from math import comb
        k = min(b_only, m_only)
        p = sum(comb(tot, i) for i in range(0, k + 1)) / (2 ** tot) * 2
        print(f'  McNemar exact two-sided p = {min(1.0, p):.4f}')
    else:
        print('  the two rankers never disagree.')

    print('\n  by landmark type (the gap this was supposed to close)')
    print(f'    {"landmark":15s} {"n":>4s} {"fusion":>8s} {"learned":>8s}')
    for k in sorted(by_lm, key=lambda x: -by_lm[x][0]):
        cnt, b, m = by_lm[k]
        print(f'    {k:15s} {cnt:4d} {100*b/cnt:7.1f}% {100*m/cnt:7.1f}%')

    lo, hi = wilson(int(model_hit.sum()), n)
    blo, bhi = wilson(int(base_hit.sum()), n)
    print(f'\n  95% CI  fusion [{100*blo:.1f}, {100*bhi:.1f}]   '
          f'learned [{100*lo:.1f}, {100*hi:.1f}]')
    if tot and min(1.0, p) < 0.05 and delta > 0:
        print('\n  VERDICT: significant improvement. Wire it into localize.py.')
    elif delta > 0:
        print('\n  VERDICT: positive but not significant at this n. Not enough to ship.')
    else:
        print('\n  VERDICT: no improvement. The hand-set fusion stands.')


if __name__ == '__main__':
    main()
