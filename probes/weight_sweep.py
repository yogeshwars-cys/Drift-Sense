"""Sweep fusion weights offline against cached per-candidate channel scores.

Re-running the whole pipeline per weight combination costs minutes each; the
channels do not change when the weights do, so they are computed once by
rank_probe.py and the fusion is replayed here in milliseconds.

    python probes/rank_probe.py dataset_primary 60     # writes channel_scores.json
    python probes/weight_sweep.py

Reports rank-1 rate of the TRUE site, which is the quantity end-to-end accuracy
is bounded by. Only trials where the true site was proposed are scored -- the
weights cannot fix a candidate that was never generated.
"""
import json
import os
import sys
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, 'channel_scores.json')))


def rank1(w_coarse, w_fine, w_lm, w_prior, w_phase, rows=D):
    hits = 0
    for r in rows:
        f = (w_coarse * np.array(r['coarse'])
             + w_fine * np.array(r['fine_app'])
             + w_lm * np.clip(np.array(r['lm_z']), 0, 12) / 12.0
             + w_prior * (-np.array(r['dist']) / r['diag'])
             + w_phase * (-np.array(r['phase_shift']) / max(r['pitch'], 1.0)))
        i = r['true_i']
        hits += int((f > f[i]).sum()) == 0
    return hits / max(len(rows), 1)


def main():
    print(f'trials with the true site proposed: n={len(D)}')
    sol = [r for r in D if r['solvable']]

    print('\nsingle channels alone:')
    for name, kw in (('coarse', (1, 0, 0, 0, 0)),
                     ('fine_app', (0, 1, 0, 0, 0)),
                     ('fine_lm', (0, 0, 1, 0, 0))):
        print(f'  {name:9s} rank1={rank1(*kw)*100:5.1f}%   '
              f'solvable={rank1(*kw, rows=sol)*100:5.1f}%')

    grid = dict(w_coarse=(0.0, 0.5, 1.0), w_fine=(0.0, 0.3, 0.6, 1.0),
                w_lm=(0.0, 0.5, 1.0, 1.5), w_prior=(0.0, 0.1, 0.3),
                w_phase=(0.0, 0.15, 0.3))
    keys = list(grid)
    best = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        if all(c == 0 for c in combo[:3]):
            continue
        best.append((rank1(*combo), combo))
    best.sort(key=lambda t: -t[0])

    print(f'\ntop weight combinations ({len(best)} evaluated):')
    print(f'  {"rank1":>6s}  {"solvable":>8s}   ' + '  '.join(f'{k:>8s}' for k in keys))
    for score, combo in best[:10]:
        s_sol = rank1(*combo, rows=sol)
        print(f'  {score*100:5.1f}%  {s_sol*100:7.1f}%   '
              + '  '.join(f'{v:8.2f}' for v in combo))

    cur = (0.4, 1.0, 0.55, 0.10, 0.30)
    print(f'\nshipped weights {cur}: rank1={rank1(*cur)*100:.1f}%  '
          f'solvable={rank1(*cur, rows=sol)*100:.1f}%')


if __name__ == '__main__':
    main()
