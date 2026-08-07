"""Reproduces every measured claim in induction.py's docstring.

    python probes/induction_probe.py --dataset dataset_primary \
                                     --results primary_results.json

Claim 1  the pitch estimator fails silently on ~13% of frames
Claim 2  induction separates those failures with no threshold tuning
Claim 3  the score predicts LOCALIZATION outcome, with inverted sign
Claim 4  it is partly independent of the commit gate's existing signals
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from induction import induction_evidence          # noqa: E402
from lattice import estimate_lattice              # noqa: E402
from localize import load_gray                    # noqa: E402

SEARCH_PITCH_BAND = (5.0, 40.0)
PITCH_TOL = 0.15      # |measured/true - 1| beyond this is a lock failure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset_primary')
    ap.add_argument('--results', default=None,
                    help='benchmark.py output, for claims 3 and 4')
    args = ap.parse_args()

    gt = json.load(open(f'{args.dataset}/ground_truth.json'))
    ind, wrong = [], []
    for g in gt:
        img = load_gray(g['search_path'])
        measured = estimate_lattice(img, SEARCH_PITCH_BAND)['pitch']
        ind.append(induction_evidence(img, measured))
        wrong.append(abs(measured / g['pitch'] - 1.0) > PITCH_TOL
                     if measured > 0 else True)
    ind = np.array(ind, dtype=np.float64)
    wrong = np.array(wrong)
    n = len(gt)

    print(f'-- claim 1: silent pitch failures ({args.dataset}, n={n})')
    print(f'   pitch within {PITCH_TOL:.0%} of truth : {int((~wrong).sum())}/{n}')
    print(f'   locked onto a wrong lag      : {int(wrong.sum())}/{n}')

    print('\n-- claim 2: induction separates them')
    print(f'   correct pitch (n={int((~wrong).sum())}): median score '
          f'{np.nanmedian(ind[~wrong]):+.3f}')
    if wrong.any():
        print(f'   wrong pitch   (n={int(wrong.sum())}): median score '
              f'{np.nanmedian(ind[wrong]):+.3f}   max {np.nanmax(ind[wrong]):+.3f}')
    caught = int((wrong & (ind < 0)).sum())
    false_alarms = int(((~wrong) & (ind < 0)).sum())
    print(f'   at the sign boundary: {caught}/{int(wrong.sum())} caught, '
          f'{false_alarms} false alarms')

    if not args.results or not os.path.exists(args.results):
        print('\n(pass --results to reproduce claims 3 and 4)')
        return

    rows = json.load(open(args.results))
    if len(rows) != n:
        print(f'\nresults has {len(rows)} rows, dataset has {n} -- skipping 3/4')
        return
    ok = np.array([r['error_px'] <= 15.0 for r in rows])
    fails, passes = ind < 0, ind >= 0

    print('\n-- claim 3: it predicts LOCALIZATION, sign inverted')
    print(f'   induction  < 0 (n={int(fails.sum()):3d}): accuracy '
          f'{100*ok[fails].mean():.1f}%')
    print(f'   induction >= 0 (n={int(passes.sum()):3d}): accuracy '
          f'{100*ok[passes].mean():.1f}%')
    print(f'   base rate                : {100*ok.mean():.1f}%')
    try:
        from scipy.stats import pointbiserialr
        r, p = pointbiserialr(ok, np.nan_to_num(ind))
        print(f'   point-biserial r = {r:+.3f} (p={p:.2g})')
    except ImportError:
        pass

    print('\n-- claim 4: independent of the existing signals')
    npk = np.array([r.get('n_near_peaks', 10 ** 6) for r in rows])
    b = npk <= 5
    for name, sel in (('n_near_peaks<=5 (existing)', b),
                      ('induction<0      (this)  ', fails),
                      ('either                   ', b | fails),
                      ('both                     ', b & fails)):
        if sel.sum():
            print(f'   {name}  {int(sel.sum()):3d} sites  '
                  f'{100*ok[sel].mean():5.1f}%')
    print(f'   overlap: {int((b & fails).sum())} of '
          f'{int(fails.sum())} / {int(b.sum())}')


if __name__ == '__main__':
    main()
