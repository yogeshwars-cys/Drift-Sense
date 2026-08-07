"""Step NCC: compare the measured-scale bracket (foot_bracket) against a plain
blind sweep, end to end, keeping everything downstream (residual channel,
full-res rescoring, fusion, fixed decision layer) identical.

Two priors are tested per the plan: the generator's actual range (8.5-11.5x)
and a deliberately wide one (6-16x), to check whether a fixed sweep generalises
or is overfit to a prior about AM's generator.
"""
import json, sys
import numpy as np

sys.path.insert(0, r'D:\Downloads\projects\semicon')
import localize as L
from benchmark import run as bench_run

GT = r'D:\Downloads\projects\semicon\dataset_primary\ground_truth.json'


def make_ncc_bracket(lo, hi, n):
    def _bracket(ref_size, scale_est, span):
        return sorted({int(round(ref_size / s)) for s in np.linspace(lo, hi, n)})
    return _bracket


def score(records, **kw):
    rows = bench_run(records, **kw)
    err = np.array([r['error_px'] for r in rows])
    hard = np.array([r['difficulty'] == 'hard' for r in rows])
    rt = np.array([r['runtime_s'] for r in rows])
    return (100 * np.mean(err <= 15), 100 * np.mean(err[~hard] <= 15),
            np.median(err[~hard]), np.median(rt))


def main():
    records = json.load(open(GT))
    orig_bracket = L.foot_bracket
    print(f'{"arm":22s} {"all":>6} {"solvable":>9} {"median":>8} {"rt/pair":>8}')
    try:
        a, s, m, rt = score(records)
        print(f'{"measured (Step 7)":22s} {a:5.1f}% {s:8.1f}% {m:7.2f}px {rt:7.2f}s')

        L.foot_bracket = make_ncc_bracket(8.5, 11.5, 13)
        a, s, m, rt = score(records)
        print(f'{"ncc 8.5-11.5x/13":22s} {a:5.1f}% {s:8.1f}% {m:7.2f}px {rt:7.2f}s')

        L.foot_bracket = make_ncc_bracket(6.0, 16.0, 21)
        a, s, m, rt = score(records)
        print(f'{"ncc 6-16x/21 (wide)":22s} {a:5.1f}% {s:8.1f}% {m:7.2f}px {rt:7.2f}s')
    finally:
        L.foot_bracket = orig_bracket


if __name__ == '__main__':
    main()
