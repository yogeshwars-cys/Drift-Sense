"""Sweep ROTATION_MIN_GAIN (plus rotation off) end-to-end on dataset_primary.

Full pipeline runs, not a cache sweep -- rotation happens in the front end
before candidates are pooled, so there is nothing to cache it against.
"""
import json, sys
import numpy as np
import cv2

sys.path.insert(0, r'D:\Downloads\projects\semicon')
import localize as L
from benchmark import run as bench_run

GT = r'D:\Downloads\projects\semicon\dataset_primary\ground_truth.json'


def score(records, **kw):
    rows = bench_run(records, **kw)
    err = np.array([r['error_px'] for r in rows])
    hard = np.array([r['difficulty'] == 'hard' for r in rows])
    rt = np.array([r['runtime_s'] for r in rows])
    return (100 * np.mean(err <= 15), 100 * np.mean(err[~hard] <= 15),
            np.median(err[~hard]), np.median(rt))


def main():
    records = json.load(open(GT))
    print(f'{"arm":16s} {"all":>6} {"solvable":>9} {"median":>8} {"rt/pair":>8}')
    orig = L.ROTATION_MIN_GAIN
    try:
        a, s, m, rt = score(records, use_rotation=False)
        print(f'{"rotation off":16s} {a:5.1f}% {s:8.1f}% {m:7.2f}px {rt:7.2f}s')
        for g in (0.01, 0.02, 0.05):
            L.ROTATION_MIN_GAIN = g
            a, s, m, rt = score(records, use_rotation=True)
            print(f'{"gain="+str(g):16s} {a:5.1f}% {s:8.1f}% {m:7.2f}px {rt:7.2f}s')
    finally:
        L.ROTATION_MIN_GAIN = orig


if __name__ == '__main__':
    main()
