"""Sweep decision rules over the cached candidate pools."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localize import _spread_z as spread_z   # import, never reimplement

CACHE = r'C:\Users\yoges\AppData\Local\Temp\claude\D--Downloads-projects-semicon\e851f7c8-fdb4-4c14-8b50-23b2f7a17ac2\scratchpad\cands.json'
W_APP, W_FA, W_LM = 1.0, 0.45, 0.50
LM_Z, LM_NCC = 4.0, 0.10


def decide(row, mode, tie_eps, override_rank=3):
    C = row['cands']
    if not C:
        return row['cx'], row['cy']
    ap = np.array([c['score'] for c in C]); fa = np.array([c['fa'] for c in C])
    fl = np.array([c['fl'] for c in C]); z = spread_z(fl)
    fu = W_APP * ap + W_FA * fa + W_LM * np.clip(z, 0, 12) / 12
    order = np.argsort(-fu); ranked = [C[i] for i in order]; fs = fu[order]
    lm_ok = bool(len(z) and z.max() >= LM_Z and fl.max() >= LM_NCC)
    tied = [r for r, f in zip(ranked, fs) if fs[0] - f < tie_eps]

    if mode == 'shipped':
        w = C[int(np.argmax(z))] if lm_ok else (
            min(tied, key=lambda k: np.hypot(k['x'] - row['cx'], k['y'] - row['cy']))
            if len(tied) > 1 else ranked[0])
    elif mode == 'rank1':
        w = ranked[0]
    elif mode == 'tie_only':          # keep the mandated tie-break, drop lm override
        w = (min(tied, key=lambda k: np.hypot(k['x'] - row['cx'], k['y'] - row['cy']))
             if len(tied) > 1 else ranked[0])
    elif mode == 'lm_gated':
        # landmark may only override if it ALSO ranks top-N on the fusion --
        # i.e. it refines the choice rather than contradicting it. rank=None
        # ("off") disables the override entirely -> pure rank1.
        i = int(np.argmax(z))
        gated = lm_ok and override_rank is not None and i in list(order[:override_rank])
        w = C[i] if gated else ranked[0]
    elif mode == 'lm_gated_tie':
        i = int(np.argmax(z))
        gated = lm_ok and override_rank is not None and i in list(order[:override_rank])
        if gated:
            w = C[i]
        elif len(tied) > 1:
            w = min(tied, key=lambda k: np.hypot(k['x'] - row['cx'], k['y'] - row['cy']))
        else:
            w = ranked[0]
    else:
        raise ValueError(mode)
    return w['rx'], w['ry']


def main():
    cache = json.load(open(CACHE))
    variants = sys.argv[1:] or list(cache)
    print(f'{"front end":8s} {"decision":14s} {"rank":>5} {"tie_eps":>8}  '
          f'{"all":>6} {"solvable":>9} {"median":>8}')
    ranks = (1, 3, 5, None)
    eps_list = (0.02, 0.005, 0.002, 0.0)
    best = None
    for v in variants:
        rows = cache[v]
        combos = [('shipped', 0.02, 3), ('rank1', 0.02, None), ('tie_only', 0.02, None)]
        for eps in eps_list:
            if eps != 0.02:
                combos.append(('tie_only', eps, None))
        for r in ranks:
            for mode in ('lm_gated', 'lm_gated_tie'):
                eps_opts = eps_list if mode == 'lm_gated_tie' else (0.02,)
                for eps in eps_opts:
                    combos.append((mode, eps, r))
        for mode, eps, r in combos:
            err, hard = [], []
            for row in rows:
                x, y = decide(row, mode, eps, override_rank=r)
                err.append(np.hypot(x - row['gt_x'], y - row['gt_y']))
                hard.append(row['difficulty'] == 'hard')
            e = np.array(err); h = np.array(hard)
            solvable = 100 * np.mean(e[~h] <= 15)
            med = np.median(e[~h])
            rlabel = 'off' if r is None else str(r)
            print(f'{v:8s} {mode:14s} {rlabel:>5} {eps:8.3f}  {100*np.mean(e<=15):5.1f}% '
                  f'{solvable:8.1f}% {med:7.2f}px')
            if best is None or solvable > best[0] or (solvable == best[0] and med < best[1]):
                best = (solvable, med, v, mode, eps, rlabel)
        print()
    print(f'BEST: {best[2]} {best[3]} rank={best[5]} eps={best[4]} -> '
          f'{best[0]:.1f}% solvable, {best[1]:.2f}px median')


if __name__ == '__main__':
    main()
