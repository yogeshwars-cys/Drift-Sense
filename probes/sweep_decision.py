"""Sweep decision rules over the cached candidate pools."""
import json, sys
import numpy as np

CACHE = r'C:\Users\yoges\AppData\Local\Temp\claude\D--Downloads-projects-semicon\e851f7c8-fdb4-4c14-8b50-23b2f7a17ac2\scratchpad\cands.json'
W_APP, W_FA, W_LM = 1.0, 0.45, 0.50
LM_Z, LM_NCC = 4.0, 0.10


def spread_z(v):
    v = np.asarray(v, float)
    if v.size < 2:
        return np.zeros_like(v)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    s = 1.4826 * mad
    if s < 1e-9:
        s = v.std() or 1e-9
    return (v - med) / s


def decide(row, mode, tie_eps):
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
        # landmark may only override if it ALSO ranks top-3 on the fusion --
        # i.e. it refines the choice rather than contradicting it
        i = int(np.argmax(z))
        w = C[i] if (lm_ok and i in list(order[:3])) else ranked[0]
    elif mode == 'lm_gated_tie':
        i = int(np.argmax(z))
        if lm_ok and i in list(order[:3]):
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
    print(f'{"front end":8s} {"decision":14s} {"tie_eps":>8}  {"all":>6} {"solvable":>9} {"median":>8}')
    for v in variants:
        rows = cache[v]
        for mode in ('shipped', 'rank1', 'tie_only', 'lm_gated', 'lm_gated_tie'):
            eps_list = (0.02,) if mode in ('shipped',) else (
                (0.02, 0.005, 0.002) if 'tie' in mode else (0.02,))
            for eps in eps_list:
                err, hard = [], []
                for r in rows:
                    x, y = decide(r, mode, eps)
                    err.append(np.hypot(x - r['gt_x'], y - r['gt_y']))
                    hard.append(r['difficulty'] == 'hard')
                e = np.array(err); h = np.array(hard)
                print(f'{v:8s} {mode:14s} {eps:8.3f}  {100*np.mean(e<=15):5.1f}% '
                      f'{100*np.mean(e[~h]<=15):8.1f}% {np.median(e[~h]):7.2f}px')
        print()


if __name__ == '__main__':
    main()
