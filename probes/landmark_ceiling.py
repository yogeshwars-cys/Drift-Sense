"""Is the ranker's via-defect / gate-crossing tradeoff a DATA limit or a
REPRESENTATIONAL one?

Measured: a global learned ranker consistently lifts via_defect (16.7% -> ~42-50%)
and consistently costs gate_crossing (33.3% -> ~17-28%), across three capacities
and a 3x change in training-set size. The hypothesis is that one global channel
weighting cannot serve both -- point defects want the residual channel, line
crossings want appearance -- and nothing in the current 30 features tells the
model which situation it is in.

This tests that directly and cheaply, by cheating in a way that is diagnostic
rather than shippable: train a SEPARATE model per landmark type, using the true
type as an oracle. At inference the type is unknown, so these numbers are not
achievable -- they are the CEILING a conditional model could reach.

  if per-type models recover both categories  -> representational limit, and
      adding a feature that distinguishes point-like from line-like landmarks
      is the fix worth building.
  if they do not                              -> the information is not in
      these features at all, and shape descriptors will not rescue it either.

    python probes/landmark_ceiling.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from train_ranker import load, standardise, init, forward, listwise_loss_grad, rank1

TRAIN = '_sweep/feats_train2.json'
VAL = '_sweep/feats_stress.json'
REPORT = '_sweep/feats_primary.json'
TYPES = ('array_corner', 'gate_crossing', 'via_defect')


def fit(tr, n_feat, hidden=8, l2=1e-2, epochs=300, lr=0.05, seed=0, val=None):
    rng = np.random.default_rng(seed)
    P = init(n_feat, hidden, rng)
    mom = {k: np.zeros_like(v) for k, v in P.items()}
    best = (-1.0, {k: v.copy() for k, v in P.items()})
    for ep in range(1, epochs + 1):
        _, g = listwise_loss_grad(P, tr, l2)
        for k in P:
            mom[k] = 0.9 * mom[k] - lr * g[k]
            P[k] = P[k] + mom[k]
        if val and ep % 20 == 0:
            a = rank1(val, lambda r: forward(P, r['Xn'])[0])
            if a > best[0]:
                best = (a, {k: v.copy() for k, v in P.items()})
    return best[1] if val else P


def sub(rows, lm):
    return [r for r in rows if r['landmark'] == lm]


def main():
    tr, names = load(TRAIN)
    va, _ = load(VAL)
    rep, _ = load(REPORT)
    mu, sd = standardise(tr)
    standardise(va, mu, sd)
    standardise(rep, mu, sd)
    n_feat = len(names)

    print(f'train {len(tr)}  val {len(va)}  report {len(rep)}\n')

    # --- global model, for reference
    g_model = fit(tr, n_feat, val=va)
    print(f'  {"landmark":15s} {"n":>4s} {"fusion":>9s} {"global":>9s} '
          f'{"per-type":>9s}   <- oracle ceiling')

    tot_f = tot_g = tot_p = tot_n = 0
    for lm in TYPES:
        r_lm = sub(rep, lm)
        if not r_lm:
            continue
        t_lm = sub(tr, lm)
        v_lm = sub(va, lm)
        # per-type model; fall back to the global one if a type is too thin
        p_model = fit(t_lm, n_feat, hidden=6, l2=2e-2,
                      val=v_lm if len(v_lm) >= 5 else None) if len(t_lm) >= 20 else g_model

        a_f = rank1(r_lm, lambda r: r['baseline'])
        a_g = rank1(r_lm, lambda r: forward(g_model, r['Xn'])[0])
        a_p = rank1(r_lm, lambda r: forward(p_model, r['Xn'])[0])
        n = len(r_lm)
        tot_f += a_f * n; tot_g += a_g * n; tot_p += a_p * n; tot_n += n
        print(f'  {lm:15s} {n:4d} {100*a_f:8.1f}% {100*a_g:8.1f}% {100*a_p:8.1f}%'
              f'   (train n={len(t_lm)})')

    print(f'\n  {"TOTAL":15s} {tot_n:4d} {100*tot_f/tot_n:8.1f}% '
          f'{100*tot_g/tot_n:8.1f}% {100*tot_p/tot_n:8.1f}%')
    print(f'\n  per-type is an ORACLE -- it is told the landmark type, which the')
    print(f'  pipeline does not know. It bounds what a conditional model could do.')
    gain = (tot_p - tot_g) / tot_n
    if gain > 0.04:
        print(f'\n  VERDICT: conditioning is worth {100*gain:.1f} points over one global')
        print(f'           model. The limit is representational -- build a feature')
        print(f'           that distinguishes point-like from line-like landmarks.')
    else:
        print(f'\n  VERDICT: conditioning buys only {100*gain:.1f} points. The information')
        print(f'           is not in these features; shape descriptors will not fix it.')


if __name__ == '__main__':
    main()
