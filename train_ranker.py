"""Learn the candidate ranking, instead of hand-weighting three channels.

localize.py currently decides with

    fused = 1.00*coarse + 0.45*fine_app + 0.50*clip(spread_z(fine_lm))/12

three constants picked by grid search. That form cannot express interactions
between channels, cannot trust different channels on different frames, and
represents "candidates compete" through a single spread-z. The measured
bottleneck is exactly here: the true site is PROPOSED in 87.5% of identifiable
trials but ranks first only ~46% of the time.

This trains a listwise ranker over the features in probes/rank_features.py:
each candidate gets a score, a softmax runs across the candidate set, and the
loss is cross-entropy against the true candidate's index. That optimises
"pick the right one out of this set", which is the metric, rather than a
per-candidate regression that ignores the set.

Deliberately small -- 30 inputs, one hidden layer -- because there are only
~120 training frames. Weights export to .npz so localize.py can apply the model
with numpy and stay torch-free.

    python train_ranker.py --train _sweep/feats_train.json \
                           --val   _sweep/feats_stress.json \
                           --out   ranker.npz
"""
import argparse, json
import numpy as np


def load(path, solvable_only=True):
    d = json.load(open(path))
    rows = []
    for r in d['rows']:
        y = np.array(r['y'])
        if y.sum() == 0:
            continue                  # true site not proposed: unrankable
        if solvable_only and not r['landmark_in_fov']:
            continue                  # unidentifiable by construction
        rows.append(dict(X=np.array(r['X'], dtype=np.float64),
                         y=y, baseline=np.array(r['baseline']),
                         landmark=r['landmark'], style=r['style'],
                         pair_id=r['pair_id']))
    return rows, d['features']


def standardise(rows, mu=None, sd=None):
    if mu is None:
        allX = np.concatenate([r['X'] for r in rows], axis=0)
        mu, sd = allX.mean(0), allX.std(0)
        sd[sd < 1e-8] = 1.0
    for r in rows:
        r['Xn'] = (r['X'] - mu) / sd
    return mu, sd


def rank1(rows, score_fn):
    hits = 0
    for r in rows:
        s = score_fn(r)
        if r['y'][int(np.argmax(s))] == 1:
            hits += 1
    return hits / max(1, len(rows))


# ------------------------------------------------------------------ model ----
def init(n_feat, n_hidden, rng):
    """n_hidden=0 gives a purely linear scorer -- 31 parameters instead of 385.

    Worth having as a real option, not a degenerate case: with under a hundred
    training frames the hidden layer is what overfits, and a linear listwise
    model is still strictly more expressive than the hand-set fusion (it gets
    all 30 features and fits every coefficient, rather than 3 channels with
    constants chosen by grid search)."""
    if n_hidden == 0:
        return dict(W2=rng.normal(0, 0.01, (n_feat, 1)), b2=np.zeros(1))
    return dict(
        W1=rng.normal(0, np.sqrt(2.0 / n_feat), (n_feat, n_hidden)),
        b1=np.zeros(n_hidden),
        W2=rng.normal(0, np.sqrt(2.0 / n_hidden), (n_hidden, 1)),
        b2=np.zeros(1),
    )


def forward(P, X):
    if 'W1' not in P:
        return (X @ P['W2'] + P['b2']).ravel(), X, None
    h_pre = X @ P['W1'] + P['b1']
    h = np.tanh(h_pre)
    return (h @ P['W2'] + P['b2']).ravel(), h, h_pre


def listwise_loss_grad(P, rows, l2):
    """Cross-entropy of the softmax over each candidate set against the true one.

    Where several candidates fall within tolerance the target is spread evenly
    over them -- they are all correct answers, and forcing a choice between two
    equally-correct candidates is noise, not signal."""
    g = {k: np.zeros_like(v) for k, v in P.items()}
    total = 0.0
    for r in rows:
        X = r['Xn']
        s, h, _ = forward(P, X)
        s = s - s.max()
        e = np.exp(s)
        p = e / e.sum()
        t = r['y'] / r['y'].sum()
        total += -np.sum(t * np.log(p + 1e-12))
        ds = (p - t)                                    # dL/ds
        g['W2'] += h.T @ ds[:, None]
        g['b2'] += ds.sum()
        if 'W1' in P:
            dh = np.outer(ds, P['W2'].ravel()) * (1 - h ** 2)
            g['W1'] += X.T @ dh
            g['b1'] += dh.sum(0)
    n = max(1, len(rows))
    for k in g:
        g[k] = g[k] / n + l2 * P[k]
    return total / n, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', default='ranker.npz')
    ap.add_argument('--hidden', type=int, default=12)
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--l2', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    tr, names = load(args.train)
    va, _ = load(args.val)
    mu, sd = standardise(tr)
    standardise(va, mu, sd)
    print(f'train frames {len(tr)}   val frames {len(va)}   features {len(names)}')

    base_tr = rank1(tr, lambda r: r['baseline'])
    base_va = rank1(va, lambda r: r['baseline'])
    print(f'hand-set fusion baseline:  train {100*base_tr:.1f}%   val {100*base_va:.1f}%\n')

    rng = np.random.default_rng(args.seed)
    P = init(len(names), args.hidden, rng)
    mom = {k: np.zeros_like(v) for k, v in P.items()}
    best = (-1.0, None, 0)

    for ep in range(1, args.epochs + 1):
        loss, g = listwise_loss_grad(P, tr, args.l2)
        for k in P:
            mom[k] = 0.9 * mom[k] - args.lr * g[k]
            P[k] = P[k] + mom[k]
        if ep % 20 == 0 or ep == 1:
            a_tr = rank1(tr, lambda r: forward(P, r['Xn'])[0])
            a_va = rank1(va, lambda r: forward(P, r['Xn'])[0])
            if a_va > best[0]:
                best = (a_va, {k: v.copy() for k, v in P.items()}, ep)
            print(f'  epoch {ep:4d}  loss {loss:.4f}  '
                  f'rank-1 train {100*a_tr:5.1f}%  val {100*a_va:5.1f}%')

    a_va, P_best, ep_best = best
    print(f'\nbest val rank-1 {100*a_va:.1f}% at epoch {ep_best} '
          f'(baseline {100*base_va:.1f}%)')
    # Early stopping picks the epoch on the validation split, so the val number
    # is optimistic by construction. The held-out report split decides.
    np.savez(args.out, mu=mu, sd=sd, **P_best)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
