"""
Trains the Stage-2 re-ranker embedding self-supervised, straight from the
synthetic dataset's own ground truth -- no extra labeling step.

Triplets:
  anchor   = reference image
  positive = true crop at (gt_x, gt_y) in the search image
  negative = (a) a phase-aligned periodic distractor (true site shifted by an
                 integer number of pitches -- the confusable case NCC alone
                 cannot resolve), and
             (b) a random crop elsewhere in the array

This directly targets the "high periodicity" failure mode: the embedding is
explicitly taught that same-phase crops are the hard negatives, not just any
random patch.
"""
import argparse, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from reranker_model import EmbedNet, batch_to_tensor

FOOT_DEFAULT = 100


def load_pair(meta):
    search = np.array(Image.open(meta['search_path']).convert('L'), dtype=np.float32)
    ref = np.array(Image.open(meta['ref_path']).convert('L'), dtype=np.float32)
    return search, ref


def crop(search, cx, cy, foot):
    h = foot / 2
    x0, x1 = int(round(cx - h)), int(round(cx - h)) + int(round(foot))
    y0, y1 = int(round(cy - h)), int(round(cy - h)) + int(round(foot))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(search.shape[1], x1), min(search.shape[0], y1)
    patch = search[y0:y1, x0:x1]
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    return patch


def make_negative(search, gt_x, gt_y, pitch, foot, rng, phase_aligned=True):
    for _ in range(20):
        if phase_aligned:
            axis = rng.choice(['x', 'y'])
            k_min = max(2, int(np.ceil(foot * 0.65 / pitch)))
            k = int(rng.integers(k_min, k_min + 4)) * int(rng.choice([-1, 1]))
            if axis == 'x':
                nx, ny = gt_x + k * pitch, gt_y
            else:
                nx, ny = gt_x, gt_y + k * pitch
        else:
            nx = rng.uniform(foot, search.shape[1] - foot)
            ny = rng.uniform(foot, search.shape[0] - foot)
        if foot < nx < search.shape[1] - foot and foot < ny < search.shape[0] - foot:
            if np.hypot(nx - gt_x, ny - gt_y) > foot * 0.6:
                patch = crop(search, nx, ny, foot)
                if patch is not None:
                    return patch
    return None


def build_triplets(records, rng):
    anchors, positives, negatives, weights = [], [], [], []
    for meta in records:
        search, ref = load_pair(meta)
        foot = meta.get('true_footprint_px', FOOT_DEFAULT)
        pos = crop(search, meta['gt_x'], meta['gt_y'], foot)
        if pos is None:
            continue
        has_lm = meta.get('landmark_in_fov', meta.get('landmark') is not None)
        w = 2.0 if has_lm else 0.5
        for phase_aligned in (True, False):
            neg = make_negative(search, meta['gt_x'], meta['gt_y'], meta['pitch'],
                                 foot, rng, phase_aligned=phase_aligned)
            if neg is not None:
                anchors.append(ref)
                positives.append(pos)
                negatives.append(neg)
                weights.append(w)
    return anchors, positives, negatives, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='dataset')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='reranker.pt')
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    with open(f'{args.dataset}/ground_truth.json') as f:
        records = json.load(f)

    anchors, positives, negatives, weights = build_triplets(records, rng)
    print(f'built {len(anchors)} triplets from {len(records)} pairs')

    model = EmbedNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.TripletMarginLoss(margin=0.3, reduction='none')

    a_t = batch_to_tensor(anchors)
    p_t = batch_to_tensor(positives)
    n_t = batch_to_tensor(negatives)
    w_t = torch.tensor(weights, dtype=torch.float32)

    n = a_t.shape[0]
    batch_size = 8
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            za = model(a_t[idx])
            zp = model(p_t[idx])
            zn = model(n_t[idx])
            loss_unweighted = loss_fn(za, zp, zn)
            loss = (loss_unweighted * w_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'epoch {epoch+1:3d}  loss {total_loss / n:.4f}')

    torch.save(model.state_dict(), args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
