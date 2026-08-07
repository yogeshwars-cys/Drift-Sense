"""Validate the unused signal: CROSS-IMAGE LATTICE PHASE LOCK.

Both images depict one globally-coherent lattice. Measuring the complex Fourier
coefficient at the fundamental gives each image's absolute lattice phase, so the
true reference centre is pinned MODULO THE PITCH -- exactly, sub-pixel, from two
FFTs and no correlation at all.

This measures how tight that congruence actually is: the residual of the true
centre against the predicted phase, in search-image pixels.
"""
import sys, os
sys.path.insert(0, r'D:\Downloads\projects\semicon\drift_sense')
os.chdir(r'D:\Downloads\projects\semicon\drift_sense')
import numpy as np
import dataset_generator as G
from lattice import estimate_scale

MARGIN = G.MARGIN
_state = {'rng': None}
G.clamp_site = lambda dx, dy: (float(_state['rng'].uniform(MARGIN, G.SEARCH_SIZE - MARGIN)),
                               float(_state['rng'].uniform(MARGIN, G.SEARCH_SIZE - MARGIN)))


def fundamental_phase(img, pitch, axis):
    """Absolute phase (px) of the periodic line family along `axis`.

    Projects onto the axis, then reads arg of the single DFT coefficient at the
    known pitch. One coefficient, so noise averages over the whole projection.
    """
    prof = img.astype(np.float64).mean(axis=axis)      # axis=0 -> varies in x
    n = len(prof)
    k = n / pitch                                      # cycles across the profile
    t = np.arange(n)
    c = np.sum((prof - prof.mean()) * np.exp(-2j * np.pi * k * t / n))
    # peak of cos(2pi(t-phi)/pitch) -> phi = -arg(c) * pitch / 2pi
    return (-np.angle(c) * pitch / (2 * np.pi)) % pitch


N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
rng = np.random.default_rng(2024); _state['rng'] = rng
res_x, res_y, scale_err = [], [], []

for i in range(N):
    style = 'dram' if i % 2 == 0 else 'finfet'
    gen = G.generate_dram_pair if style == 'dram' else G.generate_finfet_pair
    search, ref, meta = gen(rng, i, 'easy' if i % 4 < 2 else 'hard')

    s_est, lat_s, lat_r, ok = estimate_scale(search, ref)
    p = lat_s['pitch']                       # search-space pitch, measured
    foot = ref.shape[0] / s_est
    scale_err.append(abs(s_est - meta['true_scale_factor']) / meta['true_scale_factor'] * 100)

    # search-image absolute phase, and the reference's phase expressed in
    # search-space px (ref pitch = p * s_est)
    ph_s_x = fundamental_phase(search, p, axis=0)
    ph_r_x = fundamental_phase(ref, p * s_est, axis=0) / s_est
    # origin of the ref crop in search coords is pinned mod p
    ox_pred = (ph_s_x - ph_r_x) % p
    ox_true = meta['gt_x'] - foot / 2.0
    dx = (ox_true - ox_pred + p / 2) % p - p / 2     # signed residual, |.| <= p/2
    res_x.append(dx)

    if style == 'dram':                       # DRAM has a second, orthogonal family
        ph_s_y = fundamental_phase(search, p, axis=1)
        ph_r_y = fundamental_phase(ref, p * s_est, axis=1) / s_est
        oy_pred = (ph_s_y - ph_r_y) % p
        oy_true = meta['gt_y'] - foot / 2.0
        res_y.append((oy_true - oy_pred + p / 2) % p - p / 2)

for name, r in (('x (both styles)', res_x), ('y (DRAM only)', res_y)):
    if not r:
        continue
    a = np.abs(np.array(r))
    print(f'{name:18s} n={len(a):3d}  median |residual| = {np.median(a):5.2f} px   '
          f'p90 = {np.percentile(a,90):5.2f} px   max = {a.max():5.2f} px')
print(f'scale estimate      median error = {np.median(scale_err):.2f}%  '
      f'(the sweep in localize.py spans +/-3.5%)')
print('\nInterpretation: a residual well under 1px means the true centre is pinned')
print('to a discrete lattice of cells with sub-pixel accuracy, from 2 FFTs -- no NCC.')
