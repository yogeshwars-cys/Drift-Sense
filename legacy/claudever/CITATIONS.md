# Citations & Justification of Design Choices

Every augmentation, noise model, and structural parameter used in
`dataset_generator.py` and `localize.py` is justified below with a public
reference.

## 1. Sensor noise model (independent Poisson + Gaussian per image)

**Choice:** Each image (reference and search) gets its own independent
shot-noise (Poisson) + read-noise (Gaussian) realization, with the search
image noisier than the reference (lower photon budget at lower
magnification).

**Justification:** The mixed Poisson–Gaussian model is the standard,
empirically validated noise model for raw sensor data (photon shot noise
dominates the signal-dependent part; residual electronics/read noise is
well approximated as additive Gaussian).

- Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008).
  "Practical Poissonian-Gaussian noise modeling and fitting for
  single-image raw-data." *IEEE Transactions on Image Processing*,
  17(10), 1737-1754. https://doi.org/10.1109/TIP.2008.2001399

## 2. Edge brightening

**Choice:** Pixel intensity is boosted proportionally to local gradient
magnitude, mimicking brighter contrast along feature edges.

**Justification:** This is a well-documented physical phenomenon in
secondary-electron (SE) SEM imaging — the "edge effect." A larger fraction
of the electron-beam interaction volume near an edge/topographic step is
close enough to the surface for secondary electrons to escape and be
detected, producing excess brightness at edges relative to flat regions.

- Goldstein, J. I., et al. *Scanning Electron Microscopy and X-Ray
  Microanalysis*. Springer/Plenum Press. (Standard SEM reference
  describing the edge-brightening effect on SE contrast.)
- JEOL Ltd., "Edge effect" — SEM glossary entry:
  https://www.jeol.com/words/semterms/20121024.012800.php
- ETH Zürich Electron Microscopy, "Secondary electron imaging":
  https://www.microscopy.ethz.ch/se.htm

## 3. Blur (beam spot / defocus) and rotation/scale jitter

**Choice:** Gaussian blur is applied to both images (more on the search
image); a small relative rotation (±2°) and magnification jitter (±10%
around nominal 10x) are applied to the search image only.

**Justification:** Finite electron-beam spot size and defocus produce a
blur kernel well approximated by a Gaussian point-spread function in SEM
imaging (same SEM references as above, §2). Rotation and scale jitter
model realistic stage/lens calibration drift between two separate
tool visits — exactly the "navigation error" the challenge describes,
which is why the localization algorithm must search a band of scales and
angles rather than assuming a fixed 10x/0° relationship.

## 4. DRAM word-line/bit-line grid structure

**Choice:** Periodic horizontal/vertical line grid with a via dot at each
intersection, at a pitch large enough (in "reference-resolution" pixels)
to survive 10x downsampling without aliasing away.

**Justification:** Real DRAM arrays are built from a repeating word-line /
bit-line grid with a storage-node contact/via at each active cell; imec's
public technology overview describes current DRAM generations and their
half-pitch scaling.

- imec, "DRAM peripheral transistors technology platform":
  https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors

## 5. FinFET fin/gate structure

**Choice:** Dense parallel vertical fins crossed by 1-2 horizontal gate
bars.

**Justification:** This mirrors the standard FinFET layout — dense
parallel fins cut by gate lines running perpendicular to them. Published
metrology target dimensions (e.g., 22nm-node fin pitch ≈ 44nm, gate pitch
≈ 88nm, per Bunday et al., cited secondhand below) motivate keeping fin
pitch roughly half of gate pitch in the synthetic generator.

- Bunday, B. D., et al., dimensions as tabulated and cited in: "hp-finite
  element method for simulating light scattering from complex 3D
  structures" (arXiv:1503.06617), Section 4 / Table 10:
  https://arxiv.org/pdf/1503.06617

## 6. Localization algorithm: normalized cross-correlation

**Choice:** Multi-scale, multi-rotation normalized cross-correlation
(NCC) as the core matching score, computed via FFT for speed.

**Justification:** NCC is the standard, well-validated similarity measure
for template matching / image registration, robust to affine brightness
differences between the two captures (important here since reference and
search are independent sensor captures with different noise/contrast).

- Lewis, J. P. (1995). "Fast Normalized Cross-Correlation." *Vision
  Interface*, pp. 120-123.
