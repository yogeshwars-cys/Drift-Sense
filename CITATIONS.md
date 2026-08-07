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

---

# Additions from the rewrite

The sections above cover the original augmentation set. The following were added
against `IMPLEMENTATION_PLAN.md` §5 and §3, and each is justified independently.

## 7. Supersampled rasterisation (area-averaged demagnification)

The search image is rendered at 4x and area-averaged down rather than drawn
directly at output resolution. Two justifications, one physical and one
methodological.

Physically, a demagnified image is formed by integrating the scene over each
detector element through the instrument's point-spread function. Feature edges
land on fractional pixels and partially fill them; they do not snap to integer
boundaries. Area-averaging a supersampled raster is the standard discrete
approximation of that integral.

Methodologically, drawing lines at integer positions quantises each line's
sub-pixel phase differently across the frame, stamping a per-cell aliasing
fingerprint into the image. A matcher can learn that fingerprint and use it to
tell lattice cells apart -- scoring well on this generator and failing on any
other. Removing the artefact is preferable to leaving a trap in the data.

- Glassner, A. S. (1995). *Principles of Digital Image Synthesis*, ch. on
  sampling and reconstruction (area sampling as the discrete form of the
  imaging integral).
- Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray Microanalysis*,
  ch. on image formation and the sampling of the scanned field.

## 8. Anisotropic, directional edge brightening

The secondary-electron yield depends on the angle between the local surface
normal and the detector, approximately as eta(theta) ~ sec(theta). A facet
tilted toward the detector is therefore brighter than the opposite facet of the
same feature, and the bright band decays over the SE escape depth rather than
occupying a single pixel. The previous isotropic gradient-magnitude boost
brightened both sides of every line equally, which is not what an SEM does and
which partially cancels under normalized cross-correlation.

- Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray Microanalysis*
  (SE yield vs. surface tilt; detector-geometry contrast).
- Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image Formation
  and Microanalysis*, 2nd ed., Springer (topographic contrast, escape depth).

## 9. Scan-system distortion: line jitter and slow drift

An SEM builds an image one scan line at a time over seconds. Beam-deflection
noise and mains pickup perturb each line's horizontal start independently
(shearing rows by sub-pixel amounts), while stage and beam drift add a smooth
shift that accumulates down the frame. Both are per-acquisition, which is
precisely why reference and search are *not* related by a pure rigid transform
-- the assumption a naive matcher makes.

- Reimer, L. *Scanning Electron Microscopy* (scan-system linearity, drift).
- Snella, M. T. (2010). *Drift Correction for Scanning-Electron Microscopy*,
  M.Eng. thesis, MIT.
- Goldstein, J. et al., ch. on image-distortion sources in raster scanning.

## 10. Beam-induced charging

Poorly-conducting regions accumulate potential under the beam, raising the local
SE yield and smearing it along the fast-scan direction. The result is the bright
horizontal streaking characteristic of insulating features. Modelled as a
directional exponential smear of the bright regions, modulated by a smooth
random field so that charging accumulates unevenly rather than uniformly.

- Cazaux, J. (2004). "Charging in scanning electron microscopy from inside and
  outside." *Scanning* 26(4), 181-203.
- Goldstein, J. et al., ch. on charging artefacts and their mitigation.

## 11. Field shading (detector collection efficiency)

Collection efficiency varies smoothly across the scanned field, producing a
low-order multiplicative intensity gradient. Harmless to a normalized
correlation computed over a whole patch, and specifically *not* harmless to any
statistic computed on raw intensity -- which is why it belongs in the generator
rather than being assumed away.

- Goldstein, J. et al. (detector collection-efficiency variation across the
  scanned field).

## 12. Astigmatic (anisotropic) defocus

A real electron column is never perfectly stigmated, so the probe is elliptical
and its axes are not aligned to the raster. Modelled by rotating into the PSF
frame, blurring with unequal sigmas, and rotating back.

- Reimer, L. *Scanning Electron Microscopy* (astigmatism and probe shape).
- Erasmus, S. J. & Smith, K. C. A. (1982). "An automatic focusing and
  astigmatism correction system for the SEM." *Journal of Microscopy* 127(2).

## 13. Global lattice phase coherence (the phase-lock constraint)

The matcher pins the reference's position modulo the lattice pitch by comparing
the absolute lattice phase of the two images. This is not an artefact of the
simulator: a die's array is printed from a single reticle by a step-and-scan
exposure tool, so the lattice is globally phase-coherent across the die, with
pattern-placement error in the nanometre range against a pitch of tens of
nanometres.

- Mack, C. (2007). *Fundamental Principles of Optical Lithography*, Wiley
  (overlay and pattern-placement error budgets).
- Levinson, H. J. (2010). *Principles of Lithography*, 3rd ed., SPIE Press
  (step-and-scan placement accuracy).

## 14. Fourier-domain scale and rotation estimation

The magnification is recovered as the ratio of the two lattice pitches and the
relative rotation from a multi-harmonic, sub-bin spectral angle. The general
technique -- registering images by their translation-invariant magnitude spectra
-- is standard.

- Reddy, B. S. & Chatterji, B. N. (1996). "An FFT-based technique for
  translation, rotation and scale-invariant image registration." *IEEE
  Transactions on Image Processing* 5(8), 1266-1271.
- Foroosh, H., Zerubia, J. & Berthod, M. (2002). "Extension of phase
  correlation to subpixel registration." *IEEE TIP* 11(3), 188-200.
