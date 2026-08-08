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
- Healey, G. E., & Kondepudy, R. (1994). "Radiometric CCD camera
  calibration and noise estimation." *IEEE Transactions on Pattern
  Analysis and Machine Intelligence*, 16(3), 267-276.
  https://doi.org/10.1109/34.276126
- Janesick, J. R. (2001). *Scientific Charge-Coupled Devices*, SPIE
  Press, ch. 2-4 (shot noise, read noise, and the mixed
  Poisson-Gaussian sensor model underlying the photon-transfer
  method).

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
around nominal 10x) are applied to the reference image only
(`dataset_generator.py`, `rot_ref`).

**Justification:** Finite electron-beam spot size and defocus produce a
blur kernel well approximated by a Gaussian point-spread function in SEM
imaging (same SEM references as above, §2). Rotation and scale jitter
model realistic stage/lens calibration drift between two separate
tool visits — exactly the "navigation error" the challenge describes,
which is why the localization algorithm must search a band of scales and
angles rather than assuming a fixed 10x/0° relationship.

- Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image
  Formation and Microanalysis*, 2nd ed., Springer, ch. on probe
  formation and depth of focus (beam-spot size and defocus as a
  Gaussian-approximable blur).
- Mack, C. (2007). *Fundamental Principles of Optical Lithography*,
  Wiley (stage/tool overlay drift and calibration error between
  separate exposure or imaging passes, motivating the rotation/scale
  jitter between the two captures).

## 4. DRAM word-line/bit-line grid structure

**Choice:** Periodic horizontal/vertical line grid with a via dot at each
intersection, at a pitch large enough (in "reference-resolution" pixels)
to survive 10x downsampling without aliasing away.

**Justification:** Real DRAM arrays are built from a repeating word-line /
bit-line grid with a storage-node contact/via at each active cell. Current
generations are described as "10nm-class," with memory-array active-area
half-pitches of roughly 10-19nm; a line+space pattern's pitch is 2x the
half-pitch, consistent with the 4F2/6F2 cell-area conventions used
industry-wide. Our 9-15px search-scale pitch (90-150px at reference scale)
places the grid in that regime once the 10x demagnification is applied.

- imec, "DRAM peripheral transistors technology platform":
  https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors
- SemiAnalysis, "The Memory Wall: Past, Present, and Future of DRAM"
  (pitch / feature-size / cell-area relationship, and the 4F2 vs 6F2 cell
  conventions that fix the word-line:bit-line pitch ratio):
  https://newsletter.semianalysis.com/p/the-memory-wall
- Kim, D.-H., et al. / JEDEC DDR device architecture as summarised in
  Jacob, B., Ng, S. & Wang, D. (2007). *Memory Systems: Cache, DRAM, Disk*,
  Morgan Kaufmann, ch. 8 ("DRAM Device Organization") -- the standard
  textbook treatment of the word-line / bit-line / storage-cell array
  topology this generator renders.

## 5. FinFET fin/gate structure

**Choice:** Dense parallel vertical fins crossed by 1-2 horizontal gate
bars.

**Justification:** This mirrors the standard FinFET layout — dense
parallel fins cut by gate lines running perpendicular to them. Published
teardown and metrology data place fin pitch around 30nm and contacted poly
(gate) pitch around 50nm at advanced nodes, with fin pitch near 40nm at the
10nm node and gate pitch in the 70-80nm range at 14nm-class nodes. The
resulting fin:gate pitch ratio of roughly 1:1.5 to 1:2 is what the generator
reproduces (7-12px fin pitch at search scale), scaled to fit the canvas.

- ASIC North, "FinFET Technology and Layout, Part 1" (fin pitch, contacted
  poly pitch, and the fin-grid quantisation that forces gate lines to run
  perpendicular to the fins):
  https://www.asicnorth.com/blog/part-one-finfet-technology-and-layout/
- Sicard, E. "Introducing 5-nm FinFET technology" (tabulated fin and gate
  pitches across 14nm/10nm/7nm/5nm nodes):
  https://hal.science/hal-03254444/document
- Bunday, B. D., et al., dimensions as tabulated and cited in: "hp-finite
  element method for simulating light scattering from complex 3D
  structures" (arXiv:1503.06617), Section 4 / Table 10 -- 22nm-node fin
  pitch ~44nm, gate pitch ~88nm:
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

The sections above cover the original augmentation set. The following model
SEM artefacts the first pass omitted -- scan-system distortion, charging, field
shading, astigmatism -- and each is justified independently.

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

## 15. Aperiodic landmarks: array boundaries and via defects

**Choice:** The only content that breaks the lattice's translational symmetry
is injected deliberately, in three forms: an **array boundary/corner** (the
edge of the memory or logic block), a **missing or doubled via** (`'drop'` /
`'double'` in `dataset_generator.py`), and, for FinFET, a **gate crossing**.
Which of these lies within the reference footprint is recorded per pair as
`landmark`, and whether it does at all as `landmark_in_fov`.

**Justification:** Two independent reasons, one physical and one
information-theoretic.

*Physically*, these are the real aperiodic features of a die. Memory and
logic arrays are finite blocks with abrupt boundaries against periphery
circuitry, and missing-contact / bridged-contact defects at the storage-node
via are among the classical, extensively catalogued defect modes in DRAM
manufacturing — which is precisely why inspection tools look for them.

*Information-theoretically*, a perfectly periodic field is translation-
invariant modulo the pitch, so absolute position within it is not recoverable
from the pixels at all: any localization estimate is correct only up to a
lattice vector. This is the same identifiability argument that motivates
"unique markers" in wafer-alignment practice, and it is why this generator
records solvability as ground truth rather than assuming every pair is
solvable. Measured consequence on our own 100-pair split: sites whose
footprint contains an array corner localize at **95%**, those containing only
a single via defect at **19%**.

- Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray Microanalysis*
  (contrast formation at topographic discontinuities such as block edges,
  §2/§8 above).
- Mack, C. (2007). *Fundamental Principles of Optical Lithography*, Wiley,
  ch. on defectivity — missing/bridged contacts as canonical printed-defect
  modes and their role in yield inspection.
- US Patent 9,430,457, "Ambiguity reduction for image alignment
  applications" — states the periodic-ambiguity problem directly: block-wise
  NCC over a repeating pattern yields multiple indistinguishable peaks, and
  disambiguation requires content that is unique within the search range:
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9430457
- US Patent 11,481,922, "Online navigational drift correction for metrology
  measurements" — the same navigation-error setting, relying on identifiable
  (non-repeating) reference structure to re-anchor position:
  https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11481922
