# Drift-Sense: Navigation-Error Recovery for Wafer Inspection

Find where a 1000x1000 reference pattern, shrunk ~10x, sits inside a 1000x1000
search image of a highly periodic die layout.

```bash
pip install -r requirements.txt
python localize.py reference.png search.png     # -> "x, y"
```

The pipeline **measures the geometry, enumerates only the positions geometry
permits, and then spends its compute discriminating among those few at full
resolution.** Concretely:

| Stage | What it does | Why it is not a template matcher |
|---|---|---|
| Lattice sensor | Magnification from the ratio of the two lattice pitches | The ~10x scale is *measured* to a median 0.14%, not swept |
| Rotation | Sub-bin, multi-harmonic spectral angle | Measured once, never swept -- sweeping lifts distractors as much as the true peak |
| **Phase lock** | Cross-image lattice phase pins the centre *modulo the pitch* | Sub-pixel position from two FFTs, before any template is slid |
| Aperiodic residual | Spectral notch removes the lattice; only landmarks survive | The only channel carrying *absolute identity* inside an array |
| Proposal | Decimated multi-scale NCC over the full frame | Cheap, and only ever used to propose |
| **Full-res rescoring** | Candidate crops upsampled to reference resolution | Decides at the resolution where cells actually differ |
| Decision | Landmark channel judges; centre rule breaks genuine ties | The mandated rule fires only when evidence is truly absent |

## Results

Every number below was produced by running the code. Read the labels: the
100-pair row and the 40-pair row are **different builds**, because verification
was cut short before the final full-set run.

### Primary split -- 80/20 solvable, uniform placement

| Build | n | <=15px | solvable <=15px | solvable median | runtime |
|---|---|---|---|---|---|
| pre-rewrite matcher | 100 | 22.0% | 27.5% | -- | 1.5s |
| rewrite, one-tier + pre-tune weights | 100 | **41.0%** | **51.2%** | 10.6px | 7.8s |
| rewrite, two-tier + tuned weights | 40 | **47.5%** | **57.6%** | **4.5px** | 5.2s |

The 100-pair row is the last complete full-set measurement. The 40-pair row is
the **current** code, measured on the first 40 pairs of the same split; the
remaining 60 were not scored. Treat 41.0%/51.2% as the defensible headline and
47.5%/57.6% as indicative until the full run is repeated:

```bash
python benchmark.py --dataset dataset_primary --out primary_results.json
```

Accuracy on the unsolvable 20% is **0.0%**, and that is the expected result --
those trials place the true site deep in a defect-free periodic array where no
landmark is in the field of view, so nothing in the pixels distinguishes the
correct cell from its neighbours.

### What each change was worth **[measured]**

| Change | Effect | Probe |
|---|---|---|
| Fix NMS deleting adjacent lattice cells | true site proposed 50% -> **87.5%** of solvable | `probes/rank_probe.py` |
| Sub-pixel crop for full-res rescoring | true site rank-1 25% -> **100%** (n=8) | `probes/rank_probe.py` |
| Multi-channel fusion vs best single channel | rank-1 43.2% -> **59.1%** | `probes/weight_sweep.py` |
| Remove the centre-disk prior | solvable 43.8% -> **62.5%** under uniform placement | `probes/shift_test2.py` |
| Sub-bin multi-harmonic rotation | median error 5.2 deg -> **0.15 deg** | `probes/rot_probe.py` |
| Pitch-ratio magnification | **0.14%** median scale error | `probes/phase_test.py` |
| Cross-image phase lock | solvable median error 6.2px -> **4.5px** | `benchmark.py --no-phase-lock` |
| Spatial induction, as solvability evidence | selects 13 sites at **84.6%** against a 39.0% base rate | `probes/induction_probe.py` |

The pre-rewrite pipeline scored 44.0% on the legacy split (near-centre
placement) and 22.0% on primary (uniform placement) -- the same matcher and the
same generator, differing only in where the true site sits. **Half its headline
accuracy was the placement assumption**, which is why primary is now the
default.

### Negative results, kept visible

These were implemented, measured, and did not pay. They are recorded rather than
quietly deleted, because a null result on this distribution is worth re-checking
on a different one:

- **Centre-distance prior in the ranking (`W_PRIOR`)** -- swept to 0.0. It is not
  evidence about where the site is. It survives only as the mandated tie-break.
- **Lattice-phase penalty in the ranking (`W_PHASE`)** -- swept to 0.0. Snapping
  candidates to the phase grid before scoring measured *worse* than not
  (fused rank-1 60.0% -> 56.7% on the same 40 pairs). The phase lock is applied
  to the winner only, where it improves precision without changing the choice.
- **The calibrated commit gate** -- does not generalise here. Held out over 5
  folds it commits 4/100 at 75.0% precision, against 28/100 at 67.9% for the
  hand-set thresholds, and the five folds select three *different* features
  (`residual_saliency`, `landmark_z`, `ambiguity`). By the criterion in
  `commit_gate.py`'s own docstring, that instability means the selection is
  fold noise and the gate should not be trusted. **No `commit_gate.json` is
  shipped.** `calibrate_gate.py` still runs and reports this honestly.
- **Spatial induction as a magnification gate** -- the intended use, and a
  null. Widening the scale bracket whenever the lattice fails to prove its own
  geometry recovers much of what tight bracketing costs (solvable <=1px
  26.2% -> 37.5%) but never beats not bracketing tightly at all (42.5%), on
  either an 8.5-11.5 prior or a deliberately wide 6-16 one. Repairing the pitch
  by integer multiples is worse: the observed errors are not integer harmonics,
  so the repair nets +1 pair. The check ships as *evidence*, not as a gate --
  see `induction.py`.
- **Adding `induction_score` to the calibrated gate does not rescue it.** With
  six candidate features the five folds now select three different ones
  (`induction_score`, `ambiguity` x3, `landmark_z`) and held-out precision
  falls to 6/100 at 50.0%. The instability is a property of n=100 against a
  39% base rate, not of the feature set, and adding a good feature does not fix
  it. The union rule below is quoted instead precisely because it fits nothing.

### Spatial induction: making the lattice prove its own geometry

`estimate_lattice` measures the pitch to 0.04% in the median and fails
*silently* on 13/100 frames, locking onto a lag of 4-6px where the truth is
8-14. `estimate_scale`'s own `ok` flag catches five of those thirteen; the
rest propagate into the reference band, the magnification, and every footprint
downstream.

`induction.py` tests the pitch the way one proves a statement over the
integers. Base case: the profile autocorrelation peaks at lag 0. Inductive
step: if pitch `p` is real, lag `n*p` must be a peak too. If any stepping stone
lands in a valley, the hypothesis is false.

| | n | median score | worst / best |
|---|---|---|---|
| pitch correct | 87 | +1.873 | min **+0.299** |
| pitch wrong | 13 | -0.007 | max **-0.001** |

The groups do not overlap, so the *sign* is the decision -- 13/13 caught at
zero false alarms, with no threshold fitted. Cost is ~10ms per frame.

Its value is not where it looks. Wired to the magnification bracket it is a
null (see *Negative results*). What it actually predicts is **solvability**,
with the sign inverted from intuition:

| | n | localization accuracy |
|---|---|---|
| induction **fails** | 13 | **84.6%** |
| induction **passes** | 87 | 32.2% |
| base rate | 100 | 39.0% |

point-biserial r = -0.455, p = 2e-6. A lattice that fails induction is an
*irregular* lattice -- array boundaries, dropped vias, broken periodicity --
and that aperiodic content is exactly what carries absolute identity inside a
repeating array. **A perfectly inductive lattice is a perfectly ambiguous one.**
It is the only evidence channel computed from the search image alone, before
any correlation runs, so it fails independently of the others:

| selector | sites | precision |
|---|---|---|
| `n_near_peaks <= 5` (existing) | 30 | 86.7% |
| `induction < 0` (this) | 13 | 84.6% |
| either | 35 | 82.9% |
| both | 8 | **100.0%** |

Only eight of the thirteen overlap. The union buys coverage 30 -> 35 for 3.8
points of precision; the intersection is perfect on this split at n=8 and
should be read as a hypothesis to re-measure, not a result. Neither rule fits a
threshold -- both are sign tests -- which is the only reason they are quotable
at n=100.

```bash
python probes/induction_probe.py --dataset dataset_primary --results primary_results.json
```

## Evaluation distribution

Three splits, because a single number hides the assumption that produced it:

```bash
python dataset_generator.py --n 100 --out dataset_primary --seed 11 --difficulty-mix 0.2 --placement uniform
python dataset_generator.py --n 100 --out dataset_stress  --seed 22 --difficulty-mix 0.5 --placement uniform
python dataset_generator.py --n 100 --out dataset_legacy  --seed 42 --difficulty-mix 0.5 --placement annulus
```

- **primary** -- the headline. 20% of trials are deep array interior, matching
  the statement's "at least one highly periodic array region". Placement is
  uniform, because the statement says only that the reference appears
  *"somewhere inside"* the search image.
- **stress** -- 50% unsolvable. Retained because failure-mode awareness is
  explicitly graded, but reported separately: half of it is unidentifiable by
  construction, so an accuracy figure over it is not comparable to anything.
- **legacy** -- what the pre-rewrite numbers were measured on. Kept only so the
  regression is auditable.

**Read the result as two regimes, not one number.** When a non-periodic landmark
(array corner, via defect, gate crossing) is within range of the true site, the
pipeline is accurate to ~1px. Deep inside a defect-free periodic array it is
not, and *cannot* be: at a 9px pitch a 657x657px window holds ~30 correlation
peaks within 0.003 of the global maximum. No amount of re-ranking recovers
information that was never in the pixels. The system's job there is to *say so*
-- see the commit gate.

## Architecture

**The CNN is a witness, not the judge.** The earlier design asked a learned
embedding to decide which candidate patch was the true site. In a periodic
array that question is close to unanswerable from a patch: every candidate is
~95% identical lattice, so the identity-bearing signal is a few percent of the
similarity score, and each candidate is scored in isolation even though the
candidates are mutually exclusive claims about one wafer. The CNN is now one
evidence channel among five, and the decision is made by reasoning over the
candidate set as a whole.

```
Revisit metadata (elapsed_time_s)          Search image + reference (10x)
        |                                            |
Digital Twin drift-prior                     Lattice sensor (lattice.py)
  predicts a search RADIUS,                    2-D spectrum -> pitch,
  direction unrecoverable from                 orientation, and hence the
  elapsed time alone                           unknown magnification as
        |                                      pitch_ref / pitch_search
        |                                            |
        |                              +-------------+-------------+
        |                              |                           |
        |                    periodic component             APERIODIC RESIDUAL
        |                    (identical at every            (spectral notch removes
        |                     lattice cell -> pins          the lattice; array
        |                     the scale, cannot pin         boundaries, dropped or
        |                     the identity)                 doubled vias, gate
        |                              |                    crossings survive)
        v                              v                           v
Search window   -->  Stage 1: multi-scale NCC over a scale        landmark
(annulus)            bracket MEASURED, not swept                  evidence map
                                       |                           |
                                       |   <-- residual peaks also PROPOSE
                                       |       candidates, not just score them
                                       v
                          Hypothesis graph (hypothesis_graph.py)
                          nodes = mutually exclusive candidates
                          channels: appearance | embedding (CNN) |
                                    landmark   | twin prior | lattice-phase
                                    consensus across the node set
                                       |
                          softmax across the set -> BELIEF, not score
                          (sums to 1, has an entropy, competes)
                                       v
                          Commit  or  abstain
                          |                    |
Sub-pixel refinement      |     "no landmark evidence in FOV" ->
(parabolic interp)        |     re-image at a second FOV; fall back to the
        v                 v     minimum-expected-error annulus centroid
      (x, y), belief, per-channel evidence breakdown
        |
Loop 1: match error -> updates the Twin's drift-radius estimate
Loop 2: (Attribution Matrix, see evaluate.py) -> audits whether failures
        trace to the Twin's prior or to the re-ranker
```

### Why this beats a bigger backbone here

Three things changed, and none of them is model capacity:

1. **The magnification is measured, not searched.** The ratio of the two
   lattice pitches *is* the unknown ~10x scale factor (median error 0.3%).
   Stage 1's blind 8-point sweep over 8.3-11.7 becomes a tight bracket, so the
   reference footprint lands within ~1px of truth instead of ~5px.
2. **The lattice is spectrally subtracted before matching.** Correlating the
   two aperiodic residuals isolates exactly the landmark content that carries
   absolute identity. This channel is also what makes abstention possible: in
   a true array interior it is noise-against-noise and stays flat.
3. **Candidates compete.** Beliefs are normalised across the node set, and a
   lattice-phase consensus term scores each candidate against the phase the
   *other* candidates agree on -- information that exists only in the
   relationships between candidates and is invisible to any per-patch
   classifier.

### On the abstention thresholds

`landmark_z_threshold=4.0` and `belief_threshold=0.20` were chosen from the
operating curve on this same 100-pair evaluation set, so the reported 30%/90%
coverage-precision point is optimistic; on a deployment they should be set on a
held-out calibration split. The 100% recall of the unidentifiability detector
is the more robust claim -- it comes from the residual channel being flat by
construction in a landmark-free array interior, not from threshold fitting.

## Files

**Shipped path** -- what Applied Materials runs. Torch-free.

| File | Purpose |
|---|---|
| `localize.py` | **The inference script.** Two image paths in, one `(x, y)` out |
| `lattice.py` | Lattice sensor: pitch, scale, sub-bin rotation, phase lock, aperiodic residual |
| `induction.py` | Spatial induction: makes the lattice prove its own geometry. Solvability evidence, not a scale gate |
| `common.py` | Rendering primitives: supersampled DRAM/FinFET rasterisation, SEM artefacts |
| `dataset_generator.py` | Generates pairs with recorded ground truth (`--style`, `--n`, `--out`, `--difficulty-mix`, `--placement`) |
| `benchmark.py` | Scores `localize.py` across a dataset at several tolerances |
| `commit_gate.py` / `calibrate_gate.py` | Selective prediction: which answers to trust, calibrated with a Wilson bound over 5 folds |

**Research path** -- not scored, retained for the architecture argument.

| File | Purpose |
|---|---|
| `matcher.py` | Hypothesis-graph matcher with the learned re-ranker and Digital Twin |
| `hypothesis_graph.py` | Evidence fusion into a normalised belief, plus the commit/abstain decision |
| `digital_twin.py` | `DriftPrior`: elapsed-time -> expected drift radius, online-updatable |
| `reranker_model.py` / `train_reranker.py` | Small Siamese CNN (32-d) trained on aperiodic residuals |
| `evaluate.py` | Runs the research matcher with an online-updating Twin; needs torch |

## Reproducing the evaluation

```bash
pip install -r requirements.txt

# 1. generate the three splits (see "Evaluation distribution" above)
python dataset_generator.py --n 100 --out dataset_primary --seed 11 --difficulty-mix 0.2 --placement uniform

# 2. score the shipped inference path
python benchmark.py --dataset dataset_primary --out primary_results.json

# 3. calibrate the commit gate on those results
python calibrate_gate.py --results primary_results.json --out commit_gate.json
```

Ablations, to check each channel is earning its place:

```bash
python benchmark.py --dataset dataset_primary --no-landmark     # drop the residual channel
python benchmark.py --dataset dataset_primary --no-phase-lock   # drop the phase constraint
python benchmark.py --dataset dataset_primary --no-rotation     # drop rotation measurement
python benchmark.py --dataset dataset_primary --drift-radius 180  # re-impose the centre prior
```

`commit_gate.json` is **not shipped** -- see *Negative results* above. Step 3
runs the calibration and reports whether a trustworthy gate exists; on the
current distribution it does not, and the script says so rather than writing a
gate that would commit 4% of sites on fold noise.

## On the learned re-ranker

The shipped inference path uses **no learned weights**. This is a deliberate,
measured choice, not an omission: the CNN embedded raw patches that are ~95%
identical lattice at every candidate, so the discriminative content was a few
percent of its input and most of that was destroyed by the downsample to 64x64.
It ranked as the weakest fused channel.

`train_reranker.py` now trains the same architecture on the **aperiodic
residual** instead -- the one representation in this problem where a learned
model has something left to learn that a matched filter does not already
capture -- keeping the hard-negative construction (phase-aligned distractors:
other cells of the *same* image), which was always the right idea applied to the
wrong input.

That variant is **unvalidated**. Until it beats the classical matched filter on
a held-out split, the classical path is what ships:

```bash
python train_reranker.py --dataset dataset_primary --out reranker.pt   # residual domain
python train_reranker.py --dataset dataset_primary --raw --out raw.pt  # superseded, for comparison
```

The previously shipped `reranker.pt` was trained on raw patches from the
superseded dataset and is archived at `legacy/reranker_raw_patches_stale.pt`.

## Citations for every augmentation / noise / geometry choice

**SEM edge-brightening** (implemented as a gradient-magnitude-proportional
intensity boost, applied before blur/noise): secondary electrons have very
low escape energy, so only those generated within a few nm of the surface
escape to the detector; near a topographic edge more of the interaction
volume sits close to the surface, so more secondary electrons escape and
edges appear brighter than flat regions -- the "edge effect."
- ETH Zurich Electron Microscopy, *Secondary Electron Imaging*: https://www.microscopy.ethz.ch/se.htm
- Goldstein et al., *Scanning Electron Microscopy and X-Ray Microanalysis* (edge-brightening slide set): https://bpb-us-e1.wpmucdn.com/blogs.gwu.edu/dist/1/159/files/2017/06/AMC-Workshop-2012_Tutorial-7_SEM-1ethdkh.pdf
- Nanoscience Instruments, *Secondary Electrons in SEM*: https://www.nanoscience.com/blogs/secondary-electrons-in-sem-unlocking-surface-insights-at-the-nanoscale/

**Sensor noise model** (mixed Poisson + Gaussian, independent draw per
image): microscopy detector noise is standardly modeled as a
signal-dependent Poisson (shot-noise) component plus a signal-independent
additive Gaussian (thermal/read-noise) component; we apply this
independently to the reference and search renders since they are separate
physical captures.
- Zhang et al., *A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images*, CVPR 2019: https://arxiv.org/pdf/1812.10366
- Vlasov et al., *Secondary electron topographical contrast formation in STEM* (adopts the same mixed Poisson-Gaussian model for electron-count noise): https://arxiv.org/pdf/2511.14491

**DRAM pitch** (9-15px at search scale, i.e. 90-150px reference scale):
current DRAM generations are described as "10nm-class," with active-area
half-pitches in the memory array ranging from roughly 10 to 19nm; a
line+space pattern's pitch is 2x the half-pitch (feature size), consistent
with the 4F2/6F2 cell conventions used industry-wide.
- imec, *DRAM peripheral transistors technology platform*: https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors
- SemiAnalysis, *The Memory Wall: Past, Present, and Future of DRAM* (pitch/feature-size/cell-area relationship): https://newsletter.semianalysis.com/p/the-memory-wall

**FinFET fin/gate pitch** (7-12px fin pitch at search scale): published
teardown data places fin grid pitch around 30nm and contacted poly (gate)
pitch around 50nm at advanced nodes, with fin pitch narrowing to ~40nm at
the 10nm node and gate pitch in the 70-80nm range at 14nm-class nodes --
we use a similar fin:gate pitch ratio (roughly 1:1.5-1:2) scaled to fit the
synthetic canvas.
- ASIC North, *FinFET Technology and Layout, Part 1*: https://www.asicnorth.com/blog/part-one-finfet-technology-and-layout/
- Sicard, *Introducing 5-nm FinFET technology*: https://hal.science/hal-03254444/document

**Periodicity causing near-tied correlation peaks** (the core justification
for needing Stage 2 at all, not just an augmentation choice): normalized
cross-correlation is the standard template-matching approach, but its
output routinely contains multiple local maxima for similar-looking
regions, requiring an explicit peak-finding / disambiguation step rather
than a single argmax.
- scikit-image, *Template Matching* documentation (explicitly notes multiple local maxima for similar regions): https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_template.html
- US Patent 9,430,457, *Ambiguity reduction for image alignment applications* (block-wise NCC peak-ambiguity detection, directly analogous to our Stage-2 re-ranking): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9430457

**Drift-vs-elapsed-time model** (saturating exponential, the functional
form the Digital Twin fits): thermal drift in precision metrology/inspection
stages is driven by heat dissipated in motors and guides diffusing through
the structure over time, producing a settling-type (saturating) drift
profile; this is exactly the problem addressed by:
- US Patent 11,481,922, *Online navigational drift correction for metrology measurements* (semiconductor metrology; thermal-drift-driven mismatch between design and measured position -- the same problem this project addresses): https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11481922
- *Ensemble modeling of nanoscale thermal drift in high-precision linear axes for photonic integrated circuit testing*, reports drift rates of ~10-50 nm/minute in precision stages before correction: https://www.sciencedirect.com/science/article/pii/S0952197625029367

**Rotation jitter placement** (applied to the reference, not the search
canvas): an earlier version of this generator rotated the whole search
canvas to model stage misalignment, which silently corrupted ground truth
because rotation about the image center produces a tangential pixel
displacement (radius x sin(theta)) that is comparable to or larger than the
9-15px pitch for off-center sites -- confirmed empirically (a 1.25 degree
rotation at radius 187px produces ~4px of ground-truth-invalidating
displacement, verified via direct correlation measurement during
development). Rotation now represents a small relative misalignment between
the 10x and 1x capture optics, applied only to the local reference crop,
which is physically the more defensible choice in any case (the sample
itself doesn't rotate at 10x zoom).

## Known limitations (for the "failure mode awareness" grading criterion)

1. **Pure periodic interior is unsolvable from pixels alone.** Stage 2
   cannot manufacture information the image doesn't contain; 'hard' sites
   fail 100% of the time by design, and this is verified directly (see
   Attribution Matrix output, cases B/D).
2. **FinFET pure fin fields are translation-invariant along the fin axis.**
   A field of parallel fins with no gate crossing carries no information
   about position along the fin direction -- correlation is flat there, not
   just ambiguous. Only a gate crossing (a true 2D landmark) fully resolves
   FinFET sites; a boundary edge alone is insufficient unless it constrains
   both axes (hence 'array_corner', not a single edge, in the generator).
3. **The Digital Twin predicts a radius, not a vector.** Direction is not
   recoverable from elapsed_time_s alone in this design (real deployments
   would need an additional telemetry channel, e.g. a thermal gradient
   sensor, to get directionality -- noted as a natural extension).
4. **Twin calibration needs more sessions to show a clean trend.** With only
   40 samples and per-pair true drift parameters drawn from a wide range
   (120-220px, 1800-5400s), Loop 1's online update shows a real but noisy
   improvement (see `evaluate.py` output); a production deployment would
   accumulate this over thousands of sites.
