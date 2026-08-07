# Drift-Sense: Navigation-Error Recovery for Wafer Inspection

Find where a 1000x1000 reference pattern, shrunk ~10x, sits inside a 1000x1000
search image of a highly periodic die layout.

```bash
pip install -r requirements-inference.txt
python localize.py reference.png search.png     # -> "x, y"
```

Four packages, no torch, no weights to download. `requirements.txt` is the
complete `pip freeze` of the development environment (it includes torch, for the
unshipped training path); `requirements-inference.txt` is the strict subset that
`localize.py` and `dataset_generator.py` actually import, at identical pins.
Either works — the inference set is just ~2 GB smaller.

The pipeline **measures the geometry, enumerates only the positions geometry
permits, and then spends its compute discriminating among those few at full
resolution.** Concretely:

| Stage | What it does | Why it is not a template matcher |
|---|---|---|
| Lattice sensor | Magnification from the ratio of the two lattice pitches | The ~10x scale is *measured* to a median 0.10%, not swept |
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
| rewrite, one-tier + pre-tune weights | 100 | 41.0% | 51.2% | 10.6px | 7.8s |
| shipped (pre accuracy-fix pass) | 100 | 39.0% | 48.8% | 26.3px | 1.9s |
| **current** | 100 | **49.0%** | **61.2%** | **5.9px** | 2.9s |

Every row is a complete 100-pair run. The jump from 48.8% to 61.2% solvable
comes almost entirely from one change in `ACCURACY_FIX_PLAN.md` Step 1: the
decision layer let a fused ranking that is right 66% of the time be
overridden by two rules that were, on this distribution, right only 16% of
the time. Step 5 (giving landmark-proposed candidates a real appearance
score) rides along for correctness, not accuracy. Steps 6 (rotation bound), 7
(wider scale bracket), and a plain-NCC front end were all measured and
**rejected** -- none improved on Step 1 alone by more than sampling noise (1-3
pairs at n=80); see the plan's "Outcome" section for the full comparison.

```bash
python benchmark.py --dataset dataset_primary --out primary_results.json
```

Accuracy on the unsolvable 20% is **0.0%**, and that is the expected result --
those trials place the true site deep in a defect-free periodic array where no
landmark is in the field of view, so nothing in the pixels distinguishes the
correct cell from its neighbours.

### Read the number by regime, not as one figure

The headline 49.0% is a weighted average over a distribution *we* chose. What
actually determines accuracy is **which aperiodic feature is inside the
reference footprint** -- and that turns out to explain essentially all of the
variance, including the apparent DRAM/FinFET gap:

| What is in the field of view | n | <=15px |
|---|---|---|
| Array corner | 40 | **92.5%** |
| Gate crossing only (FinFET) | 19 | 42.1% |
| Single via defect only (DRAM) | 21 | **19.0%** |
| Nothing aperiodic (by construction) | 20 | 0.0% |

Per style this reads DRAM **40.0%**, FinFET **58.0%** -- but that is not an
architecture effect. It is that DRAM's non-corner landmark is a *point* defect
and FinFET's is a *line crossing*. A DRAM reference footprint spans ~78 lattice
cells and a dropped via alters exactly one of them: ~1.3% of the content
separating the true site from its lattice-shifted rival. Measured directly, the
true site's landmark-channel z-score is 3.00 at an array corner (97th percentile
among candidates) but 1.10 at a via defect (84th percentile).

This is a signal-extent limit, not a tuning failure, and the ranking stage is
where it bites: **87.5% of identifiable sites are correctly proposed** by the
candidate stage, but only ~46% survive ranking to first place.

```bash
python probes/rank_probe.py dataset_primary   # proposal vs ranking split
```

### Both splits, current build

| Split | n | solvable | <=15px | solvable <=15px | s/pair |
|---|---|---|---|---|---|
| primary (20% unsolvable, uniform) | 100 | 80 | **49.0%** | **61.2%** | 2.94 |
| stress (50% unsolvable, uniform) | 100 | 50 | 30.0% | 60.0% | 4.51 |

The solvable column barely moves (61.2 vs 60.0) while the headline column swings
19 points. That is the point of reporting both: the
headline tracks how much unsolvable material a split contains, not how good the
matcher is.

### Robustness to conditions the test set may not share

The statement promises the official test set is **more noisy** than ours, and
says nothing about its lattice pitch. Both are now measured rather than assumed,
via two controlled sweeps -- identical seed, geometry, placement and ground
truth across arms, only the swept variable changes:

```bash
python probes/robustness_sweep.py noise    # --noise-scale 1, 2, 3
python probes/robustness_sweep.py pitch    # --pitch-shift -3, 0, +4, +8
```

**Noise: no degradation.** The condition the statement explicitly promises is
the one that costs nothing.

| `--noise-scale` | <=15px | solvable <=15px | s/pair | lattice `scale_ok` |
|---|---|---|---|---|
| 1.0 (baseline) | 49.0% | 61.2% | 2.33 | 95% |
| 2.0 | 49.0% | 61.2% | 2.83 | 89% |
| 3.0 | **52.0%** | **65.0%** | 2.74 | 92% |

Tripling the search-side sensor noise costs nothing measurable -- the 3.0 arm is
*higher*, by 3 pairs, which is sampling noise, not an improvement. This is a
property of the front end being spectral: pitch, orientation and phase are
estimated from integrated Fourier magnitude, where zero-mean sensor noise
averages down. A pixel-domain template matcher would not behave this way.

**Pitch: this is the real exposure.** The statement says nothing about the test
set's lattice pitch, and the lattice sensor is the load-bearing assumption.

| `--pitch-shift` | <=15px | solvable <=15px | s/pair | lattice `scale_ok` |
|---|---|---|---|---|
| -3 (finer) | 43.0% | 53.8% | 3.67 | 85% |
| **0 (tuned band)** | **49.0%** | **61.2%** | 2.26 | 95% |
| +4 (coarser) | 37.0% | 46.2% | 2.65 | 91% |
| +8 (coarser) | **33.0%** | **41.2%** | 2.88 | 89% |

Accuracy falls away in **both** directions -- 20 points of solvable accuracy at
+8. Finer pitch pushes the lattice toward the aliasing limit after 10x
demagnification; coarser pitch puts fewer periods inside the reference
footprint, so the pitch estimate is built from fewer cycles. Either way the
magnification is measured less precisely, the footprint bracket widens, and the
proposal stage degrades before ranking ever runs.

**This is the largest unquantified risk in the submission**, and it is quantified
here rather than left implicit: Applied Materials generate their test set with
parameters known only to them. Two honest mitigations, neither implemented: widen
the tuned band (costs accuracy at the centre, per Step 7 in
`ACCURACY_FIX_PLAN.md`, which was measured and rejected), or detect the
off-band condition and widen the bracket only then -- note `scale_ok` drops
95% -> 85% at -3, so the sensor does partially know when it is out of its range.

The `--noise-scale 1.0` arm reproduces `dataset_primary` bit-for-bit (49.0% /
61.2%), which is the control that makes the other arms readable.

### What each change was worth **[measured]**

| Change | Effect | Probe |
|---|---|---|
| Fix NMS deleting adjacent lattice cells | true site proposed 50% -> **87.5%** of solvable | `probes/rank_probe.py` |
| Sub-pixel crop for full-res rescoring | true site rank-1 25% -> **100%** (n=8) | `probes/rank_probe.py` |
| Multi-channel fusion vs best single channel | rank-1 43.2% -> **59.1%** | `probes/weight_sweep.py` |
| Remove the centre-disk prior | solvable 43.8% -> **62.5%** under uniform placement | `probes/shift_test2.py` |
| Sub-bin multi-harmonic rotation | median error 5.2 deg -> **0.15 deg** | `probes/rot_probe.py` |
| Pitch-ratio magnification | **0.10%** median scale error (p90 6.9% -- see below) | `primary_results.json` |
| Cross-image phase lock | solvable median error 6.2px -> **4.5px** | `benchmark.py --no-phase-lock` |
| Spatial induction, as solvability evidence | selects 13 sites at **84.6%** against a 49.0% base rate | `probes/induction_probe.py` |

The magnification row deserves its tail spelled out, because the median flatters
it: across the 100 primary pairs the relative scale error is **median 0.10%, but
mean 1.85% and p90 6.9%**. The distribution is bimodal, not tight -- the sensor
is either right to a tenth of a percent or it has locked onto the wrong harmonic
entirely. That minority is exactly what `induction.py` detects (13/100 frames, at
zero false alarms), and it is the reason the magnification is bracketed rather
than trusted as a point estimate.

The pre-rewrite pipeline scored 44.0% under near-centre placement and 22.0%
under uniform placement -- the same matcher and the same generator, differing
only in where the true site sits. **Half its headline accuracy was the placement
assumption**, which is why uniform is now the default. (`--placement annulus`
still reproduces the old behaviour.)

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
  folds it commits 24/100 at 79.2% precision, against 33/100 at 66.7%
  in-sample for the hand-set thresholds, and the five folds select three
  *different* features (`n_near_peaks` x2, `landmark_z` x2, `ambiguity` x1)
  with different thresholds even where the feature agrees. By the criterion
  in `commit_gate.py`'s own docstring, that instability means the selection
  is fold noise and the gate should not be trusted. **No `commit_gate.json`
  is shipped.** `calibrate_gate.py` still runs and reports this honestly.
- **Spatial induction as an unconditional magnification gate** -- the original
  intended use, and a null. Widening the scale bracket on *every* frame
  whenever the lattice fails to prove its own geometry recovers much of what
  tight bracketing costs (solvable <=1px 26.2% -> 37.5%) but never beats not
  bracketing tightly at all (42.5%), on either an 8.5-11.5 prior or a
  deliberately wide 6-16 one. Repairing the pitch by integer multiples is
  worse: the observed errors are not integer harmonics, so the repair nets +1
  pair. See `induction.py`. `ACCURACY_FIX_PLAN.md` Step 7 later tried a
  narrower, conditional form of the same idea -- a modest span floor on every
  frame, plus extra widening only on the 13 frames induction actually flags --
  and it was rejected too: measured against the fixed Step 1 decision layer it
  did not beat leaving the bracket alone by more than sampling noise. Both
  forms of this idea are nulls; only the induction *signal* earns its place.
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
| induction **passes** | 87 | 43.7% |
| base rate | 100 | 49.0% |

point-biserial r = -0.346, p = 0.00043. A lattice that fails induction is an
*irregular* lattice -- array boundaries, dropped vias, broken periodicity --
and that aperiodic content is exactly what carries absolute identity inside a
repeating array. **A perfectly inductive lattice is a perfectly ambiguous one.**
It is the only evidence channel computed from the search image alone, before
any correlation runs, so it fails independently of the others:

| selector | sites | precision |
|---|---|---|
| `n_near_peaks <= 5` (existing) | 23 | 95.7% |
| `induction < 0` (this) | 13 | 84.6% |
| either | 31 | 90.3% |
| both | 5 | **100.0%** |

Five of the thirteen overlap. The union buys coverage 23 -> 31 at a cost of 5.4
points of precision (95.7% -> 90.3%); the intersection is perfect on this
split at n=5 and should be read as a hypothesis to re-measure, not a result.
Neither rule fits a threshold -- both are sign tests -- which is the only
reason they are quotable at n=100.

```bash
python probes/induction_probe.py --dataset dataset_primary --results primary_results.json
```

## Evaluation distribution

Two splits, because a single number hides the assumption that produced it:

```bash
python dataset_generator.py --n 100 --out dataset_primary --seed 11 --difficulty-mix 0.2 --placement uniform
python dataset_generator.py --n 100 --out dataset_stress  --seed 22 --difficulty-mix 0.5 --placement uniform
```

- **primary** -- the headline. 20% of trials are deep array interior, matching
  the statement's "at least one highly periodic array region". Placement is
  uniform, because the statement says only that the reference appears
  *"somewhere inside"* the search image.
- **stress** -- 50% unsolvable. Retained because failure-mode awareness is
  explicitly graded, but reported separately: half of it is unidentifiable by
  construction, so an accuracy figure over it is not comparable to anything.

Two further generator flags exist solely to test assumptions rather than to
produce headline splits. Both are no-ops at their defaults, so every number
above is unaffected:

- `--noise-scale K` multiplies **search-side** sensor noise by K (Gaussian sigma
  up, Poisson electron-count scale down). The statement promises a noisier test
  set; this makes that condition reproducible.
- `--pitch-shift PX` offsets the search-scale lattice pitch band. The lattice
  sensor is the load-bearing assumption in `localize.py` and was tuned against
  DRAM 9-14px / FinFET 7-11px; this measures what happens off that band.

Generating a 100-pair split takes **~200s** on one CPU core (supersampled render
at reference resolution, then area-averaged down 10x), measured across the seven
sweep arms above.

**Read the result as two regimes, not one number.** When a non-periodic landmark
(array corner, via defect, gate crossing) is within range of the true site, the
pipeline is accurate to ~1px. Deep inside a defect-free periodic array it is
not, and *cannot* be: at a 9px pitch a 657x657px window holds ~30 correlation
peaks within 0.003 of the global maximum. No amount of re-ranking recovers
information that was never in the pixels. The system's job there is to *say so*
-- see the commit gate.

## Architecture

**Candidates compete; nothing is scored in isolation.** An earlier design asked a
learned embedding to decide which candidate patch was the true site. In a
periodic array that question is close to unanswerable from a patch: every
candidate is ~95% identical lattice, so the identity-bearing signal is a few
percent of the similarity score, and each candidate is judged alone even though
the candidates are mutually exclusive claims about one wafer. That design was
measured and dropped (see *On the learned re-ranker*). What ships reasons over
the candidate set as a whole, with no learned weights.

```
Reference (10x)                              Search image (1x)
      |                                            |
      +---------------------+----------------------+
                            |
                   Lattice sensor (lattice.py)
                   2-D spectrum -> pitch, orientation, and hence the
                   unknown magnification as pitch_ref / pitch_search
                   Sub-bin multi-harmonic rotation; cross-image phase lock
                            |
              +-------------+-------------+
              |                           |
      periodic component           APERIODIC RESIDUAL
      (identical at every          (spectral notch removes the
       lattice cell -> pins         lattice; array boundaries,
       the scale, cannot pin        dropped or doubled vias and
       the identity)                gate crossings survive)
              |                           |
              v                           v
   PROPOSE: decimated multi-scale    landmark evidence map
   NCC over the whole frame, in       |
   a MEASURED scale bracket           |  residual peaks also PROPOSE
              |                       |  candidates, not just score them
              +-----------+-----------+
                          |
              RESCORE at full reference resolution
              (candidate crops upsampled to 10x, where
               cells actually differ -- not at the
               decimated scale where they are identical)
                          |
              DECIDE: appearance + full-res appearance +
              landmark spread-z. The mandated "closest to
              centre" rule fires only on a genuine tie.
                          |
                          v
              sub-pixel refinement -> (x, y)

Solvability evidence, computed from the search image alone:
  induction.py  -- makes the lattice prove its own geometry.
                   A perfectly inductive lattice is a perfectly
                   ambiguous one, so failing induction PREDICTS
                   that the site is localizable (84.6% vs a
                   49.0% base rate).
```

### Why this beats a bigger backbone here

Three things changed, and none of them is model capacity:

1. **The magnification is measured, not searched.** The ratio of the two
   lattice pitches *is* the unknown ~10x scale factor (median error 0.10%).
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
| `requirements-inference.txt` | The 4 packages `localize.py` actually needs. `requirements.txt` is the full `pip freeze` |
| `probes/robustness_sweep.py` | Noise and pitch sweeps -- the two conditions the official test set may not share |
| `deck/build_deck.js` | Regenerates `deck/DriftSense_Submission.pptx` from the measured numbers in one `FACTS` block |

**Learned-ranking experiments** -- measured, documented, and *not* on the shipped
path. Kept because the nulls are the argument for the classical choice.

| File | Purpose |
|---|---|
| `train_ranker.py` / `ranker.npz` | Listwise learned ranker, 257 params, numpy-loadable. +2 pairs held out, p = 0.69 -- not significant, so not wired in |
| `probes/rank_features.py` | The 30 per-candidate + per-frame features the ranker sees |
| `probes/ranker_report.py` | Held-out report with McNemar and a per-landmark breakdown |
| `probes/landmark_ceiling.py` | The oracle test that located the remaining limit in the features, not the model |
| `train_reranker.py` / `reranker_model.py` / `probes/reranker_eval.py` | The CNN re-ranker, measured a strict null |

**Submission artefacts.**

| File | Purpose |
|---|---|
| `deck/DriftSense_Submission.pptx` | The 9-slide i4C submission deck. Team details and the GitHub/video URLs are `«placeholders»` -- fill them before submitting |
| `CITATIONS.md` | 15 sections: every augmentation, noise model and structural parameter, each with 2-3 public references |
| `examples/success_case.png` / `failure_case.png` | The slide-6 visuals: reference, search, predicted and true location |

Rebuild the deck after changing any measured number:

```bash
npm install          # once; pptxgenjs only
node deck/build_deck.js
```

## Reproducing the evaluation

```bash
pip install -r requirements-inference.txt

# 1. generate the splits (see "Evaluation distribution" above)
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

That variant has now been **measured, and it is a null**. Three disjoint splits,
so no split does double duty:

| Split | Seed | Role |
|---|---|---|
| `dataset_train` | 77 | trains the embedding |
| `dataset_stress` | 22 | selects the fusion weight |
| `dataset_primary` | 11 | reports -- used for neither |

Standalone rank-1 of the true site, on the reporting split (n=70 trials where
the true site was proposed at all):

| Channel | rank-1 |
|---|---|
| `coarse` -- decimated multi-scale NCC | 44.3% |
| `fine_lm` -- residual matched filter (**the shipped judge**) | 38.6% |
| `fine_app` -- full-res NCC appearance | 30.0% |
| `cnn` -- 32-d embedding, residual domain | **2.9%** |

The interesting question is not that it is weaker -- a weaker channel can still
pay if it is *decorrelated*, which is exactly how `induction.py` earned its
place. So the additive case was tested directly:

| | n |
|---|---|
| both correct | 2 |
| classical only | 30 |
| **CNN only -- the entire case for adding it** | **0** |
| neither | 38 |

**The CNN is right nowhere the classical fusion is not.** It is strictly
redundant, not complementary, so no fusion weight can extract anything from it.
The sweep agrees without being told: on the selection split -- which the model
never saw -- rank-1 falls monotonically as the CNN is given weight (50.0% at
w=0, 45.2% at 0.5, 40.5% at 1.0, 35.7% at 1.5), so the sweep picks **w = 0**.
Held out, the delta is **+0 pairs (32/70 either way)**.

Training loss reached ~0.001 on ~240 triplets, i.e. the model fit its training
set completely and generalised none of it. That is the honest reading: at this
data scale the discriminative content is a handful of pixels around one defect,
and a triplet objective over 64x64 crops does not recover it.

So the classical path ships, and now that is a **measured** statement rather
than a stated preference. Reproduce it:

```bash
python dataset_generator.py --n 120 --out dataset_train --seed 77
python train_reranker.py --dataset dataset_train --out reranker.pt        # residual domain
python probes/reranker_eval.py --dataset dataset_stress  --dump sel.json
python probes/reranker_eval.py --dataset dataset_primary --dump rep.json
python probes/reranker_eval.py --replay sel.json rep.json
```

The weights themselves are not committed -- they are two measured nulls, and
`train_reranker.py` regenerates either in about a minute (`--raw` for the
superseded raw-patch input). Nothing in `localize.py` loads them, by design.

## On learning the ranking instead of hand-weighting it

The measured bottleneck is not proposal. The true site is **proposed in 87.5%**
of identifiable trials and ranks first in **64.3%**. Everything between those two
numbers is a ranking problem, and the ranker was three hand-set constants:

```
fused = 1.00*coarse + 0.45*fine_app + 0.50*clip(spread_z(fine_lm))/12
```

`train_ranker.py` replaces that with a **listwise** model: each candidate is
scored, a softmax runs across the candidate *set*, and the loss is cross-entropy
against the true candidate. That optimises "pick the right one out of this set",
which is the metric. It sees 30 features -- five channels x (raw, robust z, rank,
margin-to-runner-up) within the candidate set, plus seven per-frame terms
(`scale_ok`, induction, rotation confidence, phase lock, scale span, set size,
fraction of candidates from the landmark map) so it can learn *when to trust
which channel*. 257 parameters. Splits: train seed 101, early-stop seed 22,
report seed 11.

| | n | fusion | learned |
|---|---|---|---|
| **overall rank-1** | 70 | 64.3% | **67.1%** |
| array corner | 40 | 92.5% | 90.0% |
| gate crossing | 18 | 33.3% | 33.3% |
| **via defect** | 12 | **16.7%** | **41.7%** |

**It works on the category that was worst.** Via defects -- the single weakest
sub-population in the system, and the reason DRAM trailed FinFET -- go from
2/12 to 5/12, replicated across three model capacities and a 3x change in
training-set size.

**And it is not significant overall.** +2 pairs at n=70, McNemar exact
p = 0.69, 95% CI [55.5, 77.0] against the baseline's [52.6, 74.5]. So it does
**not** ship, and `localize.py` is untouched. Quoting +2.8 points off n=70 would
be exactly the error the commit-gate section above documents.

### What was ruled out, and how

Three interventions, each measured rather than assumed:

| Intervention | Result |
|---|---|
| CNN embedding as an extra fused channel | **Strict null.** Right on 0 trials the classical fusion gets wrong |
| 3x the training data (77 -> 216 frames) | **No gain.** +2 pairs became +1 |
| Conditioning on landmark type, with an **oracle** | **Worse** (64.3% vs 67.1% global) |

That last one is the informative one. Per-type models are handed the true
landmark type -- information the pipeline does not have at inference -- and still
cannot beat a single global model. So the reason gate crossings rank poorly is
not that one model is being forced to serve two regimes; **the discriminating
information is not in these features at all.** A learned ranker has extracted
approximately what is there, and further gains need new measurements from the
images, not better models on the same features.

```bash
python probes/rank_features.py --dataset dataset_train2 --out feats_train.json
python train_ranker.py --train feats_train.json --val feats_stress.json --out ranker.npz
python probes/ranker_report.py --model ranker.npz --report feats_primary.json
python probes/landmark_ceiling.py        # the oracle-conditioning test
```

`ranker.npz` is committed (4 KB, numpy-loadable, no torch) so the result is
auditable. It is not loaded by `localize.py`, by design.

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
form the generator's drift model uses): thermal drift in precision metrology/inspection
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

1. **Pure periodic interior is unsolvable from pixels alone.** No amount of
   re-ranking manufactures information the image does not contain; 'hard' sites
   fail 100% of the time by design, and the generator records that in
   `landmark_in_fov` so the failure is attributable rather than mysterious.
2. **A point landmark is not enough; the ranking stage is where it fails.**
   Localization accuracy tracks the *spatial extent* of the aperiodic feature in
   the footprint, not the die architecture: 92.5% at an array corner, 42.1% at a
   gate crossing, 19.0% at a single via defect. One altered cell out of ~78 is
   ~1.3% of the discriminating content, and at that level the landmark channel
   does not separate the true site from its lattice-shifted rivals (z = 1.10 vs
   3.00 at a corner). The candidate stage is *not* the bottleneck -- it proposes
   the true site in 87.5% of identifiable trials. Anything that improves this
   has to come from the ranking channels, and the obvious candidate is a matched
   filter at the via scale rather than the current residual-energy score. That
   is untested, so it is written here as a next step, not as a claim.

3. **FinFET pure fin fields are translation-invariant along the fin axis.**
   A field of parallel fins with no gate crossing carries no information
   about position along the fin direction -- correlation is flat there, not
   just ambiguous. Only a gate crossing (a true 2D landmark) fully resolves
   FinFET sites; a boundary edge alone is insufficient unless it constrains
   both axes (hence 'array_corner', not a single edge, in the generator).
4. **Drift direction is not recoverable from elapsed time.** The generator ties
   the true offset to `elapsed_time_s` through a saturating thermal model, so a
   prior can predict how FAR the stage has wandered but not which way. Under
   `--placement uniform` the pipeline therefore uses no positional prior at all;
   a real deployment wanting one would need an extra telemetry channel, e.g. a
   thermal gradient sensor.
5. **The lattice sensor is tuned to a pitch band.** Accuracy falls away in both
   directions outside DRAM 9-14px / FinFET 7-11px at search scale -- 20 points
   of solvable accuracy at `--pitch-shift 8`. See *Robustness* above; this is
   the largest unquantified risk against an unseen test set.
