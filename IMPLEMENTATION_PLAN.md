# Drift-Sense — Analysis & Implementation Plan

> **STATUS.** Every item in §2–§5 is implemented and the pipeline is verified
> running. Measured on the primary split: **41.0% within 15px / 51.2% of
> solvable trials** (n=100), against a **22.0% / 27.5%** pre-rewrite baseline.
> Verification surfaced five further defects — see §6b — three of which predate
> this work.
>
> **Outstanding:** the final full-set run of the *current* build (two-tier
> rescoring + swept weights) was interrupted. That build measured 47.5% / 57.6%
> on the first 40 pairs of the same split; the 100-pair figure above is from the
> immediately preceding build. To close the gap:
> ```
> python benchmark.py --dataset dataset_primary --out primary_results.json
> python benchmark.py --dataset dataset_stress  --out stress_results.json
> python benchmark.py --dataset dataset_legacy  --out legacy_results.json
> ```
> The stress and legacy splits have **not** been scored against the rewrite at
> all. The commit gate was calibrated and **failed** its own trust criterion
> (§4.2) — no gate ships.

Scope: `semicon/drift_sense/` (the live codebase), `semicon_claudever/` (superseded),
against `semicon problem statement.md`.

All numbers below were **measured in this repo**, not estimated. Probe scripts are
described in §1 so every claim is reproducible.

---

## 0. Verdict — what is already right

Do not rewrite these. They are the strongest parts of the submission and the plan
builds on them:

| Component | Why it stays |
|---|---|
| `lattice.estimate_scale` | Measures magnification from the pitch ratio instead of sweeping. **Median error 0.14%** against the recorded truth. This is the genuinely novel answer to the "10x scale difference problem" the rubric asks about (Slide 5). |
| `lattice.aperiodic_residual` | Spectral notch of the lattice harmonics. Correct identification of *where absolute identity lives* in a periodic array. This is the intellectual core of the submission. |
| Solvability audit in `evaluate.py` | Splitting reported accuracy by whether a landmark is in the FOV is exactly the "failure mode awareness" the statement asks for, and most teams will not do it. |
| `commit_gate.py` | Wilson-lower-bound threshold selection with 5-fold holdout, and a logistic model that was fitted, measured, and *dropped*. The methodology is sound even though the fitted constant needs redoing (§4). |
| Sub-pixel accuracy when it works | Pair 000: predicted `322.345, 557.200` vs truth `322.442, 557.216` → **0.1 px**. |

The problem is not the perception ideas. It is that the pipeline is **tuned to a
dataset distribution that the graders' test set will not share**, and that the
matcher **discards the reference's resolution and the lattice's phase** — the two
largest sources of information in the problem.

---

## 1. Evidence

Four probes were run against the live code (`localize.py`, `dataset_generator.py`).

### 1.1 Baseline on the shipped 100-pair set

```
<=15px 44.0%   <=25px 45.0%   <=50px 50.0%   <=100px 61.0%   median 48.1px
solvable (landmark in FOV)   n=50   <=15px 86.0%
unsolvable (array interior)  n=50   <=15px  2.0%
runtime median 1.6 s/pair
```

### 1.2 Distribution shift — the headline finding

The statement says the reference appears **"somewhere inside"** the search image.
The generator instead places every true site in a **120–220 px annulus around the
centre** (`dataset_generator.sample_true_drift`, `clamp_site`), and `localize.py`
was then fitted to that with `drift_radius=180`.

Re-running the *same generator and same matcher* with placement uniform over the
frame:

| Configuration | ≤15px | solvable ≤15px | median err |
|---|---|---|---|
| Shipped set (near-centre placement) | 44.0% | 86.0% | 48 px |
| **Uniform placement, shipped defaults** | **21.9%** | **43.8%** | 242 px |
| Uniform placement, `drift_radius=900, w_prior=0` | 31.2% | 62.5% | 203 px |
| Uniform placement, full frame, landmark channel off | 21.9% | 43.8% | 159 px |

Two things follow:

1. **Half the headline accuracy is the placement prior, not the matcher.** Deleting
   one mis-specified prior recovers solvable accuracy from 43.8% → 62.5% for +0.3 s.
2. **The safety net does not fire.** `confidence_floor=0.45` widens to a full-frame
   search only when the top NCC is weak. On uniform placement it fired on **4 of 32**
   pairs — because wrong periodic repeats routinely score 0.75–0.91 NCC. Sample of
   total failures: `err=489px conf=0.905`, `err=477px conf=0.814`, `err=413px conf=0.792`.
   The gate thresholds an absolute correlation value, which in a periodic field
   carries no information about whether the *right* repeat was chosen.

### 1.3 The reference's resolution is thrown away

`localize.py` decimates the 1000×1000 reference to ~100×100 and ranks candidates
there — i.e. discrimination happens at exactly the resolution where every lattice
cell is identical. Rescoring the *same* candidate pool at full reference
resolution, on full-frame solvable trials:

| Ranking method | picks the true site | cost |
|---|---|---|
| decimated appearance (shipped) | 7/12 (58.3%) | 2 ms |
| full-resolution appearance | 8/12 (66.7%) | 168 ms |
| **full-resolution aperiodic residual** | **9/12 (75.0%)** | 174 ms |

n=12 is small, so treat the ordering as directional rather than precise — but the
mechanism is not in doubt, and 174 ms against a 1.3 s budget is affordable.

### 1.4 The lattice phase constraint — completely unused

Both images depict **one globally coherent lattice**. Reading the complex Fourier
coefficient at the fundamental gives each image's absolute lattice phase, so the
true reference origin is pinned *modulo the pitch* with no correlation at all.
Measured residual of the true centre against the phase prediction:

```
x (DRAM + FinFET)   n=40   median 0.42 px   p90 0.88 px   max 8.50 px
y (DRAM only)       n=20   median 0.38 px   p90 0.77 px   max 2.25 px
```

**Sub-pixel, from two FFTs, before a single template is slid.** Nothing in the
codebase uses this. Details and the physical justification are in §3.1.

---

## 2. P0 — Submission blockers

These are pass/fail against the statement's explicit requirements. Do them first;
they are hours, not days.

### 2.1 The dataset generator does not accept the mandated parameters

> "Must accept parameters: architecture style (DRAM/FinFET), number of pairs to
> generate, output directory."

`dataset_generator.py:274-278` accepts only `--n`, `--out`, `--seed`, and hardcodes
an alternating 50/50 split at line 287. **`--style` does not exist.**

Fix: add `--style {dram,finfet,both}` (default `both`) and `--difficulty-mix`.
~10 lines.

### 2.2 The deliverable is split across two directories, and neither is complete

| Required artifact | `semicon/drift_sense/` | `semicon_claudever/` |
|---|---|---|
| README | ✅ (also a stale one at `semicon/README_1.md`) | ✅ |
| Dataset generator | ✅ | ✅ (superseded) |
| Inference script | ✅ `localize.py` | ✅ (superseded) |
| Weights + training script | ✅ | ❌ |
| `requirements.txt` | ❌ | ✅ |
| Citations document | ❌ (inline in README only) | ✅ `CITATIONS.md` |
| success/failure PNGs (Slide 6) | ❌ | ✅ |

Fix: promote `drift_sense/` to repo root, port `CITATIONS.md`, `requirements.txt`
and the two case PNGs across, archive `semicon_claudever/` under `legacy/` with a
one-line README pointer.

### 2.3 Contradictory published numbers

`semicon/README_1.md:11` claims **30.0% on 40 pairs**; `drift_sense/README.md:12`
claims **46.0% on 100 pairs**. A reviewer opening the repo sees two different
headline results. Delete `README_1.md`.

### 2.4 Housekeeping

- 8 stale `*_results.json` (~600 KB) in `drift_sense/`. Keep one, gitignore the rest.
- `localize.py` accepts only `--reference/--search`. Add a positional fallback
  (`python localize.py ref.png search.png`) — free insurance against a grader
  invoking it the obvious way. *(Verified: flat imports work from any cwd, since
  Python puts the script's directory on `sys.path`. Not a blocker.)*
- `requirements.txt` must not pin `torch` — the shipped inference path is
  torch-free, and adding a 2 GB dependency to a script a grader must run is pure
  downside. Keep torch in a separate `requirements-train.txt`.

---

## 3. P1 — The matcher: fit the simulated environment and use it fully

### The reframe

The current matcher is **a generic template matcher with corrections bolted on**.
It decimates a 1000×1000 reference to ~100×100, slides it, and then tries to
recover the identity it destroyed using a residual channel and a hand-weighted
fusion.

The environment being matched into is not generic. It has three properties the
matcher never exploits:

1. the lattice is a **globally coherent phase field**, so position is pinned mod pitch;
2. the magnification is **already measured to 0.14%**, so it is a constraint, not a sweep;
3. the reference carries **100× more pixels** than the template actually used.

The redesign is: **measure the geometry exactly, enumerate the only positions
geometry permits, then spend the compute budget discriminating among those few at
full resolution.** Propose cheap, discriminate expensive — the current pipeline
does the opposite.

```
          ref (1000x1000)                    search (1000x1000)
                |                                    |
                +----------- Fourier-Mellin ---------+     joint scale + rotation
                |            (magnitude spectra)     |     from the periodic content
                v                                    v
          derotate ref                        [ scale s, angle th ]
                |                                    |
                +--------- fundamental phase --------+     3.1
                |                                    |
                v                                    v
     TRUE CENTRE IS PINNED MOD PITCH  ->  discrete grid of ~10^4 valid cells
                |
                v
     coarse NCC on the decimated ref, snapped to phase-valid cells   3.2
                |  -> ~15 candidates, all geometrically legal
                v
     FULL-RESOLUTION rescoring: upsample each search crop to 1000x1000,   3.3
     correlate against the full ref AND its aperiodic residual
                |
                v
     decide: peak-to-second-peak ambiguity statistic, not absolute NCC    3.4
                |
                v
     (x, y)  -- sub-pixel comes from the phase, not a parabola fit
```

---

### 3.1 Add the cross-image lattice phase lock *(highest value, ~80 lines)*

**The signal.** `common._line_positions(size, pitch, phase)` anchors both images to
one absolute phase. So the true centre `(cx, cy)` satisfies a hard congruence:

```
cx - F/2  ≡  φ_search,x − φ_ref,x / s   (mod p)
cy - F/2  ≡  φ_search,y − φ_ref,y / s   (mod p)
```

where `F = ref_size / s` is the footprint, `p` the search-space pitch, and `φ` each
image's absolute lattice phase.

**Measuring φ.** Project onto the axis, then read the argument of the *single* DFT
coefficient at the known pitch:

```python
def fundamental_phase(img, pitch, axis):
    prof = img.astype(np.float64).mean(axis=axis)
    n = len(prof); t = np.arange(n)
    c = np.sum((prof - prof.mean()) * np.exp(-2j*np.pi*(n/pitch)*t/n))
    return (-np.angle(c) * pitch / (2*np.pi)) % pitch
```

One coefficient means the noise averages over the entire 1000-sample projection —
this is far more robust than the `argmax`-then-parabola refinement currently used
in `_subpixel`, which reads three adjacent bins of a noisy correlation surface.

**What it buys:**

- **Sub-pixel position for free** (median 0.42 px residual, §1.4), replacing
  `_subpixel` at `localize.py:206-225`.
- **A discrete candidate set.** Continuous ℝ² search collapses to ~(1000/p)² ≈ 10⁴
  lattice cells. This is literally the problem statement's framing — *"which of
  hundreds of nearly identical features?"* — made explicit in the algorithm.
- **A veto independent of appearance.** Any candidate whose phase disagrees is a
  correlation artefact sitting between cells, and can be dropped before it ever
  reaches the fusion stage. This kills a whole class of failure that the current
  `lattice_phase_consensus` only weakly discourages — and which is why that term is
  currently switched off (`W_PHASE = 0.0`, `localize.py:79`).

**Not a simulator artefact.** This is worth stating explicitly on Slide 5, because a
grader will ask. A die's lattice is stepper-printed from one reticle, so it *is*
globally phase-coherent in reality; pattern-placement error is nanometres against a
pitch of tens of nanometres. The constraint is physically right, not a trick that
works because both images came from one `numpy` call. Cite Mack, *Fundamental
Principles of Optical Lithography* (overlay/placement budget) alongside the
existing FFT citations.

**Guard rails.** The `max 8.50 px` outlier in §1.4 is a case where the wrong line
family locked. Gate the phase constraint on `lat_r['quality']` and on the two
axes' phase estimates agreeing with the appearance peak to within ~1 pitch; fall
back to the current behaviour when it does not. Never let this channel fail closed.

---

### 3.2 Stop sweeping what has already been measured

`estimate_scale` returns scale to **0.14% median error**. `localize.py:238` then
sweeps `scale_est ± 0.35` — i.e. **±3.5%, twenty-five times wider than the
measurement error** — at 5 scales × 6 peaks = 30 candidates per window.

This is worse than merely wasteful. Every extra scale injects 6 more near-tied
candidates into the pool, which inflates `n_near_peaks` — the very statistic the
commit gate is calibrated on (`commit_gate.json`, `threshold=5.0`). The sweep
manufactures the ambiguity the gate then measures.

Fix:
- narrow to `scale_est ± max(0.05, 3σ)` where σ comes from `lat_r['quality']`;
- derive the reference pitch band from the measurement instead of the hardcoded
  `REF_PITCH_BAND = (45.0, 320.0)` (`localize.py:62`) — it should be
  `lat_s['pitch'] * (8.5, 11.5)`, which is what `estimate_scale` already computes
  internally at `lattice.py:161`;
- reinvest the saved candidate budget in §3.3.

Also fix `lattice._band_mask` (`lattice.py:39-48`): `period = h / r` assumes square
images. A non-square test image silently corrupts every band mask, every pitch
estimate, and therefore every scale estimate. Use `sqrt(h*w)` or handle the axes
separately.

---

### 3.3 Coarse-to-fine: rank at full reference resolution

Measured in §1.3: 58.3% → 75.0% top-1 on solvable full-frame trials, for ~174 ms.

Implementation:
1. Keep the decimated multi-scale NCC as a **proposal** stage — that is what it is
   good at.
2. Snap survivors to phase-valid cells (§3.1) and dedupe.
3. For the top ~15, upsample the search crop to 1000×1000 (`INTER_CUBIC`) and score
   against the **full reference** *and* against the **full-resolution aperiodic
   residual**. The residual variant was the strongest single ranker measured.
4. Fuse those two at full resolution; the decimated score becomes a proposal
   confidence, not a ranking term.

This is the change that most directly answers *"use the simulated environment
fully"*: the 10× reference exists precisely so that fine structure — via diameter,
edge-brightening profile, gate-crossing geometry — is resolvable. Discriminating at
100×100 throws that away and then complains that the cells look identical.

---

### 3.4 Replace absolute-NCC confidence with an ambiguity statistic

`confidence_floor=0.45` (`localize.py:228`) and `confidence=winner['score']`
(`localize.py:349`) both treat absolute NCC as confidence. §1.2 shows this is
actively misleading: 0.905 NCC on a 489 px error.

In a periodic field the informative quantity is **separation, not height**. Replace
with peak-to-second-peak ratio measured at a separation of **at least one pitch**
(so the same peak's own shoulder does not count as its rival):

```
ambiguity = corr_best / corr_best_at_distance_>=_pitch
```

This is the standard ambiguity statistic from phase-correlation registration and it
is what `n_near_peaks` is gesturing at, computed properly and continuously rather
than as a count against a hardcoded `TIE_EPS = 0.02`.

Then:
- **fallback trigger** becomes low separation, not low height;
- **reported confidence** becomes calibrated and monotone in correctness;
- **commit gate** gets a continuous feature instead of a small integer count, which
  should let `commit_gate.fit` find a genuinely better operating point (§4.2).

---

### 3.5 Estimate rotation properly instead of sweeping or ignoring it

`localize.py` does **no rotation handling at all**. `matcher.py` has a sweep that is
disabled (`rotations = (0.0,)`, `matcher.py:208`) because on this dataset it *hurt* —
correctly diagnosed at `matcher.py:201-207`: maximising over a nuisance parameter
lifts distractors at least as much as the true peak.

That diagnosis is right, and it argues for **measuring** rotation, not for ignoring
it. The generator applies `rot_ref ~ N(0, 1.3°)`; the statement promises rotation
variation and a *noisier* test set, so the true angle may be larger.

`lattice.py:119-141` documents why the current approach fails: the reference's
fundamental sits only ~11 bins from DC, quantising the angle to ~5.2°. That is an
**argmax-bin artefact**, not an information limit, and there are three standard fixes:

1. **Zero-pad ×4 before the FFT.** Interpolates the spectrum; the peak location is
   not bandlimited by the bin grid.
2. **Use every harmonic, not just the fundamental.** The *k*-th harmonic sits at *k*×
   the radius, so its angular quantisation is *k*× finer. `_find_spectral_peaks`
   already locates them (`lattice.py:169`) — take a magnitude-weighted circular mean
   of their angles.
3. **Log-polar registration of the two magnitude spectra (Fourier–Mellin).**
   Magnitude spectra are translation-invariant, so this recovers **scale and
   rotation jointly** regardless of where the reference sits in the frame. Reddy &
   Chatterji, *IEEE TIP* 5(8), 1996. This is the principled version of what
   `estimate_scale` already does by hand, and it would subsume it.

Recommendation: try (1)+(2) first — a ~20-line change to existing functions. Adopt
(3) if the residual angular error stays above ~0.3°. Then derotate the reference
once, up front, and delete the sweep.

---

### 3.6 Make the residual channel a better detector

- **Compute the residual map over the full frame,** not the drift window
  (`localize.py:246`). One FFT; there is no reason to blind it.
- **Normalise the residual by its local envelope** before correlating. The residual
  is sparse and mostly zero; plain NCC over a mostly-zero patch is dominated by
  whichever bright landmark happens to be inside it.
- **Replace the whole-map MAD z-score** (`_robust_z`, `localize.py:184`) with the
  peak-to-second-peak statistic of §3.4. `LANDMARK_DECISIVE_Z = 4.0` is a hand-set
  constant on a scale with no units; the ratio is self-calibrating.
- **`residual_saliency` is computed and then discarded** (`localize.py:270, 351`).
  It answers "is this trial solvable at all?" *without ground truth* — that is
  precisely the Slide 6 "honest failure case" signal. Feed it into the commit gate.

---

### 3.7 The learned component — pick one story and commit to it

`EmbedNet` embeds raw 64×64 patches that are ~95% identical lattice. Its own
docstring in `hypothesis_graph.py:11-13` explains why that cannot work. It is
trained (`reranker.pt`, 139 KB), and then **dropped from the shipped inference
path**. That is the worst of both worlds: a torch dependency in the repo, a DL
claim in the README, and no DL in the script that gets scored.

Two coherent options:

- **(a) Drop it.** Move `reranker_model.py` / `train_reranker.py` / `reranker.pt` to
  `legacy/`, and state plainly on Slide 3 that a spectral-decomposition approach
  beat the learned re-ranker on this problem and why. This is a defensible, honest
  answer and the README already contains most of the argument.
- **(b) Move it where it can win.** Same triplet scaffolding, but embed the
  **full-resolution aperiodic residual** instead of the raw decimated patch. The
  hard-negative construction in `train_reranker.make_negative` (phase-aligned
  distractors — other cells of the *same* image) is already exactly right; it is
  currently being applied to an input with no signal in it.

**Recommend (b)**, with (a) as the fallback if it does not beat the matched filter
on a held-out split. It reuses work already done, gives the rubric a real DL story,
and — unlike the current design — it is pointed at the one representation where a
learned model has something to learn. Whichever wins, ship only the winner.

---

## 4. P2 — Evaluation and calibration honesty

### 4.1 The evaluation set is 50% unsolvable by construction; the test set will not be

`dataset_generator.generate_dram_pair:144` forces landmarks **>180 px away from the
true site** on every `hard` pair, guaranteeing they are outside a ~100 px FOV. Half
the eval set is therefore information-theoretically unsolvable, which is why
"unsolvable ≤15px = 2.0%" — that is chance.

The statement says the test set will *"include **at least one** highly periodic array
region where correct localization is genuinely difficult."* At least one. Not half.

Every constant in the repo was fitted against that 50% base rate: `drift_radius=180`,
`W_PHASE=0.0`, `commit_gate threshold=5.0`, `LANDMARK_DECISIVE_Z=4.0`, `TIE_EPS=0.02`.

Fix:
- add `--difficulty-mix` to the generator and make the **primary** evaluation set
  ~80/20 solvable/unsolvable, matching the statement;
- keep the 50/50 set as a named **stress split**, reported separately — it is a
  genuine strength of the submission, just not a headline;
- refit every constant above on the primary set;
- **add a uniform-placement split.** §1.2 shows this is where the current pipeline
  fails, and "somewhere inside" is what the statement literally says.

### 4.2 Refit the commit gate on the new features

Once §3.4 lands, rerun `calibrate_gate.py` with the continuous ambiguity ratio and
`residual_saliency` as candidate features. Keep the Wilson-bound + 5-fold
methodology exactly as is — it is the most rigorous thing in the repo. The README's
note that the logistic was fitted, measured, and dropped for lack of data should
survive the refit; re-check it rather than assuming it.

### 4.3 Report against a distribution you can defend

Present three numbers on Slide 6, not one:

```
primary  (80/20 solvable, uniform placement)   <-- headline
stress   (50/50 solvable, uniform placement)   <-- failure-mode awareness
legacy   (50/50, near-centre placement)        <-- what the old number meant
```

Volunteering the distribution-shift experiment, rather than quoting the flattering
number, is a stronger position with a semiconductor-industry grader than a higher
number with an unstated assumption behind it.

---

## 5. P3 — Generator realism (the 30% augmentation score)

The statement weights augmentation realism at 30% and demands 2–3 citations per
choice. Current model: Gaussian + Poisson noise, Gaussian blur, gradient-magnitude
edge brightening, small rotation. That is a solid floor and the existing citations
cover it. Gaps, in priority order:

1. **The search image is rendered directly at 1× with integer line positions**
   (`common.render_dram:52-56`, `render_finfet:85-87`). Real 10× demagnification is
   area-averaging through a PSF. Two consequences: the ref/search relationship is
   not physically consistent, and integer rounding stamps a **per-cell aliasing
   fingerprint** into the search image — a simulator artefact a matcher could learn
   to exploit and then fail on real data. Fix: render at ≥4× supersample and
   area-downsample. Do this before §3.3, so full-resolution rescoring is measured
   against an honest image.

2. **Rotation is applied to the reference only, with `fillcolor=bg`**
   (`common.rotate_image:142-146`). This stamps a constant-valued border ring into
   the reference that has no counterpart in the search image, biasing every NCC. Fix:
   render on a larger canvas and crop the valid inscribed region after rotating.

3. **Missing SEM artefacts with well-established citations:** charging-induced
   streaking and drift, scan-line jitter, horizontal scan distortion, non-uniform
   illumination/shading, astigmatism (anisotropic focus). Each is a few lines and
   each comes with textbook support — Goldstein et al., *Scanning Electron
   Microscopy and X-Ray Microanalysis*; Reimer, *Scanning Electron Microscopy*.
   These are the highest-yield additions for the augmentation score.

4. **Edge brightening is symmetric** (`common.edge_brighten:110-117`): isotropic
   gradient magnitude, globally normalised. Real SE edge effect is **directional and
   asymmetric** — a bright band with a decaying tail whose strength depends on the
   local surface tilt relative to the detector. A symmetric operator partially
   cancels under NCC and understates the difficulty. Make it anisotropic with a
   per-image detector direction.

5. **Difficulty is binary and adversarial.** `easy`/`hard` with a hard 180 px
   exclusion is a switch, not a distribution. Replace with a continuous
   *distance-from-true-site-to-nearest-landmark* axis sampled from a realistic
   density; report accuracy against that axis. It is a better graph for Slide 6 and
   it stops the generator from manufacturing an unsolvable half.

6. **Only one landmark per pair.** Real dies have several classes at once. Allow
   independent sampling of array edges, via defects and gate crossings.

---

## 6. Sequencing

Ordered so that each step is measurable before the next begins, and so that if the
clock runs out the work already done is the work that mattered.

| # | Work | Effort | Expected effect |
|---|---|---|---|
| **1** | §2 P0 blockers: `--style`, repo consolidation, delete `README_1.md`, requirements split | 2 h | Removes pass/fail risk. Do first. |
| **2** | §4.1 generator splits: `--difficulty-mix`, uniform placement; regenerate primary/stress/legacy sets | 2 h | You cannot measure anything else honestly until this exists. |
| **3** | §3.2 narrow the scale sweep, derive the pitch band, fix `_band_mask` | 1 h | Runtime down; candidate pool cleaner; unblocks the gate refit. |
| **4** | §1.2 fix: default to full-frame, demote the disk prior to a soft tie-break, §3.4 ambiguity statistic | 3 h | **Measured: solvable 43.8% → 62.5% on uniform placement.** Largest single win. |
| **5** | §3.1 lattice phase lock | 5 h | Sub-pixel position; discrete candidate set; an appearance-independent veto. The idea worth presenting. |
| **6** | §3.3 full-resolution rescoring | 3 h | **Measured: 58.3% → 75.0% top-1 on solvable full-frame trials.** |
| **7** | §5.1 + §5.2 supersampled rendering, rotation border fix | 2 h | Physically honest images; removes an artefact the matcher could learn. |
| **8** | §3.5 rotation estimation (zero-pad + multi-harmonic) | 3 h | Robustness to a test set with more rotation than this one. |
| **9** | §4.2 refit commit gate; §3.6 residual improvements | 2 h | Calibrated confidence on the real distribution. |
| **10** | §5.3–5.6 SEM artefacts, anisotropic edges, continuous difficulty | 4 h | Directly targets the 30% augmentation score. |
| **11** | §3.7 residual-domain re-ranker, or excise the DL path | 4 h | Optional. Ship only if it beats the matched filter held-out. |

Steps 1–6 are the critical path and are roughly two days. They are also where all
the measured gains are.

---

## 6b. Bugs found during verification

Running the rewritten pipeline surfaced four defects that no amount of reading
would have caught. Recorded because three of them predate this work and one of
them is the likely explanation for a chunk of the original architecture's
difficulty.

### 6b.1 NMS was deleting adjacent lattice cells *(pre-existing, significant)*

The suppression radius was `max(20, 2 * pitch)`. The pitch is 7-15px. So the
radius was **always at least one full lattice cell and usually two**, and
non-maximum suppression was removing the neighbouring cells — which are exactly
the mutually exclusive hypotheses the whole system exists to compare.

Whenever a neighbouring cell out-scored the true one on raw appearance, the true
site was **deleted before any re-ranking could see it**. Measured: on trials
where the true site was absent from the candidate pool, the nearest surviving
candidate sat 30-60px away, i.e. two to five cells.

No re-ranker, learned or otherwise, can recover a hypothesis that was suppressed
during proposal. Fixed to `0.7 * pitch` (`NMS_PITCH_FRACTION`).

### 6b.2 Full-resolution rescoring needs sub-pixel crop extraction

`_upsampled_crop` sliced at `int(round(x - foot/2))` and resized. Rounding the
origin costs up to 0.5px at search scale — which is **5px at reference scale**,
enough to decorrelate precisely the fine structure the stage exists to compare.
The integer `foot` compounded it, since the true footprint is fractional.

Measured effect on the true site's rank-1 rate: **25% -> 100%** (n=8) after
replacing the slice with an inverse affine warp at the exact fractional scale
and sub-pixel origin. Full-resolution scoring is far more placement-sensitive
than the decimated scoring it replaces; this is not optional detail.

### 6b.3 Candidate budget did not scale with search area

`keep=15` was calibrated when the search was a 180px disk. Searching the full
frame is ~10x the area for the same budget, and it showed up directly as the
true site not being proposed on half of trials. Now scaled by window area and
capped by `MAX_CANDIDATES`.

### 6b.4 Rotation correction had an inverted sign

`relative_rotation` returns the angle that brings the reference *back* into the
search frame — it is already the correction. Applying `rotate(ref, -angle)`
therefore doubled the misalignment instead of removing it. Verified against the
generator's recorded `rotation_ref_deg`: the estimate is `-rotation_ref_deg` to
a median 0.15 deg.

This was masked by the A/B verification step (§3.5), which correctly rejected
the harmful correction — so the pipeline was not visibly broken, it was just
silently never applying rotation. A guard that hides a bug is still doing its
job, but the bug was only found by probing the estimator against truth
separately from the pipeline that consumes it.

### 6b.5 Two latent crashes

`lattice._find_spectral_peaks` returned `[]` on flat spectra but a 2-tuple
otherwise, while its caller unpacks two values — a crash on exactly the
degenerate inputs where graceful degradation matters most. And `_band_mask`
built its grids with `np.mgrid`, allocating six float64 planes; on the
zero-padded spectra used for angle estimation that was ~750MB of memory traffic
per call and **12 seconds of a 14-second pair**. Rebuilt with broadcasting.

## 7. Risks

- **Everything above is validated against *this* generator.** AM's differs in ways
  nobody can see. This is the argument for §4.1 and §4.3: widen the distribution the
  system is tuned against, and report the spread rather than the best cell.
- **The phase lock is the most powerful and the most brittle idea here.** It assumes
  a single coherent lattice family. If AM's generator tiles a *pattern* with
  internal structure rather than ruling lines, phase recovery may lock to the wrong
  family — that is the `max 8.50 px` outlier in §1.4. It must be gated on a quality
  check and must fall back cleanly (§3.1, "guard rails"). Never let it fail closed.
- **Do not chase the aliasing fingerprint.** Integer-rounded line positions (§5.1)
  make lattice cells subtly distinguishable in *this* simulator. A matcher tuned to
  that would score beautifully here and collapse on any other generator. Fix the
  renderer rather than exploit it.
- **n=12 in §1.3.** The full-resolution result is directional. Re-measure on the
  primary set from step 2 before quoting it on a slide.
- **Two days of the plan is matcher surgery on a pipeline that currently works.**
  Keep `localize.py` runnable at every commit; gate each new channel behind a flag
  so the last-known-good configuration is always one argument away.

---

## Appendix — probes used

Reproducible; each imports the live modules and monkeypatches only the placement
sampler.

| Probe | What it establishes | §  |
|---|---|---|
| `shift_test.py` | Uniform placement collapses shipped defaults 44% → 21.9%; the confidence fallback fires 4/32 | 1.2 |
| `shift_test2.py` | Removing the disk prior recovers solvable 43.8% → 62.5% | 1.2 |
| `fullres_test.py` | Full-resolution residual rescoring: 58.3% → 75.0% top-1 | 1.3 |
| `phase_test.py` | Cross-image phase lock pins the centre to 0.42 px median; scale error 0.14% | 1.4 |
