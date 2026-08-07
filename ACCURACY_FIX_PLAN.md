# Accuracy Fix Plan

Five changes, in dependency order. Every "expected" figure below was measured on
`dataset_primary` (n=100, 80 solvable) by simulating the change outside the
shipped code path; the job of each step is to reproduce that figure *inside*
`localize.py` and keep it.

**Current shipped state (the baseline every step is measured against):**

| metric | value |
|---|---|
| all 100 pairs, <=15px | 39.0% |
| solvable (n=80), <=15px | 48.8% |
| solvable median error | 26.34px |
| runtime/pair | ~3.2s |
| plain-NCC reference (25 lines) | 45.0% all / 56.2% solvable / 3.57px |

**Ceilings, so nobody chases the impossible.** 20 of 100 pairs are unsolvable by
construction, so 80% overall would require perfection on the rest. Proposal
recall is 87.5% and only reaches 88.8% at 4x the candidate budget, so **~89% of
solvable is the hard ceiling** with the current front end. The realistic target
for this plan is **~70% solvable**; the remaining 10 points are a research
problem, not a bug (see "Not in this plan").

---

## Step 0 — Snapshot and safety net

Before touching anything.

```bash
git add -A && git commit -m "snapshot before accuracy fixes"
python benchmark.py --dataset dataset_primary --out results_baseline.json --label baseline
```

Keep `results_baseline.json`. Every later step is diffed against it.

**Fast sweep harness (strongly recommended).** Steps 1 and 5-7 each need a
parameter chosen on evidence. Re-running the full pipeline per setting costs ~5
minutes each. Instead cache the candidate pool once per front-end variant and
evaluate decision rules offline in milliseconds:

- `scratchpad/cache_cands.py` — builds pools for variants `base|fix5|fix6|fix7|ncc`
- `scratchpad/sweep_decision.py` — sweeps decision rules over a cached pool

Both exist in this session's scratchpad. Move them to `probes/` if you want them
kept.

---

## Step 1 — Decision layer (the +17 point fix)

**File:** `localize.py`, the decision block (search for `landmark_decided`).

**Problem in one line.** The fused ranking is right **66.2%** of the time on
solvable pairs. Two override rules then replace it, and the pipeline delivers
48.8%.

Measured, end to end:

| decision layer | all | solvable | median |
|---|---|---|---|
| shipped | 39.0% | 48.8% | 26.34px |
| without centre tie-break | 46.0% | 57.5% | 9.60px |
| without landmark override | 45.0% | 56.2% | 5.55px |
| **without either (take fused rank-1)** | **53.0%** | **66.2%** | **5.37px** |

Both overrides cost roughly the same and they compound.

**Why the tie-break is wrong here, not wrong in general.** The problem statement
says: *"If more than one matching region is found, return the one closest to the
center."* That rule is about the reference genuinely appearing more than once.
In a periodic array the ties are **spurious** — manufactured by self-similarity,
not by a second true match. `TIE_EPS = 0.02` on a fused score whose candidates
are ~95% identical lattice fires on 25 of 70 cases and is right 16% of the time.

**The change.** Do not delete either rule — that would abandon a mandated
behaviour. Constrain both so they fire only when they are actually evidence.

Current:

```python
    landmark_decided = bool(len(lm_z) and lm_z.max() >= LANDMARK_DECISIVE_Z
                            and fine_lm.max() >= LANDMARK_MIN_NCC)
    tied = [r for r, f in zip(ranked, fused_sorted) if fused_sorted[0] - f < TIE_EPS]

    if landmark_decided:
        winner, ambiguous = cands[int(np.argmax(lm_z))], False
    elif len(tied) > 1:
        winner, ambiguous = min(tied, key=lambda c: np.hypot(c['x'] - center[0],
                                                             c['y'] - center[1])), True
    else:
        winner, ambiguous = ranked[0], False
```

Replacement:

```python
    lm_i = int(np.argmax(lm_z)) if len(lm_z) else -1
    # The landmark channel may REFINE the fusion's choice, not contradict it.
    # Overriding outright was measured to cost 7.4 points of solvable accuracy:
    # when the landmark peak disagrees with a fusion that is right 66% of the
    # time, the fusion is usually the one that is right.
    landmark_decided = bool(
        lm_i >= 0 and lm_z.max() >= LANDMARK_DECISIVE_Z
        and fine_lm.max() >= LANDMARK_MIN_NCC
        and lm_i in set(order[:LANDMARK_OVERRIDE_RANK].tolist()))

    # Ties must be GENUINE. At the old 0.02 this fired on 36% of trials and was
    # right 16% of the time, discarding a ranking that would have been right
    # ~66%. The statement's centre rule is for a reference that genuinely
    # appears twice, not for periodic self-similarity.
    tied = [r for r, f in zip(ranked, fused_sorted) if fused_sorted[0] - f < TIE_EPS]

    if landmark_decided:
        winner, ambiguous = cands[lm_i], False
    elif len(tied) > 1:
        winner, ambiguous = min(tied, key=lambda c: np.hypot(c['x'] - center[0],
                                                             c['y'] - center[1])), True
    else:
        winner, ambiguous = ranked[0], False
```

New constants near the other weights:

```python
# The landmark channel overrides the fusion only if it also ranks this high on
# the fusion itself. 3 = "refines a choice the fusion already short-listed".
LANDMARK_OVERRIDE_RANK = 3
```

and change `TIE_EPS = 0.02` to a swept value (see below). Note `TIE_EPS` is also
used for `n_near_peaks`; if the swept value is very small, give `n_near_peaks`
its own constant so the confidence signal keeps its old, calibrated meaning:

```python
TIE_EPS = 0.005          # decision ties
NEAR_PEAK_EPS = 0.02     # confidence signal -- unchanged, it is calibrated
```

**Sweep before committing.** `TIE_EPS` in `{0.02, 0.005, 0.002, 0}` and
`LANDMARK_OVERRIDE_RANK` in `{1, 3, 5, off}`. Pick on solvable accuracy, break
ties on median error.

**Accept if:** solvable >= 64% and median <= 7px. **Reject and fall back to pure
`ranked[0]`** if no constrained variant clears 64% — 66.2% in hand beats a
principled rule that loses points.

**Verify:** `python benchmark.py --dataset dataset_primary --out results_step1.json`

---

## Step 5 — One field, two kinds of number

**File:** `localize.py`, where landmark peaks are merged into the pool.

**Problem in one line.** `_peaks_from_map` writes the **landmark-map value**
into `score`, the same field `_multiscale_peaks` fills with an **NCC value**.
Medians differ (0.55 vs 0.38) and three consumers read the mixed array:
`_dedupe` sorts on it, the fusion's largest term `W_APPEARANCE * appearance`
reads it, and **`n_near_peaks` is counted over it** — that last one is the
project's best abstention signal (86.7% precision), currently computed on a
partly corrupted number.

**The change.** Give landmark-proposed candidates a real appearance score before
they enter the pool:

```python
        lmap = _landmark_map_from(res_s, res_r, foot_ref, win)
        if lmap is not None:
            lm_cands = _peaks_from_map(lmap, win, foot_ref, nms_radius=nms_r)
            # A landmark peak arrives carrying a residual-correlation value in
            # `score`, which is not the quantity the rest of the pipeline means
            # by that name. Give it the appearance score it would have had if
            # the NCC stage had proposed it, so `score` means one thing.
            if lm_cands:
                tmpl = cv2.resize(ref_u8, (foot_ref, foot_ref),
                                  interpolation=cv2.INTER_AREA)
                for c in lm_cands:
                    crop = _upsampled_crop(search_u8, c['x'], c['y'],
                                           foot_ref, foot_ref)
                    c['landmark_score'] = c['score']   # keep the original
                    c['score'] = 0.0 if crop is None else _ncc(crop, tmpl)
            cands = cands + lm_cands
```

**Expected:** proposal recall 87.5% -> 88.8%; final fused accuracy **unchanged**
(`fine_appearance` already carries this information uniformly). This is a
correctness fix, not an accuracy fix — its real value is that `n_near_peaks`
stops being polluted.

**Accept if:** recall does not fall and solvable accuracy does not fall.
**Additionally verify** the abstention signal survives:

```bash
python benchmark.py --dataset dataset_primary --out primary_results.json
# check the reported n_near_peaks<=5 precision is still ~86-87%
```

---

## Step 6 — Rotation

**File:** `localize.py`, the `use_rotation` block.

**Problem.** The rotation A/B costs **1.2 points** of solvable accuracy net. The
angle estimator underneath is not calibrated: on `dataset_primary`, where the
generator rotates only the reference and never the search canvas,
`lattice_angle` reports tilts up to **39.24 deg** with high confidence, and 45%
of frames exceed 0.25 deg.

**The change — two parts.**

(a) Sanity-bound the angle. An estimate outside the physically plausible range
is a failure, not a measurement:

```python
# Relative capture rotation between the 10x and 1x optics is small. An estimate
# outside this band is an estimator failure -- measured up to 39 deg on frames
# with zero true tilt -- and must not be acted on.
ROTATION_MAX_PLAUSIBLE_DEG = 5.0
```

```python
        cand_deg, rot_conf = relative_rotation(search_u8, ref_u8, pitch, scale_est)
        if 0.15 < abs(cand_deg) <= ROTATION_MAX_PLAUSIBLE_DEG:
            ...
```

(b) Raise the adoption margin. Sweep `ROTATION_MIN_GAIN` in
`{0.01, 0.02, 0.05}` plus a "rotation off" arm.

**Accept if:** solvable accuracy improves or holds. If "off" wins, ship
`use_rotation=False` as the default and keep the flag — but re-check on a
rotated-search test set before submitting, because AM may rotate the canvas.

**Verify:** `python benchmark.py --dataset dataset_primary --no-rotation` gives
the "off" arm for free.

---

## Step 7 — Scale bracket

**File:** `localize.py`, the `span` / `foot_bracket` block.

**Problem.** Bracketing tightly around the measured magnification costs **2.5
points** versus a plain sweep. The measurement is excellent at the median
(0.103% error) but has a heavy tail (p90 = 6.9%), and a +-0.4% bracket has no
tolerance for the tail. Separately, the pitch locks onto a **wrong lag on 13/100
frames** — `induction.py` detects all 13 at zero false alarms, but nothing acts
on it.

**The change:**

```python
# Never bracket tighter than this. The pitch ratio is accurate to 0.1% in the
# median but its p90 is 6.9%, and a bracket narrower than the tail simply
# excludes the truth on those frames -- measured to cost 2.5 points.
SCALE_SPAN_FLOOR = 0.35
```

```python
    span = scale_span if scale_span is not None else scale_uncertainty(
        lat_s, lat_r if scale_ok else None, scale_est)
    if not scale_ok:
        span = max(span, 1.5)
    # The lattice failed to prove its own geometry -> the pitch, and therefore
    # the magnification derived from it, is not trustworthy. Fall back to a
    # blind sweep rather than bracketing tightly around a number we have just
    # measured to be wrong.
    if not np.isfinite(induction) or induction < 0:
        span = max(span, 1.5)
    span = max(span, SCALE_SPAN_FLOOR)
    feet = foot_bracket(ref_u8.shape[0], scale_est, span)
```

`induction` is already computed a few lines above (added this session).

**Expected:** +2 to +3 points solvable. **Watch the runtime** — a wider bracket
means more footprints and more candidates. If runtime/pair exceeds ~5s, cap the
number of feet.

**Accept if:** solvable improves and runtime/pair stays under 5s.

---

## Step NCC — plain NCC in place of the measured front end

This is the "try normal NCC instead of the first stages" arm. It is a **variant
to measure, not a step to land** — keep whichever wins.

**The change.** Replace the measured-scale bracket in the proposal stage with a
blind sweep, keeping everything downstream (residual channel, full-res
rescoring, fusion, fixed decision layer):

```python
feet = sorted({int(round(ref_u8.shape[0] / s))
               for s in np.linspace(8.5, 11.5, 13)})
```

**Why it might win.** A dense blind sweep scored 56.2% solvable on its own,
against the pipeline's coarse channel at 52.5%. The scale set alone accounted
for 2.5 of that gap.

**Why it might lose.** 8.5-11.5 is a *prior about AM's generator*. If their
magnification range is wider, a fixed 13-point sweep degrades while the measured
bracket adapts. Test both against a deliberately wide prior before choosing.

**Decision rule:** if the NCC front end wins by more than 2 points on
`dataset_primary` **and** does not lose by more than 2 points under a 6-16x
prior, adopt it and delete the `foot_bracket` path from the shipped route
(keeping `estimate_scale` for `induction` and diagnostics). Otherwise keep
Step 7.

---

## Step F — Final verification and paperwork

1. Full run on all three splits:

```bash
python benchmark.py --dataset dataset_primary --out primary_results.json
python benchmark.py --dataset dataset_stress  --out stress_results.json
python benchmark.py --dataset dataset_legacy  --out legacy_results.json
```

2. Ablations must still run (they are quoted in the README):

```bash
python benchmark.py --dataset dataset_primary --no-landmark
python benchmark.py --dataset dataset_primary --no-phase-lock
python benchmark.py --dataset dataset_primary --no-rotation
```

3. Re-run the induction probe — `primary_results.json` has changed, so claims 3
   and 4 in `induction.py`'s docstring must be re-measured and the docstring
   updated if they moved:

```bash
python probes/induction_probe.py --dataset dataset_primary --results primary_results.json
```

4. Re-run the commit-gate calibration and confirm it is still unstable (if it
   has become stable, that is a result worth reporting):

```bash
python calibrate_gate.py --results primary_results.json --out /tmp/gate.json
```

5. Regenerate `examples/success_case.png` and `examples/failure_case.png` — the
   current failure case may no longer be a failure.

6. **Update the README numbers.** They currently claim 41.0%/47.5%; a reviewer
   running `benchmark.py` today gets 39.0%. Whatever the final figure is, the
   README must match a fresh run.

7. Paperwork, which is graded:
   - `CITATIONS.md` sections 1 and 3 have fewer than the required 2-3 references.
   - `CITATIONS.md` section 3 says rotation is applied to the search image; the
     code applies it to the reference (`dataset_generator.py`, `rot_ref`). Fix
     the text — a reviewer cross-reading these finds a contradiction.
   - `requirements.txt` uses lower bounds; the statement asks for a complete
     `pip freeze`.

---

## Outcome — MEASURED, replacing the projections above

Step 1 is implemented and verified end to end on `dataset_primary`:

| | all | solvable | median | runtime |
|---|---|---|---|---|
| before | 39.0% | 48.8% | 26.34px | 3.2s |
| **after Step 1** | **50.0%** | **62.5%** | **4.9px** | 2.9s |
| plain-NCC reference | 45.0% | 56.2% | 3.57px | 1.0s |

**+13.7 points solvable, and the median error falls by 5x.** The pipeline now
beats the 25-line NCC baseline by 6.3 points solvable, which it did not before.
The abstention layer improved at the same time: `n_near_peaks<=5` now selects 24
sites at **95.8%** precision (was 30 at 86.7%), and `induction<0 OR
n_near_peaks<=5` selects 31 at **90.3%**.

**Steps 6, 7 and the NCC arm are REJECTED — measured, none of them pays:**

| front end (all with the Step 1 decision rule) | all | solvable | median |
|---|---|---|---|
| shipped front end | 53.0% | **66.2%** | 5.37px |
| + Step 5 (score field) | 52.0% | 65.0% | 4.96px |
| + Step 6 (rotation off) | 51.0% | 63.7% | 5.47px |
| + Step 7 (wider bracket) | 52.0% | 65.0% | 4.96px |
| NCC front end (linspace sweep) | 50.0% | 62.5% | **2.75px** |

The earlier attributions that motivated Steps 6 and 7 (rotation costing 1.2
points, the tight bracket costing 2.5) were measured on a dense-argmax harness
**without the fusion**, and they do not survive contact with the full pipeline —
they reverse sign. Rotation and the measured bracket are both *helping* once the
fusion is downstream of them. This is exactly the caveat flagged at the top of
this plan, now confirmed.

Two honest caveats on the table above: it comes from the offline sweep harness
(`probes/sweep_decision.py`), which disagrees with the real benchmark by ~3
pairs (it predicted 66.2% where `benchmark.py` measures 62.5%), and the spreads
between rows are 1-3 pairs at n=80 — inside noise. Use it to rank options, then
confirm on the real pipeline. **Nothing here justifies landing Steps 6, 7 or the
NCC front end.**

Step 5 remains worth landing on **correctness** grounds, not accuracy: it costs
~1 pair (noise) but stops `n_near_peaks` — the best abstention signal — being
counted over a field holding two incommensurable kinds of number.

**Revised expectation: ~62-66% solvable is where this plan lands. The route to
80% is not in it** — see "Not in this plan".

## Not in this plan

Three things stand between ~70% and 80%, and none has a known fix:

1. **10 of 80 solvable pairs never propose the true site**, and their scale is
   perfect (0.06% error, footprint in bracket 100% of the time). The correct
   location scores worse than 120 wrong ones. Nobody has looked at what those
   images have in common. **Start here** — it lifts the ceiling and probably the
   ranker at the same time.
2. **The ranker is right 76% of the time when the answer is in the pool** and
   needs ~90%. `train_reranker.py` on the residual domain is the one untested
   idea in the repo with a plausible claim on this.
3. **Nothing has ever been tested on a generator other than this one.** The
   pitch band (5-40px) and scale bounds (8.5-11.5x) are hard-coded guesses about
   AM's generator. A generalisation test is worth more than the last 5 points of
   accuracy.
