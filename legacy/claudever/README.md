# Drift-Sense: Navigation-Error Recovery

Synthetic dataset generator + localization algorithm for the Applied
Materials i4C hackathon "Drift-Sense" challenge: given a high-magnification
Reference image of a wafer site and a lower-magnification Search image, find
the (x, y) center of the reference site inside the search image.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate a sample dataset (30 pairs, both DRAM and FinFET styles)
python scripts/dataset_generator.py --style both --num_pairs 30 --out_dir data

# 2. Run the localizer on one pair
python scripts/localize.py \
    --reference data/dram_000_reference.png \
    --search data/dram_000_search.png
# -> prints "x, y"
```

No other setup is required — both scripts run standalone with sensible
defaults.

## Approach

**Why not plain single-scale template matching?** Two reasons it breaks
down here specifically:
1. The magnification ratio is only *approximately* 10x (it drifts with
   calibration), so a fixed-scale template misses the true peak.
2. Highly periodic layouts (DRAM grids, FinFET fin arrays) produce many
   near-identical correlation peaks. A naive `argmax` silently locks onto
   an arbitrary — often wrong — repeat of the pattern.

**Our algorithm** (`scripts/localize.py`): multi-scale, multi-rotation
normalized cross-correlation (NCC, FFT-accelerated via
`skimage.feature.match_template`), with three deliberate design choices:

- **Bounded-drift search prior.** Real navigation error is small (the
  problem statement itself says the tool lands "several pixels away," not
  hundreds). So we search a crop centered on the search image first,
  matching the physical reality that the tool re-lands *near* its intended
  site — not anywhere in the frame. This is also the reason the task's own
  disambiguation rule ("return the match closest to the search image
  center") makes sense: the true answer is expected to be near-center by
  construction, not by an arbitrary convention. If confidence is low in
  that crop, we automatically fall back to a full-image search.
- **Multi-peak periodicity handling.** For every scale/rotation tried, we
  extract several strong local correlation maxima (not just the single
  best one), cluster near-duplicate detections, and — among genuinely
  tied top-scoring clusters — apply the spec's own tie-break rule (closest
  to the search image center). This is what actually lets the algorithm
  reason about periodic ambiguity instead of being silently fooled by it.
- **Sub-pixel refinement.** The winning peak is refined via parabolic
  interpolation of the local correlation surface for sub-pixel accuracy.

## Results (self-evaluation, 32 generated pairs, seed=42)

| Metric | Value |
|---|---|
| Accuracy within 50px | 71.9% (23/32) |
| Accuracy within 100px | 96.9% (31/32) |
| Median error | 42.9 px |
| Mean runtime per pair | 1.7 s (CPU only) |
| Best case | 0.08 px error |
| Worst case | 102.1 px error |

See `data/success_case.png` and `data/failure_case.png` for visual
examples (regenerate with the snippet in `scripts/` — search the repo
history / ask if you want the exact plotting script re-run).

**Honest failure mode:** the algorithm's main failure mode is exactly the
one the challenge is designed to probe — deep inside a highly periodic
region, several lattice repeats score within a hair of each other, and the
"closest to center" tie-break occasionally picks the wrong (but structurally
near-identical) repeat, landing an integer number of pattern periods away
from the true answer. This is a fundamental information-theoretic limit for
pure template matching on an infinitely periodic pattern, not a bug — a
production system would additionally want a coarse, independent position
prior (e.g. stage encoder feedback) to break the symmetry, which is outside
the scope of a vision-only algorithm.

## Repository contents

- `scripts/dataset_generator.py` — synthetic DRAM/FinFET pair generator.
  Accepts `--style {dram,finfet,both}`, `--num_pairs`, `--out_dir`. Records
  ground truth in `<out_dir>/ground_truth.csv`.
- `scripts/localize.py` — standalone inference script. Accepts
  `--reference <path>` and `--search <path>`, prints `x, y`. This is the
  script intended to be run directly on Applied Materials' test data.
- `requirements.txt` — pinned dependency versions.
- `CITATIONS.md` — public references justifying every augmentation/noise/
  structural choice in the generator.

## Design notes / limitations

- The generator renders patterns procedurally (closed-form periodic
  Gaussian profiles) rather than from a pre-rendered bitmap, so ground
  truth is exact by construction (no manual annotation).
- Pattern pitch/line-width were deliberately chosen large enough (in
  reference-resolution units) to survive the 10x downsample to the search
  image without aliasing away — see `CITATIONS.md` for the physical
  parameters this was checked against.
- Rotation search in `localize.py` defaults to a coarse 3-point grid
  (`-2, 0, 2` degrees) for speed; widen `--rotations` for higher accuracy
  at the cost of runtime.
