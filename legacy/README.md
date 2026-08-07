# legacy/

Superseded material, kept for provenance. Nothing here is on the submission path.

## `claudever/`

The second of the two codebases this project was merged from. Its useful parts
were absorbed into the root package and it is retained only so the merge is
auditable:

- the **two-image inference interface** (`localize.py --reference --search`,
  no metadata, no weights) — carried forward into the root `localize.py`;
- the **bounded-drift prior** — carried forward, but demoted to a soft
  tie-break after it was measured to cost more than it earned once the true
  site is not constrained to sit near the frame centre
  (see `IMPLEMENTATION_PLAN.md` §1.2);
- `CITATIONS.md` — promoted to the repo root and extended;
- `success_case.png` / `failure_case.png` — promoted to `examples/`.

Its dataset generator and localizer are both strictly worse than the root
versions and should not be run.

## `results/`

`merged_results.json` — the 100-pair run of the pre-rewrite pipeline, kept as
the "legacy" baseline row in the README results table. Regenerating it requires
checking out an earlier revision; it is retained rather than reproduced.
