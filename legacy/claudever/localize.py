"""
localize.py
===========
Navigation-error-recovery localization: given a Reference image and a
Search image, find the (x, y) pixel center of the reference pattern's
location inside the search image.

Approach: multi-scale, rotation-tolerant Normalized Cross-Correlation (NCC)
with explicit periodicity disambiguation and sub-pixel refinement.

Why not plain single-scale template matching?
  1. The true magnification ratio is only APPROXIMATELY 10x -- it drifts.
     A fixed-scale template misses the true peak. We search a band of
     scales around the nominal ratio.
  2. Periodic layouts (DRAM grids, FinFET fin arrays) produce many
     near-identical correlation peaks. Taking a naive argmax silently
     picks an arbitrary -- often wrong -- repeat. We instead collect all
     near-tied peaks and apply the disambiguation rule specified by the
     task: if genuinely ambiguous, prefer the candidate closest to the
     search image's center.
  3. A single best-scale/best-location pixel is coarse. We refine the
     winning peak to sub-pixel precision with a parabolic fit to the local
     correlation surface.

Usage
-----
    python localize.py --reference ref.png --search search.png
    python localize.py --reference ref.png --search search.png --json
    python localize.py --reference ref.png --search search.png \
        --scale_min 0.08 --scale_max 0.12 --n_scales 21 \
        --rotations -2 -1 0 1 2

Output
------
Prints "x, y" (predicted center, sub-pixel float, search-image coordinates)
to stdout. With --json, prints a JSON object with the prediction plus
diagnostic info (confidence, whether periodicity ambiguity was detected,
which scale/rotation won).

This script has no manual-edit requirements: all parameters have sensible
defaults tuned to the Drift-Sense task (~10x magnification ratio, small
rotation jitter) and can be overridden via CLI flags for different test
conditions.
"""

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate
from skimage.feature import match_template, peak_local_max
from skimage.transform import resize as sk_resize


def load_gray(path):
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def normalize_contrast(img):
    """CLAHE-like local contrast normalization is overkill for a hackathon
    scaffold; a robust global normalization (percentile stretch) is enough
    to bring reference/search onto comparable footing since NCC is already
    mean/variance invariant per-window."""
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        return img
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def best_peak(corr_map):
    """Return (score, row, col) of the global max in a correlation map."""
    idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    return corr_map[idx], idx[0], idx[1]


def subpixel_refine(corr_map, row, col):
    """Parabolic (quadratic) interpolation of the correlation peak using
    its immediate neighbors, for sub-pixel accuracy. Falls back to the
    integer location near the map boundary."""
    h, w = corr_map.shape
    if 1 <= row < h - 1 and 1 <= col < w - 1:
        dx = 0.5 * (corr_map[row, col + 1] - corr_map[row, col - 1]) / (
            corr_map[row, col + 1] - 2 * corr_map[row, col] + corr_map[row, col - 1] + 1e-8)
        dy = 0.5 * (corr_map[row + 1, col] - corr_map[row - 1, col]) / (
            corr_map[row + 1, col] - 2 * corr_map[row, col] + corr_map[row - 1, col] + 1e-8)
        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)
        return row + dy, col + dx
    return float(row), float(col)


def localize(ref_img, search_img, scale_min=0.075, scale_max=0.125, n_scales=11,
             rotations=(-2, 0, 2), tie_epsilon=0.03, verbose=False,
             search_radius_px=110, confidence_floor=0.45, full_search=False):
    """
    Core algorithm. Returns (x, y, diagnostics_dict).

    x, y are in SEARCH image pixel coordinates (float, sub-pixel), measured
    at the CENTER of the matched reference footprint.

    search_radius_px: navigation drift between visits is physically bounded
    (the tool re-lands NEAR its intended site). By default we therefore
    search a crop centered on the search image -- of half-width
    `search_radius_px` plus template margin -- rather than the full frame.
    This is both faster (smaller correlation maps) and more accurate (fewer
    unrelated periodic repeats to get confused by). If the best match found
    there is low-confidence, we fall back to a full-image search
    (set full_search=True to force the full-image search from the start).
    """
    ref_img = normalize_contrast(ref_img)
    search_img_full = normalize_contrast(search_img)

    crop_offset = (0, 0)
    search_img = search_img_full
    if not full_search:
        h, w = search_img_full.shape
        cy, cx = h / 2.0, w / 2.0
        # generous margin beyond the drift radius for template half-width
        pad = max(ref_img.shape) * scale_max + 5
        r0 = max(0, int(cy - search_radius_px - pad))
        r1 = min(h, int(cy + search_radius_px + pad))
        c0 = max(0, int(cx - search_radius_px - pad))
        c1 = min(w, int(cx + search_radius_px + pad))
        search_img = search_img_full[r0:r1, c0:c1]
        crop_offset = (r0, c0)

    scales = np.linspace(scale_min, scale_max, n_scales)
    candidates = []  # each: dict(score,row,col,tmpl_h,tmpl_w,scale,rot)

    for rot in rotations:
        if rot == 0:
            ref_rot = ref_img
        else:
            ref_rot = nd_rotate(ref_img, angle=-rot, reshape=True, mode="nearest", order=1)

        for scale in scales:
            th = max(8, int(round(ref_rot.shape[0] * scale)))
            tw = max(8, int(round(ref_rot.shape[1] * scale)))
            if th >= search_img.shape[0] or tw >= search_img.shape[1]:
                continue
            template = sk_resize(ref_rot, (th, tw), anti_aliasing=True, anti_aliasing_sigma=1.0, order=1)

            corr = match_template(search_img, template, pad_input=False)
            # Extract several strong local maxima (not just the single
            # global best) so that genuinely near-tied peaks -- the
            # signature of periodic ambiguity -- aren't silently dropped
            # just because they weren't the #1 peak at this particular
            # scale/rotation.
            min_dist = max(3, min(th, tw) // 3)
            peak_rc = peak_local_max(
                corr, min_distance=min_dist, threshold_rel=0.75, num_peaks=8,
            )
            if len(peak_rc) == 0:
                continue
            for r, c in peak_rc:
                score = float(corr[r, c])
                center_r = r + th / 2.0
                center_c = c + tw / 2.0
                candidates.append({
                    "score": score, "row": center_r, "col": center_c,
                    "th": th, "tw": tw, "scale": scale, "rot": rot,
                    "corr_map": corr, "raw_r": int(r), "raw_c": int(c),
                })

    if not candidates:
        raise RuntimeError("No valid scale/rotation produced a template smaller than the search image.")

    candidates.sort(key=lambda d: -d["score"])
    top_score = candidates[0]["score"]

    # Merge near-duplicate detections (same physical site found at
    # adjacent scales/rotations) by clustering on (row, col) proximity,
    # keeping the highest-scoring representative of each cluster.
    clustered = []
    dedup_radius = max(candidates[0]["th"], candidates[0]["tw"]) * 0.5
    for cand in candidates:
        matched = False
        for cluster in clustered:
            rep = cluster[0]
            if np.hypot(cand["row"] - rep["row"], cand["col"] - rep["col"]) < dedup_radius:
                cluster.append(cand)
                matched = True
                break
        if not matched:
            clustered.append([cand])
    cluster_reps = [max(c, key=lambda d: d["score"]) for c in clustered]
    cluster_reps.sort(key=lambda d: -d["score"])

    # Among distinct clusters, find those tied for the top score -- this is
    # the periodicity-ambiguity case.
    tied = [c for c in cluster_reps if (top_score - c["score"]) <= tie_epsilon]

    full_h, full_w = search_img_full.shape
    img_center = (full_h / 2.0, full_w / 2.0)

    ambiguous = len(tied) > 1
    if ambiguous:
        winner = min(
            tied,
            key=lambda c: np.hypot(
                (c["row"] + crop_offset[0]) - img_center[0],
                (c["col"] + crop_offset[1]) - img_center[1],
            ),
        )
    else:
        winner = cluster_reps[0]

    # Low-confidence result within the bounded-drift crop -> fall back to a
    # full-image search once, in case drift exceeded the expected radius.
    if (not full_search) and winner["score"] < confidence_floor:
        if verbose:
            print(f"[localize] low confidence ({winner['score']:.2f}) in "
                  f"drift-radius crop -> falling back to full-image search",
                  file=sys.stderr)
        return localize(ref_img, search_img_full, scale_min=scale_min, scale_max=scale_max,
                         n_scales=n_scales, rotations=rotations, tie_epsilon=tie_epsilon,
                         verbose=verbose, search_radius_px=search_radius_px,
                         confidence_floor=confidence_floor, full_search=True)

    # Sub-pixel refine the winning peak on its own correlation map.
    r_sub, c_sub = subpixel_refine(winner["corr_map"], winner["raw_r"], winner["raw_c"])
    final_row = r_sub + winner["th"] / 2.0 + crop_offset[0]
    final_col = c_sub + winner["tw"] / 2.0 + crop_offset[1]

    diagnostics = {
        "confidence": winner["score"],
        "scale_used": winner["scale"],
        "rotation_used_deg": winner["rot"],
        "periodicity_ambiguous": ambiguous,
        "num_tied_clusters": len(tied),
        "num_distinct_clusters": len(cluster_reps),
        "full_image_search_used": full_search,
    }
    if verbose:
        print(f"[localize] {len(candidates)} scale/rotation candidates -> "
              f"{len(cluster_reps)} distinct clusters, {len(tied)} tied "
              f"(ambiguous={ambiguous}, full_search={full_search})", file=sys.stderr)

    return final_col, final_row, diagnostics


def main():
    ap = argparse.ArgumentParser(description="Locate a reference pattern inside a search image.")
    ap.add_argument("--reference", required=True, help="Path to reference image.")
    ap.add_argument("--search", required=True, help="Path to search image.")
    ap.add_argument("--scale_min", type=float, default=0.075)
    ap.add_argument("--scale_max", type=float, default=0.125)
    ap.add_argument("--n_scales", type=int, default=11)
    ap.add_argument("--rotations", type=float, nargs="*", default=[-2, 0, 2])
    ap.add_argument("--json", action="store_true", help="Print full JSON output instead of 'x, y'.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref_img = load_gray(args.reference)
    search_img = load_gray(args.search)

    t0 = time.time()
    x, y, diag = localize(
        ref_img, search_img,
        scale_min=args.scale_min, scale_max=args.scale_max, n_scales=args.n_scales,
        rotations=args.rotations, verbose=args.verbose,
    )
    diag["runtime_sec"] = time.time() - t0

    if args.json:
        print(json.dumps({"x": x, "y": y, **diag}, indent=2))
    else:
        print(f"{x:.3f}, {y:.3f}")


if __name__ == "__main__":
    main()
