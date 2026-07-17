# Two-Plane Residual Refinement

Homography-only coupled refinement (no CSPR).

## Method

For each plane after the first corrected capture:

1. Fit observed corrected source→camera transform on evaluation keys.
2. Compare with predetermined `H_desired_cam_from_source` / desired points.
3. Classify residual (translation / projective / noisy / …).
4. Derive full-strength candidate forward pre-warp:
   `H_proj' = H_proj @ inv(H_cam_obs) @ H_desired`
5. Interpolate strengths `1.0, 0.75, 0.5, 0.25` in **projector control-point
   space** (not coefficient lerp), then refit each H.
6. Treat planes A and B as a **coupled** candidate pair.

## Acceptance

Accept only when all hold (jitter-aware):

- both plane medians improve or already pass
- both plane p95 values improve or already pass
- global median and p95 improve
- seam mismatch does not regress
- coverage and mask invariants remain valid

Reject and roll back the complete pair when either plane or the seam regresses.
Never keep plane A from one iteration and plane B from another without joint
validation.

## Limits

- Maximum physical refinement iterations: **4**
- Always preserve `H_proj_from_source_plane_*_best.npy`,
  `prewarp_piecewise_best.png`, `best_metrics.json`

## Module

`procam_calibrate/two_plane_refine.py`
