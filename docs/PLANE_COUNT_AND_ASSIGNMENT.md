# Plane Count and Assignment

**Module:** `procam_calibrate/two_plane.py` (`fit_two_homographies`)

## Procedure

1. Fit one robust homography; accept one-plane if median ≤ 2 px and p95 ≤ 5 px.
2. Otherwise sequential two-RANSAC + iterative assign/refit.
3. Canonicalize labels: plane A = left/top by projector-space centroid (stable).
4. Require ≥12 points/plane, hull area, contiguity, non-interleaving, and two-H robust error < 0.55 × one-H.

## Multi-frame keys

Observations use `"<pattern_index>:<corner_id>"` so repeated ChArUco IDs across patterns never merge.

## Nearly coplanar

Weak folds may correctly remain `one_plane` or `two_plane_rejected`. Extended synthetic case `nearly_coplanar_angle_15` accepts non-crash behaviour.
