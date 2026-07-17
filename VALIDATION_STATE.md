# VALIDATION_STATE

## Status

**VALIDATED**

## Scope

Hardening loop for production flat-wall path only:

- one projector + one fixed iPhone + planar wall
- automatic ChArUco + planar homography + projector pre-warp
- automatic physical validation

Canonical run (immutable): `procam-test-data/runs/flatwall_20260717_141610`  
Fresh live run: `procam-test-data/runs/flatwall_live_20260717_154312`

## Gates

| Gate | Name | Status | Evidence |
|------|------|--------|----------|
| 1 | Contract and documentation consistency | PASS | docs + stale-search |
| 2 | Single implementation source | PASS | `charuco.py` + tests |
| 3 | Coordinate and warp mathematics | PASS | round-trip tests |
| 4 | Metric correctness (independent target) | PASS | `METRIC_DEFINITIONS.md` |
| 5 | Detection coverage | PASS | expected_visible_mask |
| 6 | Synthetic test matrix | PASS | `FAILURE_INJECTION_RESULTS.md` |
| 7 | Canonical run replay | PASS | gate7 JSON |
| 8 | Clean environment reproduction | PASS | `CLEAN_REPRODUCTION_REPORT.md` |
| 9 | Process and state-machine safety | PASS | lock/crash tests |
| 10 | Fresh live regression run | PASS | `flatwall_live_20260717_154312` |

## Authoritative production path

```
discover display and camera
→ display ChArUco board
→ capture
→ detect known projector IDs in camera image
→ estimate H_cam_from_proj
→ calculate H_proj_from_cam
→ define desired target geometry
→ generate projector-space pre-warp
→ display corrected validation board
→ capture
→ compare detected points against the independent desired target
→ save results
```

## Live regression summary

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| target_median_err_px | 66.89 | 20.81 |
| target_p95_err_px | 139.59 | 24.56 |
| horizontal_axis_deviation_deg | 11.27 | 0.025 |
| orthogonality_error_deg | 3.00 | 0.084 |
| coverage_fraction | — | 1.0 |

See `FINAL_VALIDATION_REPORT.md`.
