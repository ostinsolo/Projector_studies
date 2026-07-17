# PROJECT_STATE

## Status

**DONE** — flat-wall production path **VALIDATED**
(ChArUco + planar homography)

## Validation

`VALIDATION_STATE.md` → **VALIDATED**  
Report: `FINAL_VALIDATION_REPORT.md`  
Matrix: `docs/VALIDATION_MATRIX.md`

## Runs

| Role | Path |
|------|------|
| Canonical (immutable) | `procam-test-data/runs/flatwall_20260717_141610` |
| Fresh live regression | `procam-test-data/runs/flatwall_live_20260717_154312` |

## Production command

```bash
procam-calibrate auto-wall --run-dir <NEW_RUN_DIR>
```

## Decisions

- Production = classical ChArUco + planar homography (`--pipeline homography`).
- Shared module: `procam_calibrate/charuco.py`.
- Acceptance uses **independent desired target** metrics (category B), not
  fit residuals or contour AA alone.
- CSPR-Net: experimental non-planar only — not in production state machine.
- GS-ProCams: `REUSABLE_COMPONENTS_ONLY`.

## Live before / after (target-based)

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| target_median_err_px | 66.89 | 20.81 |
| target_p95_err_px | 139.59 | 24.56 |
| horizontal_axis_deviation_deg | 11.27 | 0.025 |
| orthogonality_error_deg | 3.00 | 0.084 |
