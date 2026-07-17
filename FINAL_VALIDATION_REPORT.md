# Final Validation Report — Flat-Wall Homography Production

## Verdict

**VALIDATED**

Production path is mathematically consistent, independently measurable,
reproducible from a clean environment, and confirmed on a fresh physical run.

## One command

```bash
procam-calibrate auto-wall --run-dir <NEW_RUN_DIR>
```

## Validation matrix

See `docs/VALIDATION_MATRIX.md` (all gates PASS).

## Canonical replay (immutable inputs)

Run: `procam-test-data/runs/flatwall_20260717_141610`

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| target_median_err_px | 75.55 | 18.24 |
| target_p95_err_px | 164.61 | 21.47 |
| horizontal_axis_deviation_deg | 12.69 | 0.033 |
| coverage vs prewarp expected | — | 27/29 ≈ 0.93 |

Raw captures/patterns hashes unchanged during replay.

## Fresh live regression

Run: `procam-test-data/runs/flatwall_live_20260717_154312`

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| target_median_err_px | 66.89 | 20.81 |
| target_p95_err_px | 139.59 | 24.56 |
| horizontal_axis_deviation_deg | 11.27 | 0.025 |
| orthogonality_error_deg | 3.00 | 0.084 |
| coverage_fraction | — | 1.0 |

Homography matrices differ from the canonical run (not reused).
Artifacts: `desired_target.json`, `H_*.npy`, `prewarp_charuco.png`, `metrics.json`.

## Clean environment

Documented in `docs/CLEAN_REPRODUCTION_REPORT.md`:

- fresh venv, no torch required
- 14/14 validation tests PASS
- doctor + canonical replay PASS

## Shared implementation

`procam_calibrate/charuco.py` owns board, dictionary, detection, matching,
homography, desired target, coverage, and acceptance metric helpers.

## Documentation

Production instructions no longer require CSPR for flat wall.
CSPR-Net / GS-ProCams remain research-only.

## Reproduce validation suite

```bash
cd procam-calibration
python -m pytest tests/test_validation_suite.py -v
```
