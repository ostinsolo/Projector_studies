# ACCURACY_STATE

## Flat-wall MVP

**DONE / VALIDATED** (ChArUco + planar homography)

Immutable validation runs:

| Role | Run ID |
|------|--------|
| Canonical | `flatwall_20260717_141610` |
| Fresh live MVP | `flatwall_live_20260717_154312` |

## Absolute-accuracy enhancement

**COMPLETE** — absolute gate achieved on new physical runs.

| Gate | Threshold | Status |
|------|-----------|--------|
| Corrected target median | ≤ 5 camera px | **PASS** |
| Corrected target p95 | ≤ 12 camera px | **PASS** |
| Coverage | expected-visible | **PASS** |
| Geometry (axis / ortho) | relative + absolute criteria | **PASS** |

## Best physical results

### Fresh calibration (absolute pass without refine)

`procam-test-data/runs/flatwall_abs_20260717_165506`

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| target_median_err_px | 48.32 | **0.43** |
| target_p95_err_px | 104.11 | **0.81** |
| absolute_gate_pass | — | **true** |
| relative_gate_pass | — | **true** |

Forward matrix: `homography_baseline/H_proj_from_source_current.npy`

### Residual refinement (from live relative-pass baseline)

`procam-test-data/runs/flatwall_abs_refine_20260717_165533`

| Stage | median | p95 | note |
|-------|--------|-----|------|
| Baseline (copied live corrected) | 20.76 | 24.61 | residual class: **translation** |
| Refine iter 1, strength 1.0 | **0.34** | **0.64** | accepted; jitter median 0.29 px |
| absolute_gate_pass | | | **true** |

Best matrix: `homography_refinement/H_proj_from_source_best.npy`

## Metric consistency (Phase 1)

Recalculated under `analysis_remeasure_v2/` (original `metrics.json` untouched):

| Run | Corrected median / p95 (v2) | absolute | relative |
|-----|----------------------------|----------|----------|
| `flatwall_20260717_141610` | 18.24 / 21.47 | false | true |
| `flatwall_live_20260717_154312` | 20.76 / 24.61 | false | true |

Dominant residual on live corrected capture: **translation**
(median dx ≈ −1.7 px, median dy ≈ +19.5 px; after translation ≈ 4.7 px remains).
