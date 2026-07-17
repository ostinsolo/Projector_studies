# Residual Model Report

Diagnosis performed **before** applying residual correction, on the immutable
live MVP corrected capture.

Source run: `procam-test-data/runs/flatwall_live_20260717_154312`  
Artifacts: `analysis_remeasure_v2/residual_diagnosis/`

## Evaluation set

- `evaluation_ids`: 29 IDs (detected ∩ projector ∩ desired ∩ expected-visible)
- `detected_but_not_expected_ids`: [12, 13]
- `expected_but_not_detected_ids`: []

## Residual summary (detected − desired)

| Quantity | Value (camera px) |
|----------|-------------------|
| median magnitude | 20.76 |
| p95 magnitude | 24.61 |
| median dx | −1.71 |
| median dy | **+19.55** |
| p95 \|dx\| | 8.26 |
| p95 \|dy\| | 23.88 |
| mean translation | (−1.17, +20.00) |

Distribution is narrow (median ≈ p95), consistent with a near-constant offset.

## Model fits (remaining median magnitude)

| Model | Remaining median px |
|-------|---------------------|
| none (raw) | 20.76 |
| translation | **4.69** |
| similarity | 0.33 |
| affine | 0.34 |
| homography | 0.34 |

## Classification

**Dominant residual class: `translation`**

- `constant_translation_hypothesis`: **true**
- Translation removes most of the ~20 px error; similarity/homography then
  reduce the remainder to ~0.3 px (near measurement noise).

## Display / capture inspection (context)

| Item | Observation |
|------|-------------|
| Projector framebuffer | 1920×1080, scale **1.0** (USB Display: MStar Demo) |
| macOS main display | 2056×1329 logical @ scale 2.0 (not used for ProjFB) |
| Fullscreen placement | external extended display (origin_x = −1920) |
| Overscan | not indicated by scale=1.0 framebuffer match |
| Capture resolution | 1920×1440 (iPhone Continuity) |
| Stale-frame mitigation | flush_n=20 before captures; refine uses burst≥3 |
| Pre-warp convention | `H_proj_from_source = inv(H_cam_from_proj) @ H_desired`; OpenCV `warpPerspective` receives this forward matrix |

Likely physical/system cause of the live ~20 px Y bias: residual planar
misalignment (constant image shift) between the saved desired target and the
corrected projection, not lens distortion or random detection noise.

## Spatial grid (3×3 mean magnitude on desired positions)

Upper rows slightly larger (~24 px) than lower (~17 px); still dominated by the
global Y translation rather than a strong projective fan-out.
