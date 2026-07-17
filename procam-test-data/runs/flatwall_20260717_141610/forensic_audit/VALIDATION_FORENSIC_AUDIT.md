# Validation forensic audit — flatwall_20260717_141610

Status: **ACCEPTANCE_FAILED** (not BLOCKED_EXTERNAL). CSPR train artifacts preserved; no retrain.

## First invalid result

**Stage: MEASURE_RESIDUAL**

The pipeline produced real images through GENERATE_PREWARP → PROJECT_VALIDATION → CAPTURE_VALIDATION. The first *invalid scientific result* appears when `rectangle_metrics()` compared detected outer-contour corners to an axis-aligned ideal built from the **same** corners’ centroid and mean width/height. Any axis-aligned detection therefore yields residual ≈ **0.0 by construction**. Both uncorrected and corrected overlays used such AA boxes, so metrics reported perfect residuals with an empty improvement block.

This is a software metric bug, not proof of calibration success.

## Stage-by-stage

### 1. GENERATE_PREWARP

| Field | Value |
|-------|--------|
| Input | `calibration/cspr_work` checkpoints + `projected_validation/content_validation.png` (1920×1080 validation target) |
| Output | `prewarps/prewarp_best.png`, `projected_validation/prewarp_to_project.png` |
| Dimensions | 1920×1080×3 uint8 |
| Coord system | ProjFB |
| Features | N/A (image synthesis) |
| Materially different from identity? | **Yes** — mean abs diff vs content ≈ 12.7; Farneback median displacement ≈ 23.6 px; ~85% pixels >2 px |
| Failure | None for file generation |

### 2. PROJECT_VALIDATION

| Field | Value |
|-------|--------|
| Input | `prewarp_to_project.png` (sha matches `prewarp_best.png`) |
| Output | Fullscreen display on USB Display MStar Demo 1920×1080 |
| Failure | None proven; display path rebound successfully in logs |

### 3. CAPTURE_VALIDATION

| Field | Value |
|-------|--------|
| Output | `captured_validation/validation_corrected.png` 1920×1440 |
| Capture meta | blur≈40.9, contrast≈50.5, mean≈77 |
| Timestamp | ~15:12 (mtime) vs uncorrected ~14:51 — **Δ ≈ 1395 s** |
| vs uncorrected | **Not identical** — SSIM≈0.078, mean abs diff≈67, max=255, hashes differ |

Stale-frame / duplicate-capture hypothesis for unc≡cor: **rejected**.

### 4. MEASURE_RESIDUAL — FIRST INVALID RESULT

| Field | Uncorrected | Corrected |
|-------|-------------|-----------|
| Input | `09_validation_uncorrected.png` | `validation_corrected.png` |
| Outer corners detected | Small AA box (1686–1919, 677–1207), area≈1.2e5 | Large AA box (159–1919, 0–1021), area≈1.8e6 |
| Chessboard/ChArUco | **0 detections** | **0 detections** |
| Reported residuals | 0 / 0 / 0 | 0 / 0 / 0 |
| Aspect | 0.44 | 1.72 |
| Valid scientific measurement? | **No** | **No** |

Failure reason: tautological AA residual + wrong contour (not validation fiducials). Pattern is a line grid, not a chessboard — detector found ambient bright regions.

### 5. REFINE_PREWARP

Eight refine iterations ran on the same broken metric (median 0). No trustworthy before/after improvement was possible. Refine should not have looped on invalid measurements (now short-circuits to FINAL when `valid:false`).

## Pre-warp mapping notes

- Size exactly **1920×1080** ProjFB.
- `prewarp_best` ≡ `prewarp_to_project` (same hash).
- Non-identity warp present (displacement viz in this folder).
- Direction for CSPR path follows repo convention: content sampled into ProjFB (`pre_warped`). Forensic does **not** claim CSPR inverse was double-applied; metric failure precedes that claim.

## Captures physical checks (summary)

1. Both files exist; timestamps differ by ~23 min.  
2. Visual/content difference large (SSIM 0.08) — not a copy.  
3. Corrected capture contrast/blur consistent with a live Continuity frame.  
4. Projected-region mask from black/white covers a large FOV bbox.  
5. Validation pattern type (line grid) was not auto-detectable as chessboard — measurement stage was the weak link.

## Remediation applied

1. Metrics: empty/failed fiducial detection → `valid:false`, null residuals, no improvement %.  
2. Acceptance state: `ACCEPTANCE_FAILED` instead of `BLOCKED_EXTERNAL`.  
3. Classical ChArUco + homography baseline command: `procam-calibrate flatwall-homography`.  
4. GS-ProCams feasibility: `docs/GS_PROCAMS_MVP_FEASIBILITY.md` → **REUSABLE_COMPONENTS_ONLY**.

## Homography baseline (automatic) — Case A

| Metric | Before | After |
|--------|--------|-------|
| ChArUco matched corners | 49 | 27 |
| RANSAC inliers (fit) | 49 | — |
| Median reprojection (CamImg px) | 0.54 | 0.29 (refit) |
| AA hull median corner error (px) | **79.17** | **0.14** |
| Homography condition number | 6.7e5 | — |

**Case A:** planar homography corrects the oblique flat-wall keystone. Accept as flat-wall MVP. Keep CSPR as experimental; its prior “0 px” result was an invalid metric, not a proven geometric win.

Why CSPR looked “perfect”: `MEASURE_RESIDUAL` never measured the validation grid. Captures were real and different; the detector locked onto bright AA blobs and scored them as zero-error.
