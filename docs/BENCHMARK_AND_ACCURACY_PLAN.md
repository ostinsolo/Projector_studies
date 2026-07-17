# Benchmark and Accuracy Plan

Numerical evaluation for calibration and geometric correction. Visual inspection is never sufficient alone.

---

## Required metrics (every calibration result)

Report in `metrics.json` and human `report.md`.

### Primary (flat-wall production — independent fiducials)

Known projector ChArUco corner coordinates ↔ detected camera corners
(`procam_calibrate/charuco.py`). Contour / AA-box metrics are **secondary only**.

| Metric | Unit | Role |
|--------|------|------|
| expected / detected marker count | count | detection health |
| interpolated ChArUco corner count | count | detection health |
| matched projector/camera point count | count | correspondence |
| RANSAC inliers / rejected points + reasons | count | H fit |
| image resolution | px | context |
| board definition + dictionary id | — | reproducibility |
| preprocessing parameters | — | reproducibility |
| target-point median / p95 / max vs saved desired target | camera px | **primary (B)** |
| horizontal / vertical axis deviation | degrees | **primary (B)** |
| orthogonality / parallelism / aspect errors | deg / ratio | **primary (B)** |
| H-refit residual / RANSAC inliers | camera px | **fit health (A)** only |
| AA / contour hull corner error | camera px | **secondary visual (C)** only |

### Additional / dense-path metrics (when applicable)

| Metric | Unit |
|--------|------|
| number of projected patterns | count |
| number of captured frames | count |
| correspondence-validity percentage | % |
| dense-map valid coverage | % |
| calibration / capture / warp / validation runtime | s |
| correction-loop iterations | count |
| total wall-clock time | s |
| peak memory | MB / GB |

---

## Milestone gates

### M4 — Sparse planar calibration (flat-wall production — VALIDATED)

| Criterion | Threshold |
|-----------|-----------|
| Automatic ChArUco correspondences | no manual points |
| Desired target saved before corrected analysis | required |
| Target median + p95 improve vs uncorrected | required |
| Axis / orthogonality improve | required |
| Coverage vs expected-visible | ≥ 80% or documented mask |
| Fit-health residuals (category A) | reported, not sole acceptance |

Method: project ChArUco → detect (shared module) → estimate `H_cam_from_proj` →
define desired target → pre-warp → project corrected board → recapture →
measure vs desired target.

Canonical: `flatwall_20260717_141610` (target median 75.6→18.2; axis 12.7°→0.03°).  
Live: `flatwall_live_20260717_154312` (target median 66.9→20.8; axis 11.3°→0.025°).

### M5 — Dense correspondence

Exit: valid region + confidence thresholds documented and reproducible; lossless maps stored (`cam_to_proj_*`, confidence, mask). Optional for flat wall (homography MVP does not require dense maps).

### M6 — Pre-warp

Exit: corrected projection **numerically more accurate** than uncorrected on the same validation capture using **independent fiducial metrics** (not contour AA). **Met** for flat-wall homography MVP.

### M7 — Residual loop

Homography MVP: optional ROI recapture if corrected markers are clipped; no CSPR-style iterative refine required. Exit: acceptance gate pass or documented physical limit; best H + pre-warp retained.

---

## Scenes (M9 order)

1. Flat wall  
2. Angled planar  
3. Two connected planes  
4. Cube / piecewise planar  
5. Curved — only if audited method supports it (CSPR neural path is the audited curved method)

---

## Comparison baselines

| Baseline | When |
|----------|------|
| Uncorrected projection | Always for M6+ |
| Previous integration revision | Every change |
| CSPR bundled `pre_warped` on **their** data | After M1 (repo baseline, not iPhone) |
| GS compensate on **their** data | After M2 if CUDA+data available |

Integrations (M8) only if accuracy/robustness/performance improves vs baseline on identical test data.

---

## Failure handling

If accuracy fails or regresses:

1. Stop feature work  
2. Find first divergent stage  
3. Preserve failing artifacts  
4. Fix only that stage  
5. Rerun focused test  

---

## Manual correspondences

Allowed only as **temporary diagnostic** baseline and must be labelled `manual_diagnostic=true` in `metrics.json`. Must not be claimed as automatic calibration.
