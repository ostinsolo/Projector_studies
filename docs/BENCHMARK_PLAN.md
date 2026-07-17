# Benchmark Plan

Evaluate geometric projection correction for the three target surfaces in [AUDIT_SPEC.md](../AUDIT_SPEC.md): flat wall, angled planar surface, and cube / simple non-planar object.

---

## Test content

- Primary: high-contrast **rectangle** (axis-aligned in desired camera / logical frame) with known corner markers or a checkerboard insert.
- Secondary: thin grid lines for qualitative inspection.

Desired appearance: rectangle sides appear straight and corners form right angles **in the camera image** after projecting the pre-warp.

---

## Scenes

| ID | Scene | Geometry model | Pass bar |
|----|-------|----------------|----------|
| S1 | Flat wall | Single homography | Corners within threshold; sides straight |
| S2 | Angled planar board | Single homography (recalib) | Same as S1 |
| S3 | Cube / non-planar | Dense remap (no single H) | Per-face or overall corner error within looser threshold |

Each scene: new capture session (camera/projector fixed during that session).

---

## Metrics

### Geometric (primary)

| Metric | Definition |
|--------|------------|
| Corner RMSE | Pixel RMSE between detected rectangle corners and ideal rectangle in camera image |
| Max corner error | Worst-case corner distance (px) |
| Line straightness | Max deviation of each side from fitted line (px) |
| Aspect ratio error | `\|measured_aspect - target_aspect\| / target_aspect` |

### Correspondence (calibration health)

| Metric | Definition |
|--------|------------|
| Valid coverage | Fraction of projected region with valid decode |
| Decode consistency | Agreement between pos/neg bitplanes (or temporal consistency) |
| Homography inlier ratio | RANSAC inliers / samples (S1/S2) |

### Optional photometric (not MVP gate)

- PSNR/SSIM vs desired — only after geometry passes (CSPR/GS style metrics are secondary).

---

## Suggested thresholds (initial — tune after first session)

| Scene | Corner RMSE | Max corner | Straightness |
|-------|-------------|------------|--------------|
| S1 Flat | ≤ 3 px | ≤ 6 px | ≤ 2 px |
| S2 Angled | ≤ 4 px | ≤ 8 px | ≤ 3 px |
| S3 Cube | ≤ 8 px | ≤ 15 px | ≤ 5 px per visible edge |

Camera resolution scales thresholds (state resolution in report). Values above assume ~3K-wide stills; rescale linearly with width if different.

---

## Procedure per scene

1. Run structured-light calibration session ([IPHONE_CAPTURE_PLAN.md](IPHONE_CAPTURE_PLAN.md)).
2. Build maps / H (or dense LUT).
3. Generate `pre_warped_test.png` for the rectangle content.
4. Project pre-warp; capture verification still (same camera pose).
5. Detect corners (manual annotation acceptable for first runs; automate later).
6. Log metrics + save `verify/` images and `metrics.json`.

---

## Ablations

| Ablation | Purpose |
|----------|---------|
| Homography vs dense remap on S1 | Confirm H is enough on flat |
| Fewer Gray-code bits | Find minimum pattern count |
| Brightness / exposure variants | Robustness |
| Without black/white mask | Show mask necessity |

---

## Repo baselines (optional, not MVP gates)

| Baseline | What to measure |
|----------|-----------------|
| CSPR-Net bundled `pre_warped.png` | Qualitative only on **their** data — not iPhone MVP |
| GS-ProCams compensate | Radiometric metrics on **their** datasets if CUDA+data available |

Do not block MVP on reproducing GS-ProCams paper numbers.

---

## Report template

```text
scene: S1
camera_size: [W, H]
projector_size: [W, H]
method: graycode+homography
corner_rmse_px: ...
max_corner_px: ...
straightness_px: ...
inlier_ratio: ...
pass: true|false
notes: ...
```
