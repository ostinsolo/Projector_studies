# Metric Definitions — Flat-Wall Production

Three categories. Acceptance uses **B** only (with **A** as health gates).
**C** is diagnostic.

---

## A. Calibration-fit health

May use a homography refitted to detected correspondences.

| Metric | Definition |
|--------|------------|
| `n_markers_detected` | ArUco markers found |
| `n_charuco_corners_interpolated` | ChArUco corners from detector |
| `n_matched` | Shared projector/camera corner IDs |
| `n_inliers` / `n_outliers` | RANSAC mask on calib fit |
| `median_fit_residual_px` | Median ‖Ĥ·p − c‖ on inliers |
| `p95_fit_residual_px` | 95th percentile of same |
| `max_fit_residual_px` | Max of same |

**Rule:** Fit residuals prove correspondence + planar model quality.
They **do not** by themselves prove keystone correction.

Empty detection → `valid: false`, numeric metrics `null`, explicit `failure_reason`.
Never report zero error for empty measurements.

---

## B. Production correction metrics (acceptance)

Compare detected CamImg points to an **independently predetermined desired target**.

### Desired target definition

Saved **before** analysing the corrected capture as `desired_target.json`:

1. From calibration `H_cam_from_proj` and projector framebuffer corners, compute
   the axis-aligned destination rectangle that the pre-warp aims for
   (bounding rectangle of `H_cam_from_proj` applied to ProjFB corners).
2. Build `H_desired_cam_from_proj` = perspective map from ProjFB rectangle → that
   AA rectangle in CamImg.
3. For each known projector ChArUco ID with ProjFB coordinate `p_i`,
   `desired_cam[i] = H_desired_cam_from_proj · p_i`.

The desired target is **not** derived from the corrected capture contour.

### Metrics vs desired target

| Metric | Definition |
|--------|------------|
| `target_median_err_px` | median ‖c_i − desired_cam[i]‖ over matched IDs |
| `target_p95_err_px` | 95th percentile of same |
| `target_max_err_px` | maximum of same |
| `horizontal_axis_deviation_deg` | median |angle| of board rows vs image horizontal |
| `vertical_axis_deviation_deg` | median |angle−90°| of board cols vs image vertical |
| `orthogonality_error_deg` | |90 − angle(mean row direction, mean col direction)| |
| `opposite_edge_parallelism_deg` | mean absolute angle difference of opposite outer edges |
| `aspect_ratio_error` | |measured_aspect − desired_aspect| / desired_aspect |
| `corner_position_err_px` | mean error of four outer board corners vs desired |
| `grid_intersection_err_px` | alias of mean target error over all matched IDs |

### Acceptance (production)

Both uncorrected and corrected measurements must be `valid: true`.

Corrected must improve vs uncorrected:

- `target_median_err_px`
- `target_p95_err_px`
- axis / orthogonality geometry (`horizontal_axis_deviation_deg` and/or
  `orthogonality_error_deg`)

Absolute soft targets (camera px) when optics allow:

- corrected `target_median_err_px` ≤ 5
- corrected `target_p95_err_px` ≤ 12

---

## C. Secondary visual metrics

| Metric | Role |
|--------|------|
| AA hull / contour corner error | visualization only |
| Contour-derived axis-aligned box residual | visualization only |

Must **never** be the sole acceptance criterion.

---

## Coverage (Gate 5)

| Metric | Definition |
|--------|------------|
| `expected_visible_corners` | corners expected in valid ROI / FOV |
| `detected_corners` | interpolated ChArUco count |
| `matched_corners` | IDs matched to projector map |
| `coverage_fraction` | matched / expected_visible |
| `quadrant_counts` | matched points per image quadrant |
| `hull_area_fraction` | convex hull area / image area |
| `min_boundary_distance_px` | min distance of detections to image border |

Gate: `coverage_fraction ≥ 0.80` **or** a saved `expected_visible_mask` documenting
why fewer markers are valid (e.g. intentional ROI clip).
Spatial spread must not be a central cluster alone (all four quadrants of the
projected board region should be represented when expected).
