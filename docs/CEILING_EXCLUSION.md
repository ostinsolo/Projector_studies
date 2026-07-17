# Ceiling Exclusion

## Role of the ceiling

In this project the ceiling is a **forbidden projection region**, not a third artistic surface.

Calibration patterns may temporarily illuminate the ceiling so geometry can be measured. Final video content must not.

## Classification

`procam_calibrate.excluded_geometry.classify_active_and_excluded_planes`:

1. Peel candidate ceiling points (upper projector band and/or upper camera band).
2. Fit wall A and wall B homographies on the remaining points.
3. Label ceiling as excluded (`label == 2`).
4. Never promote the ceiling to a production content plane.

Assignment rule of thumb:

| Cluster | Role |
|---------|------|
| Largest valid connected wall | Active wall A |
| Second valid connected wall | Active wall B |
| Upper spatial planar cluster | Candidate ceiling (excluded) |
| Unsupported / noisy | Outliers |

If two active walls cannot be formed after peeling the ceiling, the pipeline stops with a precise diagnostic instead of contaminating the wall solution.

## Wall–ceiling boundary

`estimate_wall_ceiling_boundary` prefers the interface between wall and ceiling correspondences. Architectural horizontals are priors only. A line is **not** classified as the ceiling boundary solely because it is horizontal (false shadow rejection).

## Masks

`build_usable_and_exclusion_masks` writes:

- `usable_mask_camera.png` / `usable_mask_projector.png`
- `excluded_ceiling_mask_camera.png` / `excluded_ceiling_mask_projector.png`
- `wall_ceiling_boundary.json`
- `confidence_map_camera.png`
- `plane_classification.json`

Usable = projector-reachable ∩ active walls ∩ below ceiling boundary ∩ confidence / safety margins.

## Production model

Content remains **two-plane**. Ceiling evidence improves the exclusion mask only.
