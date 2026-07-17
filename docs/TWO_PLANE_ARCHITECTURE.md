# Two-Plane Architecture

Separate from the validated single-plane flat-wall production path.

## Goal

Automatically correct one projected image spanning two connected planar surfaces
(e.g. two walls meeting at a corner).

## Pipeline

```
project ChArUco spanning both planes
→ detect projector/camera correspondences
→ test one-homography sufficiency
→ if insufficient: multi-model RANSAC → two H
→ reassign / refit until stable
→ estimate straight seam in ProjFB
→ exclusive plane masks
→ per-plane forward pre-warps
→ composite 1920×1080 framebuffer
→ (physical later) project / capture / validate per plane + seam
```

## Modules

| Module | Role |
|--------|------|
| `procam_calibrate/two_plane.py` | fit, seam, masks, piecewise compose |
| `procam_calibrate/synthetic_two_plane.py` | synthetic scenes + suite |
| CLI `auto-two-plane` | software entry (synthetic-only by default) |

## Matrices

- `H_cam_from_proj_plane_A` / `_B`
- `H_proj_from_source_plane_A` / `_B` (OpenCV forward point maps)

## Non-goals (this phase)

- CSPR / GS-ProCams
- Soft alpha blending as a substitute for geometry
- Modifying flat-wall production state or immutable runs
