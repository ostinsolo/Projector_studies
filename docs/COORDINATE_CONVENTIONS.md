# Coordinate Conventions

**Module:** `procam_calibrate/coordinate_conventions.py`

## Spaces

| Name | Meaning |
|------|---------|
| `source` | Content layout = ProjFB resolution (pre-warp input) |
| `proj` | Projector framebuffer pixels |
| `cam` | Camera image pixels as captured |
| `desired` | Predetermined camera-space target |

Origin: top-left. Axes: +u right, +v down.

## Homography naming

`H_<dst>_from_<src>` means `p_dst ~ H @ p_src` (column vectors).

Examples used in production:

- `H_cam_from_proj_plane_A` / `_B`
- `H_desired_cam_from_source`
- `H_proj_from_source_plane_A` / `_B`

## OpenCV `warpPerspective`

`dst(p) = src(H^{-1} p)`. Passing `H_proj_from_source` places content point `s` at projector pixel `p = H_proj_from_source @ s`.

## Guards (tested)

- Forward composition: `H_cam_from_proj @ H_proj_from_source ≈ H_desired`
- Wrong-direction detection: `inv(H)` yields much larger reprojection error
- Mirror/orientation flags on corner order and `det(A)`
