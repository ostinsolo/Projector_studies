# Coordinate Systems

Every matrix, map, and warp must declare **source → destination**. Ambiguous names (`H`, `warp`, `transform`) are forbidden without these annotations.

---

## Primitive spaces

| ID | Description | Units | Origin | Axes |
|----|-------------|-------|--------|------|
| `ProjFB` | Projector framebuffer | pixels | top-left | +u right, +v down |
| `CamImg` | Camera image (after any *documented* crop) | pixels | top-left | +u right, +v down |
| `CamNDC` | Torch sampling grid for camera-sized tensors | normalized | center 0 | x∈[-1,1] ↔ u, y∈[-1,1] ↔ v |
| `ProjNDC` | Torch sampling grid for projector-sized tensors | normalized | center 0 | same |
| `World` | Shared 3D frame | metric or arbitrary | scene-dependent | right-handed; CSPR Z-up; GS COLMAP/OpenCV |
| `ViewCam` | Camera view / OpenCV camera | — | camera center | 3DGS/COLMAP convention |
| `ViewProj` | Projector as camera | — | projector center | same |
| `TexUV` | Desired content / texture | pixels or normalized | document per asset | — |

NDC conversion (CSPR `get_base_grid`, `align_corners=True`):

- pixel `u ∈ [0, W-1]` ↔ NDC `x = 2u/(W-1) - 1` (equivalent construction via `torch.linspace(-1,1,W)`).

GS uses `uv * 2 / size - 1` with `align_corners=False` — **not interchangeable** with CSPR without conversion.

---

## Named transforms — CSPR-Net

| Name in code | Direction | Type | Notes |
|--------------|-----------|------|-------|
| `cam_to_proj_map_u/v` | `CamImg → ProjFB` | dense float maps `(H_c,W_c)` | invalid = -1; not saved |
| `proj_to_cam_map_u/v` | `ProjFB → CamImg` | dense float maps `(H_p,W_p)` | invalid = -1 |
| `net_c2p` / `CoordinateNet` | `CamNDC → ProjNDC` | MLP residual | residual: out = in + delta |
| `net_p2c` | `ProjNDC → CamNDC` | MLP residual | |
| `pre_warped` via `grid_sample(..., c_coords_hd)` | samples **CamImg-layout content** at coords indexed by **ProjFB** | output ∈ `ProjFB` | **Cam content → ProjFB** |
| `fake_cam` via `grid_sample(t_proj, p_coords_hd)` | samples **ProjFB** at coords indexed by **CamImg** | output ∈ `CamImg` | **ProjFB → CamImg** |
| `K_cam`, `Rt_cam` (ray-trace) | `World → CamImg` (project) | 3×3 + 3×4 | pinhole |
| `K_proj`, `Rt_proj` | `World → ProjFB` | pinhole | |
| Ray intersect | `CamImg` ray → `World` on quadric | | then project to `ProjFB` |

### Homography (production flat-wall)

Authoritative production transforms in `INTEGRATION_ROOT`:

- `H_cam_from_proj`: **ProjFB homogeneous → CamImg homogeneous** (`x_c ~ H x_p`)  
- `H_proj_from_cam`: **CamImg → ProjFB** (`H_cam_from_proj^{-1}`)  
- `H_desired_cam_from_proj`: **ProjFB → desired CamImg AA rectangle** (independent target)  
- `desired_target.json`: per-ID CamImg points saved before corrected analysis

---

## Named transforms — GS-ProCams

| Name in code | Direction | Type | Notes |
|--------------|-----------|------|-------|
| `getWorld2View2(R,T)` | `World → ViewCam` or `ViewProj` | 4×4 | used for cameras and projector |
| `K` on camera/projector | `View* →` image pixels | 3×3 | PINHOLE |
| `warp_points(projector, surf_pts3d)` | `World → ProjFB` | per-point | |
| `prj2cam_grid` | For each `CamImg` pixel, NDC sample coord into projector pattern | `ProjNDC` sized to camera HxW | used to pull pattern into camera |
| Pattern `grid_sample` | `ProjFB → CamImg` irradiance | | then × BRDF |
| `cam2prj_grid` (compensater) | Supports init of compensation | toward `ProjFB` | see `procams_compensater.py` |
| Compensated pattern PNG | Lives in `ProjFB` | radiometric, not geometric LUT export | |
| Nepmap Blender fix | Blender c2w → OpenCV `World` | Y/Z flip | `dataset_readers.py` |

---

## MVP naming rules (`INTEGRATION_ROOT`)

Required artifact names:

| Artifact | Direction |
|----------|-----------|
| `cam_to_proj_u.npy` / `cam_to_proj_v.npy` | `CamImg → ProjFB` |
| `proj_to_cam_u.npy` / `proj_to_cam_v.npy` | `ProjFB → CamImg` |
| `confidence.npy` | per-`CamImg` pixel ∈ [0,1] |
| `valid_mask.png` | `CamImg` boolean |
| `H_cam_from_proj.npy` | `ProjFB → CamImg` (3×3) |
| `H_proj_from_cam.npy` | `CamImg → ProjFB` (3×3) |
| `remap_x.npy` / `remap_y.npy` for pre-warp | OpenCV remap maps for building `ProjFB` from content — document whether content is `CamImg` or `TexUV` |
| `pre_warped.png` | image in `ProjFB` |

Every function docstring must state direction in the form `source → destination`.

---

## Image orientation policy

| Rule | Enforcement |
|------|-------------|
| No silent mirroring | Fail if EXIF orientation applied inconsistently — record orientation in metadata |
| No silent rotation | Store device orientation; apply only with logged transform |
| No silent crop/resize | CSPR’s crop is repo-specific; MVP must not apply it to iPhone stills unless documented as a named stage |
| Distortion | If undistortion used: name maps `CamImg_distorted → CamImg_undistorted` |

---

## Cross-repo pitfalls

1. CSPR `align_corners=True` vs GS `False`.  
2. CSPR camera size 3072×1728 ≠ iPhone native.  
3. GS projector modeled as camera in COLMAP — `R,T` are view transforms, not “extrinsic only” without K.  
4. Compensation PNG is **not** a geometric `CamImg→ProjFB` LUT.
