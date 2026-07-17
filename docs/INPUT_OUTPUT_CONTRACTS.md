# Input / Output Contracts

Concrete tensor shapes, image dimensions, coordinate conventions, normalization, and file formats for both repositories and the proposed MVP.

---

## Coordinate conventions (shared vocabulary)

| Convention | Definition |
|------------|------------|
| Pixel `(u, v)` | `u` = column (x, horizontal), `v` = row (y, vertical); origin top-left |
| OpenCV image | `ndarray` shape `(H, W, C)`, BGR when using `cv2.imread` |
| Torch image | Tensor `(B, C, H, W)`, RGB, float unless noted |
| `grid_sample` NDC | `(x, y)` in `[-1, 1]`, `align_corners=True` when used by CSPR |
| Projector space | Full native projector framebuffer, origin top-left |
| Camera space (2D) | Full camera image (after any documented crop) |

---

## CSPR-Net

### Resolutions

| Device | Height | Width | Symbol |
|--------|--------|-------|--------|
| Camera (native / neural) | 1728 | 3072 | `H_CAM, W_CAM` |
| Camera (real origin JPEG) | 2048 | 3072 | before crop |
| Camera (after crop) | 1728 | 3072 | crop `[160:1888, 0:3072]` |
| Projector | 1080 | 1920 | `H_PROJ, W_PROJ` |
| Train | 270 | 480 | `TRAIN_SCALE=0.25` of projector |

Evidence: `deep_learning_*_for_gradient.py` `Config`; exp `preprocess_and_crop`.

### Inputs

| Name | Shape / size | Range | Format | Path pattern |
|------|--------------|-------|--------|--------------|
| Projector pattern | `(1080, 1920, 3)` | uint8 0–255 | PNG RGB | `sim_data/{1,2,3}_pro.png`, `real_data/{1,2,3}_pro.png` |
| Sim camera view | `(1728, 3072, 3)` | uint8 | PNG | `sim_data/{1,2,3}_image_distorted.png` |
| Real camera origin | `(2048, 3072, 3)` | uint8 | JPEG | `real_data/{1,2,3}_image_distorted_origin.jpeg` |
| Red mask capture | same as origin | uint8 | JPEG | `real_data/red_distorted_origin.jpeg` |
| Custom content | arbitrary → resized to proj | — | JPEG/PNG | `real_data/test.jpg` default |
| Torch training batch | `(1, 3, 270, 480)` | float `[0, 1]` RGB | in-memory | from `load_image_tensor` |

### Outputs

| Name | Shape / size | Range | Format |
|------|--------------|-------|--------|
| `pre_warped.png` | `(1080, 1920, 3)` | uint8 | PNG |
| `net_c2p.pth` / `net_p2c.pth` | MLP state dict | — | PyTorch |
| Ray-trace maps (RAM only) | `(1728, 3072)` float32 u/v each | pixel coords or -1 invalid | not saved |
| Neural coord field | residual on NDC `[-1,1]` | float | via `CoordinateNet` |

### Camera / projector models (ray-trace only)

| | Camera | Projector |
|-|--------|-----------|
| Model | Pinhole K | Pinhole from throw ratio |
| fx, fy | ~3945.65 (from 10 mm, 1/1.8") | 2304 (= 1.2 × 1920) |
| cx, cy | (1536, 864) | (960, 540) |
| Pose | `CAM_C=[0,-7,1]` | `PROJ_C=[0,-4.5,2]` |
| World up | +Z | +Z |

Evidence: `qurdric_transfer.py:347–388`, `get_camera_matrices_physical`, `get_projector_matrices`.

### Neural warp application

- Base grid: `get_base_grid(h, w)` → NDC `[-1, 1]`
- Network: `coords_out = coords_in + delta`
- Sampling: `F.grid_sample(..., align_corners=True)`

---

## GS-ProCams

### Resolutions (defaults)

| Device | Default in `registrate.py` |
|--------|---------------------------|
| Camera | 800×800 (`--cam_width/height`) |
| Projector | 800×800 (`--prj_width/height`) |

Real datasets may use other resolutions as stored in COLMAP / images.

### Inputs

| Name | Shape | Range | Format | Notes |
|------|-------|-------|--------|-------|
| Pattern images | `(3, H_p, W_p)` after load | float `[0, 1]` | PNG under `patterns/` | Stem matches camera image |
| Camera train images | `(3, H_c, W_c)` | float `[0, 1]` | `img_1XXX.png` | |
| Mask | `(1, H_c, W_c)` or HxW | 0/1 | `mask/mask.png` | |
| COLMAP model | cameras.bin / images.bin / points3D | — | sparse/0 | PINHOLE or SIMPLE_PINHOLE |
| Nepmap `transforms.json` | JSON | — | includes `K_cam`, `K_proj` | Blender→OpenCV Y/Z flip |

### Outputs

| Name | Format | Notes |
|------|--------|-------|
| `point_cloud.ply` | PLY | Gaussians + PBR attrs |
| `procams.ckpt` | PyTorch | gamma, gain, PSF, optim state |
| `cameras.json` / `projector.json` | JSON | K, R, T, etc. |
| Compensated patterns | PNG `(H_p, W_p, 3)` | `patterns/{i:03d}.png` |
| Relit / sim camera | PNG | under render output |
| Warp grids | in-memory NDC for `grid_sample` | **not written to disk** |

### Coordinate systems

| System | Usage |
|--------|-------|
| COLMAP / OpenCV | World↔camera via `getWorld2View2` |
| Blender (Nepmap) | Y/Z flip to OpenCV |
| NDC warp | `uv * 2 / size - 1` for `grid_sample` |

### Compensation desired-path mismatch

| Source | Path |
|--------|------|
| README | `.../cam/desired/test` |
| `compensate.py` default | `.../cam/desire/test` (typo) |

Pass `--desired` explicitly when reproducing.

---

## Proposed MVP contract (new system)

Target: flat wall, one projector, one iPhone — classical structured light.

### Devices (configurable, not hardcoded)

| Parameter | MVP default suggestion |
|-----------|------------------------|
| Projector resolution | Native framebuffer (e.g. 1920×1080) — confirm on hardware |
| Camera resolution | iPhone still native or fixed crop — record EXIF / exact pixels used |
| Capture mode | Still photos (Files/USB import) — not wireless stream |

### Calibration sequence I/O

| Step | Output artifact |
|------|-----------------|
| Pattern set | `patterns/gray_{bit}_{pos|neg}.png` + `white.png` + `black.png` |
| Captures | `captures/{name}.jpg` aligned 1:1 with projected pattern |
| Correspondence | `maps/cam_to_proj_u.npy`, `cam_to_proj_v.npy` shape `(H_c, W_c)` float32; invalid = -1 |
| Inverse (optional) | `maps/proj_to_cam_u.npy`, `proj_to_cam_v.npy` shape `(H_p, W_p)` |
| Homography | `calib/H_cam_from_proj.npy` 3×3 float64 (or inverse as documented) |
| Valid mask | `maps/valid_mask.png` uint8 |

### Pre-warp I/O

| Name | Shape | Format |
|------|-------|--------|
| Desired camera appearance | `(H_c, W_c, 3)` or logical content rect | PNG |
| Projector pre-warp | `(H_p, W_p, 3)` | PNG — **display this** |
| Remap tables for content | `(H_p, W_p)` map_x/map_y | float32 `.npy` |

### Normalization

- Correspondence stored in **absolute pixel coordinates**
- Images for display: uint8 sRGB PNG
- Homography maps **homogeneous pixel coordinates** (same origin top-left)

### Explicit non-goals for MVP contract

- No BRDF / PEModel tensors
- No COLMAP sparse models required
- No neural `.pth` coordinate fields required
