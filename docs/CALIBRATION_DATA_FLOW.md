# Calibration Data Flow

End-to-end flows for CSPR-Net, GS-ProCams, and the planned MVP. Every stage lists type, dimensions, dtype, range, spaces, and transforms.

Legend for spaces:

- `ProjFB` — projector framebuffer pixels `(u_p, v_p)`, origin top-left  
- `CamImg` — camera image pixels `(u_c, v_c)`, origin top-left  
- `CamNDC` / `ProjNDC` — PyTorch `grid_sample` coords in `[-1, 1]`  
- `World` — metric/arbitrary world frame  
- `ViewCam` / `ViewProj` — OpenCV/3DGS camera / projector view space  

---

## Shared pipeline sketch

```text
projector image
→ physical projection
→ camera capture
→ preprocessing
→ correspondence estimation
→ calibration parameters
→ warp generation
→ corrected projector image
→ camera validation
```

---

## A. CSPR-Net — neural experimental path

### A1. Projector image

| Field | Value |
|-------|-------|
| File | PNG `real_data/{1,2,3}_pro.png` |
| Dims | `(1080, 1920, 3)` HWC |
| Dtype / range | uint8 0–255 |
| Space | `ProjFB` |
| Transform | None |

### A2. Physical projection

Outside software. Surface: real setup (repo used curved).

### A3. Camera capture

| Field | Value |
|-------|-------|
| File | JPEG `*_image_distorted_origin.jpeg` |
| Dims | `(2048, 3072, 3)` |
| Dtype / range | uint8 |
| Space | `CamImg` (full sensor crop of their camera) |

### A4. Preprocessing

| Op | Detail |
|----|--------|
| Crop | rows `[160:1888]`, cols `[0:3072]` → `(1728, 3072, 3)` |
| Color | BGR→RGB for torch |
| Resize (train) | to `(270, 480)` |
| Norm | `/255` → float `[0,1]` |
| Mask | red channel > 120 from `red_distorted_origin.jpeg` same crop |
| Interpolation | OpenCV default on resize |
| Invalid | mask = 0 |

Evidence: `preprocess_and_crop`, `create_mask`, `Config` in `deep_learning_exp_for_gradient.py`.

### A5. Correspondence estimation

| Field | Value |
|-------|-------|
| Representation | MLP `net_c2p`, `net_p2c` |
| Train tensor | coords `(N,2)` in `CamNDC` / `ProjNDC` |
| Direction | `CamNDC → ProjNDC` and `ProjNDC → CamNDC` |
| Invalid | outside mask / poor cycle |

### A6. Calibration parameters

No explicit K/R/T. Parameters = network weights `.pth`.

### A7. Warp generation

| Field | Value |
|-------|-------|
| Op | `pre_warped = grid_sample(desired_in_cam_layout, c_coords_hd)` |
| `c_coords_hd` | `(1, H_p, W_p, 2)` NDC — **for each ProjFB pixel, sample location in CamImg content** |
| Direction | **CamImg content → ProjFB** |
| Interp | bilinear `grid_sample`, `align_corners=True` |
| Output | `pre_warped.png` `(1080,1920,3)` uint8 |

### A8. Validation

Project `pre_warped.png`; optional SSIM/PSNR vs desired camera appearance (`calculate_metrics`). No automatic residual loop.

---

## B. CSPR-Net — ray-trace simulation path

### B1–B2. Projector / surface

Synthetic cylinder quadric; patterns from `sim_data/{1,2,3}_pro.png`.

### B3–B4. “Capture”

Generated: `cam_to_proj_map_u/v` shape `(1728, 3072)` float32; invalid = `-1.0`.  
Direction: **CamImg → ProjFB**.  
Inverse maps `(1080, 1920)`: **ProjFB → CamImg**.

### B5–B7. Warp via `cv2.remap`

| Field | Value |
|-------|-------|
| Maps | absolute pixel maps sized to destination |
| Interp | `cv2.INTER_LINEAR` (typical remap) |
| Outputs | `data/*_image_prewarped.png` etc. |
| Persist maps | **No** |

---

## C. GS-ProCams — train / compensate path

### C1. Projector image

| Field | Value |
|-------|-------|
| File | PNG under `patterns/{train,test,calib,ref}/` |
| Tensor | `(3, H_p, W_p)` float `[0,1]` |
| Space | `ProjFB` |

### C2–C3. Physical projection + multi-view capture

Directory: `setups/<s>/views/<NN>/cam/raw/{calib,ref,train,test}/`.  
Naming: `img_1XXX.png`. Default registrate size 800×800.

### C4. Preprocessing

| Op | Detail |
|----|--------|
| Mask | white−black difference → `mask/mask.png` |
| Load | float `[0,1]`, 3-channel |
| COLMAP | poses + K; PINHOLE only |
| Resize | optional `--resolution` in train |
| Silent mirror | none documented |

### C5. Correspondence (implicit)

| Field | Value |
|-------|-------|
| Path | depth → `surf_pts3d` → `warp_points(projector, …)` → `prj2cam_grid` |
| Direction | for each **CamImg** pixel: sample from **ProjFB** (via surface) |
| NDC | `prj_pts2d * 2 / prj_size - 1`, `align_corners=False` |
| Persist | **No** |

### C6. Calibration parameters

| Param | Source | File |
|-------|--------|------|
| Camera K, R, T | COLMAP / JSON | `cameras.json` |
| Projector K, R, T | COLMAP calib view | `projector.json` |
| Scene geometry | trained Gaussians | `point_cloud.ply` |
| Radiometry | `PEModel` | `procams.ckpt` |

### C7. Warp / compensation output

| Mode | Output | Direction |
|------|--------|-----------|
| Render | simulated CamImg | ProjFB (+BRDF) → CamImg |
| Compensate | optimized ProjFB PNG | Desired CamImg → ProjFB (radiometric opt) |

### C8. Validation

`evaluate.py` / logs: PSNR, SSIM, LPIPS vs GT camera images.

---

## D. Planned MVP (INTEGRATION_ROOT) — flat wall

Not implemented in Phase 0. Contract:

| Stage | Type | Dims | Dtype | Range | Src space | Dst space | Notes |
|-------|------|------|-------|-------|-----------|-----------|-------|
| Pattern | PNG | `(H_p,W_p,3)` | uint8 | 0–255 | ProjFB | physical | Gray-code + black/white; lossless |
| Capture | still | `(H_c,W_c,3)` | uint8 | as shot | CamImg | — | **no silent resize/crop/mirror** |
| Preprocess | copy | same | same | — | CamImg | CamImg | metadata only unless documented |
| Correspondence | `.npy` maps | `(H_c,W_c)` | float32 | pixel or -1 | CamImg | ProjFB | + confidence, validity |
| Calibration | `H_cam_from_proj` 3×3 | — | float64 | — | ProjFB homog → CamImg homog | document inverse separately |
| Warp maps | `map_x,map_y` | `(H_p,W_p)` | float32 | | ProjFB ← CamImg content | for `cv2.remap` |
| Corrected | PNG | `(H_p,W_p,3)` | uint8 | | ProjFB | display |
| Validation capture | still | `(H_c,W_c,3)` | uint8 | | CamImg | metrics |

Invalid pixels: `-1` in maps; `0` in validity mask.

---

## Interpolation summary

| System | Method | align_corners |
|--------|--------|---------------|
| CSPR neural | `F.grid_sample` bilinear | `True` |
| GS render | `F.grid_sample` bilinear | `False` |
| CSPR ray-trace | `cv2.remap` | N/A (pixel maps) |
| MVP (planned) | `cv2.remap` / `warpPerspective` | document per call |
