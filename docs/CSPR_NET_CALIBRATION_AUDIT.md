# CSPR-Net Calibration Audit

**Root:** `/Users/ostino/Documents/Code/Projector_studies/CSPR-Net` (`CSPR_NET_ROOT`)  
**Git commit:** `a2a0a28ac2022c876f1ac1e52620b44bfe96672a`  
**Scope:** Calibration / geometric correction / radiometry only. External learned models outside this repo are out of scope.

---

## 1. Calibration problem solved

**Geometric inverse projection rectification** for a fixed camera–projector–surface setup: learn bidirectional pixel maps so a projector pre-warp makes the **camera view** of a (typically curved) surface appear undistorted.

Evidence: `README.md` L3–9; outputs `pre_warped.png` via `NeuralWarper.save_hd_results` in `deep_learning_exp_for_gradient.py` / `deep_learning_simulation_for_gradient.py`.

---

## 2. Geometry, radiometry, or both?

| | Present? | Evidence |
|-|----------|----------|
| **Geometry** | Yes — primary | `CoordinateNet` residual maps; `pre_warped` via `F.grid_sample` |
| **Radiometry** | No explicit compensation | Losses are L1 + Sobel gradient + cycle + smooth + mask (`Config.W_*`); no gain/gamma map |

---

## 3. Camera calibration performed?

| Path | Answer |
|------|--------|
| Neural (`deep_learning_*`) | **No** — no K estimation |
| Ray-trace (`qurdric_transfer.py`) | **Uses** hardcoded physical params → builds K via `get_camera_matrices_physical` (L21–66, L363–373); does not estimate from images |

---

## 4. Projector calibration performed?

| Path | Answer |
|------|--------|
| Neural | **No** — no projector K estimation |
| Ray-trace | **Uses** throw ratio → `get_projector_matrices` (L78–105, L385–388); not estimated from captures |

---

## 5. Camera intrinsics must already be known?

| Path | Required? |
|------|-----------|
| Neural train/infer | **No** |
| Ray-trace GT maps | **Yes** (or derived: `f_mm=10.0`, sensor `"1/1.8"`) |

---

## 6. Projector intrinsics must already be known?

| Path | Required? |
|------|-----------|
| Neural | **No** |
| Ray-trace | **Yes** (`tr=1.2`, 1920×1080) |

---

## 7. Camera–projector extrinsics must already be known?

| Path | Required? |
|------|-----------|
| Neural | **No** (fixed unknown pose; learned from images) |
| Ray-trace | **Yes** (`CAM_C`, `PROJ_C`, `CAM_R`, `PROJ_R` at L365–391) |

---

## 8. Structured-light patterns required?

**Not classical Gray-code.** Uses **3 colorful grid patterns** (`generate_ununiform_mesh.py`) plus a **solid red** projection for real-data mask extraction (`real_data/4_proj.png`, `red_distorted_origin.jpeg`).

No binary/Gray structured-light decoder exists in the repository.

---

## 9. Number and type of required captures

| Item | Count | Type |
|------|-------|------|
| Train pairs | **3** | Camera photo of each grid + corresponding projector PNG |
| Mask (real) | **1** | Red projection capture |
| Inference content | **1+** | Arbitrary projector image (`inference_custom`) |

Evidence: `Config.TRAIN_PAIRS` in `deep_learning_exp_for_gradient.py:18–22`, `MASK_EXTRACT_PATH` L24.

---

## 10. Multiple camera viewpoints required?

**No.** Single fixed camera pose for all captures.

---

## 11. Training required?

**Optional.** Pretrained weights ship in `neural_results_simulation/` and `neural_results_exp/`. Training: `ITERS=6000` (`Config.ITERS`). New physical setup requires **retraining** (or fine-tune via `PRETRAINED_C2P/P2C`).

---

## 12. Pretrained checkpoints available?

**Yes (local):**

- `neural_results_simulation/net_c2p.pth`, `net_p2c.pth`
- `neural_results_exp/net_c2p.pth`, `net_p2c.pth`

Loaded by `NeuralWarper.load_models()`.

---

## 13. Exact inference / calibration entry points

| Command | Role |
|---------|------|
| `python generate_ununiform_mesh.py` | Pattern generation (no CLI args) |
| `python qurdric_transfer.py` | Ray-trace dense maps + GT warps (simulation calibration) |
| `python deep_learning_simulation_for_gradient.py train\|inference` | Sim neural train/infer; default `inference` (L508) |
| `python deep_learning_exp_for_gradient.py train\|inference\|inference_custom [path]` | Real neural; default `inference_custom` (L828) |

---

## 14. Required datasets, checkpoints, configuration

| Asset | Local? |
|-------|--------|
| `sim_data/`, `real_data/`, `data/` | Yes |
| `neural_results_*/net_*.pth` | Yes |
| External download | None required for bundled inference |
| Config | Hardcoded `class Config` in each script |

---

## 15. Exact image and tensor input formats

| Input | Shape | Dtype / range | Notes |
|-------|-------|---------------|-------|
| Projector pattern | H=1080, W=1920, C=3 | uint8 PNG | `{1,2,3}_pro.png` |
| Sim camera | H=1728, W=3072, C=3 | uint8 PNG | `*_image_distorted.png` |
| Real camera origin | H=2048, W=3072 | JPEG | Crop `[160:1888, 0:3072]` → 1728×3072 (`preprocess_and_crop` L179–188) |
| Torch image | `(1,3,H,W)` | float `[0,1]` RGB | `load_image_tensor` |
| Train size | `(1,3,270,480)` | float `[0,1]` | `TRAIN_SCALE=0.25` |

---

## 16. Exact output formats

| Output | Format | Meaning |
|--------|--------|---------|
| `pre_warped.png` | 1920×1080 PNG | Projector framebuffer image |
| `net_c2p.pth` / `net_p2c.pth` | PyTorch state dict | Implicit maps |
| `fake_cam.png`, masks, etc. | PNG | Diagnostics |
| Ray-trace maps | float32 HxW in RAM | **Not saved to disk** (`cam_to_proj_map_u/v` L399–415) |

---

## 17. Coordinate conventions

| Object | Convention | Direction |
|--------|------------|-----------|
| `net_c2p` | NDC `[-1,1]`, `align_corners=True` | **camera pixel NDC → projector pixel NDC** |
| `net_p2c` | same | **projector pixel NDC → camera pixel NDC** |
| Ray-trace `cam_to_proj_map_*` | absolute pixels | **camera pixel (u,v) → projector pixel (u,v)** |
| Ray-trace `proj_to_cam_map_*` | absolute pixels | **projector pixel → camera pixel** |
| `cv2.remap` in ray-trace | OpenCV maps | sized to destination |

`get_base_grid(h,w)` builds sampling grids for `F.grid_sample` (sim L184–186).

---

## 18. Image orientation conventions

- OpenCV load: BGR uint8; converted to RGB for tensors.
- Origin: top-left; `u` = column, `v` = row.
- No horizontal mirror applied in code.
- Real crop removes top/bottom bands only (`[160:1888, :]`).

---

## 19. Resolution assumptions

Hardcoded: camera **3072×1728** (after crop), projector **1920×1080** (`Config.H_CAM,W_CAM,H_PROJ,W_PROJ`). Changing hardware requires code edits.

---

## 20. Lens-distortion assumptions

- Neural: **none modeled** (absorbed into learned map if present in photos).
- Ray-trace: **ideal pinhole**, no distortion coefficients (`get_camera_matrices_physical`).

---

## 21. Projector-response assumptions

No PEModel / gamma / PSF. Photometric L1 assumes roughly consistent appearance between projector pattern and camera crop of the projection.

---

## 22. Dense camera–projector map generated?

| Form | Generated? | Persisted? |
|------|------------|------------|
| Ray-trace float maps | Yes | **No** (RAM only) |
| Neural field | Yes (implicit) | As `.pth` only |
| Exportable `.npy` LUT | **No** | — |

---

## 23. Projector-space warp generated?

**Yes.** `pre_warped = F.grid_sample(desired_cam_content, c_coords_hd)` where `c_coords_hd` is **projector→camera** sampling locations (`save_hd_results` exp L525–526). Output is a projector framebuffer PNG.

Direction: **desired appearance in camera space → pixels to display on projector**.

---

## 24. Closed-loop correction method?

**No.** Single offline train/infer. No residual capture→update→reproject loop.

---

## 25. Expected runtime and hardware

| Mode | Hardware | Notes |
|------|----------|-------|
| Inference | CPU or CUDA | `DEVICE` auto-select; exp chunks 200k coords |
| Train | CUDA preferred | 6000 iters at 480×270 |
| Ray-trace | CPU | Dense double map; PyVista viz needs display |

No official VRAM/time benchmarks in README. Inference prints elapsed seconds.

---

## 26. Missing files / undocumented dependencies

1. No `LICENSE`, no `requirements.txt` version pins  
2. README omits `generate_ununiform_mesh.py` run command and exp `inference_custom`  
3. Dense maps not saved from `qurdric_transfer.py`  
4. No paper link  

---

## 27. License constraints

**Unknown** — no license file in repository. Treat as all-rights-reserved until clarified. Do not copy into product without approval.

---

## 28. Independently reusable components

| Component | Symbol / file | Reuse for MVP |
|-----------|---------------|---------------|
| Pattern generator | `generate_ununiform_mesh.py` | Adapt ideas; not Gray-code |
| Ray-trace maps | `qurdric_transfer.py` | Simulation / validation |
| `CoordinateNet` + `NeuralWarper` | `deep_learning_*.py` | Optional curved later; **exclude from flat-wall MVP core** |
| Mask from red | `create_mask` (exp) | Adapt |
| Metrics SSIM/PSNR | `calculate_metrics` | Validation reference |

---

## Calibration path summary

```text
Patterns (3 grids [+ red])
  → project / or simulate (qurdric_transfer)
  → camera images
  → [optional train] CoordinateNet C↔P
  → inference: grid_sample → pre_warped.png
```

**Closest to MVP deliverable among the two repos:** produces projector-space geometric pre-warp.  
**Gap vs MVP:** no classical SL, no exportable LUT, hardcoded resolutions, retrain per setup, no closed loop.
