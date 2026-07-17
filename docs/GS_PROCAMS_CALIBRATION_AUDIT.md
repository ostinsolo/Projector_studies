# GS-ProCams Calibration Audit

**Root:** `/Users/ostino/Documents/Code/Projector_studies/GS-ProCams` (`GS_PROCAMS_ROOT`)  
**Git commit:** `dfb7f449710ce41a63369a447ca728e0cb220d97`  
**Paper:** Deng et al., “GS-ProCams…”, IEEE TVCG/ISMAR 2025, DOI 10.1109/TVCG.2025.3616841 (`README.md` citation)  
**Scope:** Calibration / geometric correction / radiometric compensation present in this repository only.

---

## 1. Calibration problem solved

Joint **projector–camera system modeling** with 2D Gaussian surfels:

1. Estimate/use camera and projector poses+intrinsics (via COLMAP or dataset JSON).  
2. Train scene BRDF + projector radiometry (`PEModel`).  
3. **Simulate** camera images under novel projector patterns.  
4. **Radiometrically compensate** projector patterns so the camera matches a desired appearance.

Evidence: `README.md` L1–5; `train.py`; `gaussian_renderer/procams_render.py`; `compensate.py`.

This is **not** primarily a classical geometric pre-warp / homography calibrator.

---

## 2. Geometry, radiometry, or both?

| | Present? | Evidence |
|-|----------|----------|
| **Geometry** | Yes — scene Gaussians + pinhole ProCam poses; internal warp grids | `GaussianModel`; `prj2cam_grid` in `procams_render.py:264–267` |
| **Radiometry** | Yes — primary compensation product | `PEModel` (`scene/projector.py:82–109`); `Compensater.compensate` |

---

## 3. Camera calibration performed?

**Indirectly via COLMAP** in `scripts/registrate.py` (feature matching, mapper, bundle adjustment with principal-point refine). Not a custom checkerboard intrinsic solver inside the Python train loop.

Nepmap synthetic: intrinsics **provided** in `transforms.json` (`scene/dataset_readers.py`).

---

## 4. Projector calibration performed?

**Indirectly:** projector treated as an extra COLMAP camera (`calib` view / `--calib_name`). Intrinsics/extrinsics come from that COLMAP entry (`dataset_readers.py` projector load ~L260–272). Radiometric params learned in `PEModel` during `train.py`.

---

## 5. Camera intrinsics must already be known?

| Path | Required a priori? |
|------|-------------------|
| Real-world + `registrate.py` | **No** — COLMAP estimates |
| Nepmap | **Yes** — in `transforms.json` |

---

## 6. Projector intrinsics must already be known?

| Path | Required a priori? |
|------|-------------------|
| Real-world + COLMAP | **No** |
| Nepmap | **Yes** — `K_proj` |

---

## 7. Camera–projector extrinsics must already be known?

**No** for real-world (joint COLMAP world frame).  
**Yes** for Nepmap (Blender matrices in `transforms.json`).

---

## 8. Structured-light patterns required?

**Not Gray-code decoding.** Uses:

- Shared **calibration texture** for COLMAP (`patterns/calib/`, e.g. `calib.png`)  
- Many **RGB training/test patterns** under `patterns/train|test|ref/`  
- Black/white **ref** images for mask (`registrate.py` mask from white−black)

No structured-light bitplane decoder in the repo.

---

## 9. Number and type of required captures

Defaults (`arguments/__init__.py`, `dataset_readers.py`):

| Item | Typical default |
|------|-----------------|
| Viewpoints | **25** train (+ novel 26–33 for eval) |
| Pattern pool | **100** train patterns (~4 per view + ref) |
| Calib captures | **1 per view** + projector calib image |
| Compensation desired | Images under `cam/desired/test` (README) |

---

## 10. Multiple camera viewpoints required?

**Yes for the published real-world pipeline** (default 25). Novel-view registration supports adding one view (`registrate.py --view_id`). Single-view-only operation is not the documented path.

---

## 11. Training required?

**Yes.** No pretrained GS-ProCams scene checkpoints in the clone. Default **20,000** iterations (`OptimizationParams.iterations`).

---

## 12. Pretrained checkpoints available?

**No** in this clone (`data/`, `output/` absent / gitignored). Must train or obtain externally. LPIPS VGG weights download at eval time (`lpipsPyTorch/`).

---

## 13. Exact inference / calibration entry points

| Entry | Role |
|-------|------|
| `python scripts/registrate.py -r … -s …` | COLMAP calibration + masks |
| `python train.py -r … -s … -m …` | Joint geometry + radiometry train |
| `python render.py -r … -s … -m … -o …` | Simulate / relight |
| `python compensate.py -r … -s … -m … --view_id …` | Radiometric compensation |
| `python evaluate.py …` | Metrics batch |
| `bash scripts/{synthetic,real-world,compensate}.sh` | Paper pipelines (path caveats) |

Install: `README.md` L17–28.

---

## 14. Required datasets, checkpoints, configuration

| Asset | Local status |
|-------|--------------|
| SharePoint real-world / compensation | **Missing** — download required (`README.md` L32) |
| Nepmap synthetic | **Missing** |
| Trained `point_cloud.ply` + `procams.ckpt` | **Missing** |
| COLMAP executable | External install |
| `environment.yml` + `requirements.txt` | Present (torch cu118) |

---

## 15. Exact image and tensor input formats

| Input | Shape | Range | Format |
|-------|-------|-------|--------|
| Patterns | `(3,H_p,W_p)` after load | float `[0,1]` | PNG under `patterns/` |
| Camera images | `(3,H_c,W_c)` | float `[0,1]` | `img_1XXX.png` naming |
| Mask | HxW | 0/1 | `mask/mask.png` |
| Default registrate res | 800×800 | — | `--cam_width/height`, `--prj_width/height` |
| COLMAP | PINHOLE / SIMPLE_PINHOLE | — | sparse model |

---

## 16. Exact output formats

| Output | Path / format | Role |
|--------|---------------|------|
| Gaussians | `point_cloud/iteration_N/point_cloud.ply` | Geometry + PBR |
| Radiometry | `procams/iteration_N/procams.ckpt` | gamma, gain, PSF |
| Poses | `cameras.json`, `projector.json` | K, R, T |
| Compensated pattern | `patterns/{i:03d}.png` | Projector RGB |
| Relit camera | render tree PNGs | Simulation |
| Warp grids | **in-memory only** | Not exported |

---

## 17. Coordinate conventions

| Transform | Direction | Evidence |
|-----------|-----------|----------|
| `getWorld2View2(R,T)` | **world → camera/projector view** | `utils/graphics_utils.py` |
| `warp_points(projector, surf_pts3d)` | **world 3D → projector pixels** | used in `procams_render.py:264` |
| `prj2cam_grid` | projector UV normalized to NDC for sampling into camera-sized grid | L265–267: `prj_pts2d * 2 / prj_size - 1` |
| Pattern sample | **projector framebuffer → camera image plane** (via surface) | `F.grid_sample` L267 |
| Nepmap Blender | Blender c2w → OpenCV (Y/Z flip) | `dataset_readers.py` |

`align_corners=False` in GS `grid_sample` (unlike CSPR’s `True`).

---

## 18. Image orientation conventions

- Top-left origin; OpenCV/COLMAP camera convention.  
- No automatic mirroring documented.  
- Blender data explicitly flipped to OpenCV.

---

## 19. Resolution assumptions

Configurable; registrate defaults **800×800** for cam and proj. Real datasets may differ. Projector size stored on `Projector.prj_size` as `(width, height)`.

---

## 20. Lens-distortion assumptions

COLMAP models limited to **PINHOLE / SIMPLE_PINHOLE** — **no brown distortion model** in loader (`dataset_readers.py` rejects others). Distortion assumed pre-corrected or negligible.

---

## 21. Projector-response assumptions

`PEModel` (`scene/projector.py:82–109`):

- Learnable `gamma` (init 2.2): sRGB→linear via `pow(x, gamma)`  
- Learnable `gain`  
- Optional 5×5 `Conv2d` PSF  

Applied in `Projector.forward` before geometric sampling.

---

## 22. Dense camera–projector map generated?

**Internally yes** (`prj2cam_grid`, compensation `cam2prj_grid` in `procams_compensater.py`) — **not written** as `.npy`/LUT artifacts.

---

## 23. Projector-space warp generated?

**Radiometric compensation PNG**, not a reusable geometric warp field. Geometry is baked into the trained 3D model + per-frame depth. Output of `compensate.py`: optimized projector RGB images.

---

## 24. Closed-loop correction method?

**Partial / radiometric only:** `Compensater.compensate` runs Adam (~100 iters) against desired camera image using the **fixed trained scene** — not a physical capture→remeasure geometric loop. No automatic residual geometric recalibration loop.

---

## 25. Expected runtime and hardware

| Requirement | Evidence |
|-------------|----------|
| NVIDIA CUDA GPU | `README.md` L11 |
| CUDA Toolkit 11.8 | L13 |
| torch 2.6.0+cu118 | `requirements.txt` |
| Train | 20k iters; time in `extra_log.json` minutes |
| Compensate | ~100 Adam steps per desired image |
| macOS / MPS | **Unsupported** |

---

## 26. Missing files / undocumented dependencies

1. Datasets not in clone  
2. No pretrained scene checkpoints  
3. COLMAP external  
4. Shell scripts often omit `-r` / misuse `-s` vs README  
5. `compensate.py` default path `cam/desire/test` vs README `cam/desired/test` (L36)  
6. DiffMorpher/CDC/LAMA mentioned in README — **not in repo**

---

## 27. License constraints

`LICENSE.md`: **Academic / non-profit / non-commercial research use only**; derivative of 3DGS Inria/MPII terms; citation required. Blocks commercial product integration without separate rights.

---

## 28. Independently reusable components

| Component | Symbol | MVP decision |
|-----------|--------|--------------|
| COLMAP registrate + masks | `scripts/registrate.py` | Exclude for single-view flat-wall MVP (heavy) |
| `PEModel` | `scene/projector.py` | Exclude until radiometry milestone |
| `Compensater` | `procams_compensater.py` | Exclude from geometric MVP |
| 2DGS rasterizer | `submodules/diff-gaussian-rasterization` | Exclude (CUDA) |
| Eval metrics | `evaluate.py`, LPIPS/PSNR/SSIM | Optional later |

---

## Calibration path summary

```text
Multi-view calib pattern captures
  → COLMAP (registrate.py) → cameras.json / projector.json
  → train Gaussians + PEModel (train.py)
  → [render] geometric warp via depth + prj2cam_grid
  → [compensate] optimize projector RGB for desired camera look
```

**MVP relevance:** radiometry and multi-view ProCam reference only. **Does not** provide an exportable geometric pre-warp LUT for one iPhone + one projector flat-wall workflow without substantial adaptation and CUDA hardware.
