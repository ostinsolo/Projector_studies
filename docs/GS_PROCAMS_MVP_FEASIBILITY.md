# GS-ProCams Flat-Wall MVP Feasibility Audit

**Repository:** `/Users/ostino/Documents/Code/Projector_studies/GS-ProCams`  
**Git commit:** `dfb7f449710ce41a63369a447ca728e0cb220d97`  
**Prior audit:** `/Users/ostino/Documents/Code/Projector_studies/docs/GS_PROCAMS_CALIBRATION_AUDIT.md`  
**Scope:** Read-only feasibility note (historical). Production flat-wall MVP is
ChArUco + planar homography; GS-ProCams remains `REUSABLE_COMPONENTS_ONLY`.

---

## Executive answers

### 1. Does it estimate camera–projector calibration or expect it as input?

**Estimates it (real-world path); expects it as input (synthetic path).**

- **Real-world:** `scripts/registrate.py` runs a full COLMAP SfM pipeline (feature extraction, matching, mapper, bundle adjustment) over calibration-pattern captures from multiple camera viewpoints plus one projector reference image. Camera and projector intrinsics/extrinsics are read from the resulting COLMAP sparse model (`scene/dataset_readers.py` `readColmapSceneInfo`, `readColmapCameras`).
- **Synthetic (Nepmap):** poses and intrinsics are read from `transforms.json`; no COLMAP step (`readNepmapSceneInfo`, `readNepmapCameras`).

Evidence: `scripts/registrate.py:10–85`, `README.md:78–82`, `scene/dataset_readers.py:350–395`, `474–600`.

---

### 2. Must camera poses already exist?

**No for the documented real-world pipeline; yes for Nepmap synthetic.**

- Real-world: COLMAP reconstructs camera poses jointly with the projector “camera” entry (`calib_name`, default `"calib"`).
- Nepmap: Blender camera matrices in `transforms.json` are converted to OpenCV convention and loaded directly.

Evidence: `arguments/__init__.py:55` (`_calib_name = "calib"`), `scene/dataset_readers.py:260–272`, `474–475`.

---

### 3. Is COLMAP reconstruction required?

**Required for the real-world / custom-setup path; not required for Nepmap synthetic.**

- README custom-setup instructions: “We provide a script that runs calibration using COLMAP” (`README.md:78–82`).
- `registrate.py` default path calls `run_colmap()` unless registering a single novel view into an existing model (`--view_id`).
- Training loader for real data reads `setups/<setup>/colmap/sparse/0` (`readColmapSceneInfo`).

Evidence: `scripts/registrate.py:245–252`, `scene/dataset_readers.py:385–395`.

---

### 4. How many camera viewpoints are required?

**Default 25 training viewpoints; published ablations down to 2; not 1.**

| Setting | Value | Evidence |
|---------|-------|----------|
| Default train views | **25** | `arguments/__init__.py:62–63` (`num_view = 25`, `max_train = 25`) |
| `registrate.py` default | **25** | `scripts/registrate.py:230` (`--views`, default=25) |
| Paper real-world script | **25** (ablation list includes 20, 10, 5, 4, **2**) | `scripts/real-world.sh:16` |
| Minimum in code | **2** | `scene/dataset_readers.py:380–381` (`num_view == 2` → views `[1, 21]`) |
| Novel-view add-on | **+1** per new view after initial COLMAP | `README.md:90–94`, `registrate.py:87–149` |

---

### 5. How many projected images are required per viewpoint?

**Depends on stage; training uses a shared pool of 100 patterns split across views.**

| Stage | Per viewpoint | Evidence |
|-------|---------------|----------|
| COLMAP calibration | **1** camera capture of the shared calib pattern + **1** projector-side calib image copied into COLMAP ref folders | `registrate.py:24–38`, `28–34` |
| Mask generation | **2** ref captures (black `img_0001`, white `img_0002`) | `registrate.py:213–214`, `make_masks()` |
| Training (default) | **4** train pattern–image pairs per view (100 patterns ÷ 25 views) + **1** black ref pair | `dataset_readers.py:217–223`, `318–325` |
| Test / compensation | Additional test patterns and desired camera images per view | `README.md:102–106`, `compensate.py:23–36` |

Default filename convention: `img_1XXX.png` paired with patterns from `patterns/train/` (`dataset_readers.py:311`).

---

### 6. Is a single fixed iPhone viewpoint supported?

**No as a documented or validated operating mode.**

- Pipeline assumes `views/01 … views/25` (or fewer in ablation, minimum 2) with the camera moved between viewpoints (`README.md:51–75`, `registrate.py:28–34`).
- Novel-view registration adds views to an **existing multi-view COLMAP model**, not a from-scratch single-view calibration (`README.md:90–94`).
- No code path sets `num_view=1`; `get_train_view_ids` minimum is 2 (`dataset_readers.py:380–381`).

A single fixed iPhone at one pose is incompatible with the published capture protocol and COLMAP multi-camera reconstruction design.

---

### 7. Does it directly generate a projector pre-warp?

**No — not as an exportable geometric pre-warp for the flat-wall MVP.**

- Internal geometry uses per-pixel grids (`prj2cam_grid`, `cam2prj_grid`) built from trained 3D Gaussians and depth (`gaussian_renderer/procams_render.py:264–267`, `procams_compensater.py:41–45`).
- These grids are **in-memory only**; the prior audit confirms they are not written as `.npy`/LUT artifacts.
- `compensate.py` output is **radiometrically optimized projector RGB PNGs** for a desired camera appearance, not a reusable projector-space geometric warp map.

Evidence: `compensate.py:37–38` (output under `prj/cmp/`), `procams_render.py:259–271`.

---

### 8. Which stages require CUDA-specific kernels?

| Stage | CUDA required? | Evidence |
|-------|----------------|----------|
| Environment / install | **Yes** | `README.md:11–13`, `requirements.txt:1–5` (`torch==2.6.0+cu118`) |
| `train.py` | **Yes** | Hard-coded `device="cuda"`, `torch.cuda.set_device` (`train.py:45, 164–165`) |
| 2D Gaussian rasterizer | **Yes** | `submodules/diff-gaussian-rasterization/setup.py` (`CUDAExtension`, `.cu` sources) |
| KNN initialization | **Yes** | `submodules/simple-knn/setup.py` (`CUDAExtension`), `gaussian_model.py:17` (`distCUDA2`) |
| `render.py` | **Yes** | `device = torch.device(f"cuda:{args.gpu_id}")` |
| `compensate.py` / `Compensater` | **Yes** | `compensate.py:31–32`, `procams_compensater.py:31, 45` |
| `registrate.py` (COLMAP) | **Optional GPU for SIFT** | `--no_gpu` flag; COLMAP itself is external CPU/GPU (`registrate.py:13, 50, 240`) |
| Mask generation (`make_masks`) | **No CUDA in repo code** | OpenCV + scikit-image on CPU (`registrate.py:205–224`) |

All stages that produce calibration geometry, simulation, or compensation require CUDA PyTorch extensions.

---

### 9. Can any relevant calibration stage run on macOS CPU or MPS?

**No for calibration/training/inference; partial CPU-only preprocessing only.**

- README lists **CUDA-capable GPU** as a requirement; no MPS/CPU backend is documented or implemented.
- `ModelParams.data_device` defaults to `"cuda"`; cameras/projector fall back to CUDA on failure (`arguments/__init__.py:57`, `scene/cameras.py:39–40`, `scene/projector.py:27–28`).
- No `mps` or Apple-backend references in application code (only third-party GLM docs).
- **Partial exception:** `registrate.py` mask generation and COLMAP with `--no_gpu` can run on CPU/macOS, but this only prepares masks and sparse poses; it does **not** replace the CUDA-bound `train.py` / `render.py` / `compensate.py` loop.

**Conclusion:** The flat-wall-relevant calibration product (trained ProCam model + compensation) cannot run on macOS CPU or MPS.

---

### 10. Does it contain a reusable component that improves the current flat-wall pipeline?

**Limited — a few isolated utilities; nothing that replaces or directly improves CSPR-Net single-view geometric pre-warp generation without the full multi-view CUDA training stack.**

| Component | Path | Flat-wall MVP utility |
|-----------|------|------------------------|
| Projector-region mask from white−black ref | `scripts/registrate.py` `make_masks()`, `threshDeProCams()` | **Potentially reusable** for capture validation / ROI masking (DeProCams-derived, CPU/OpenCV) |
| `PEModel` (gamma, gain, PSF) | `scene/projector.py:82–109` | **Post-geometric radiometry only**; requires trained checkpoint |
| `Compensater` | `gaussian_renderer/procams_compensater.py` | **Radiometric compensation** after full GS-ProCams train; not geometric MVP |
| COLMAP registrate wrapper | `scripts/registrate.py` | **Heavy multi-view path**; mismatched to single iPhone flat wall |
| Eval metrics (PSNR/SSIM/LPIPS) | `evaluate.py`, `utils/eval_utils.py` | **Optional later** for radiometric comparison |
| Dense warp grids | `procams_render.py`, `procams_compensater.py` | **Not exported**; tied to trained 3D scene |

For the current CSPR-Net-centric geometric flat-wall MVP, GS-ProCams does not offer a drop-in improvement. The only low-friction borrow is mask/thresholding logic from `registrate.py`.

---

## Blocker summary

```text
Flat-wall MVP needs:
  1 fixed projector + 1 fixed iPhone + exportable geometric pre-warp + macOS-friendly path

GS-ProCams provides:
  25-view (min 2) COLMAP capture → CUDA 2DGS training → radiometric compensation PNGs
  Internal warp grids only; no exported projector-space geometric LUT
```

---

## Classification

**REUSABLE_COMPONENTS_ONLY**

GS-ProCams is a multi-view, CUDA-only joint scene + radiometry system. It does not directly satisfy the flat-wall MVP’s single-view geometric pre-warp requirement, but isolated utilities (notably white/black mask generation) may be borrowed without adopting the full pipeline.
