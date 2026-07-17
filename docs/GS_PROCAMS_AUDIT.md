# GS-ProCams Audit

**Repository:** `GS-ProCams/`  
**Remote:** https://github.com/RealQingyue/GS-ProCams  
**Audit type:** Documentation only (no implementation changes)  
**Local status:** No `data/` or `output/` directories; no pretrained checkpoints.

---

## 1. Research objective

Build a **differentiable projector–camera (ProCam) system** using **2D Gaussian surfel splatting** to:

- Jointly model scene geometry/materials (Gaussians + BRDF) and projector radiometry (`PEModel`)
- Simulate camera images for given projector patterns
- Optimize **radiometric compensation** patterns so the camera sees a desired appearance

Evidence: `README.md` L1–5; `train.py`; `gaussian_renderer/procams_render.py`; `gaussian_renderer/procams_compensater.py`.

---

## 2. Paper and implementation correspondence

| Field | Value | Evidence |
|-------|-------|----------|
| Title | GS-ProCams: Gaussian Splatting-Based Projector-Camera Systems | `README.md` L5, L133–139 |
| Authors | Qingyue Deng, Jijiang Li, Haibin Ling, Bingyao Huang | citation / `LICENSE.md` |
| Venue | IEEE TVCG / ISMAR 2025 | `README.md` L5 |
| DOI | 10.1109/TVCG.2025.3616841 | `README.md` L138 |
| Project page | https://realqingyue.github.io/GS-ProCams/ | `README.md` L3 |
| ArXiv | Not listed in repo | — |

Lineage: 3DGS + 2DGS rasterizer; DeProCams-inspired registration (`scripts/registrate.py`); Nepmap synthetic data. README mentions DiffMorpher/CDC/LAMA integrations — **no corresponding code** in this clone.

---

## 3. Supported operating systems and hardware

| Requirement | Evidence |
|-------------|----------|
| CUDA-capable GPU | `README.md` L11 |
| CUDA Toolkit 11.8 | `README.md` L13 |
| C++ compiler for extensions | `README.md` L12 (MSVC/VS2019 on Windows) |
| COLMAP (external) | `README.md` L78–79 |
| macOS without NVIDIA CUDA | **Not supported** |

---

## 4. Required Python, CUDA, PyTorch and compiled dependencies

**Conda** (`environment.yml`): Python **3.11**.

**Pip** (`requirements.txt`): `torch==2.6.0+cu118`, matching torchvision/torchaudio, plus numpy, scipy, matplotlib, tqdm, pyyaml, scikit-image, opencv-python, plyfile.

**CUDA extensions** (`README.md` L27):

```bash
pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn
```

| Extension | Package | Role |
|-----------|---------|------|
| diff-gaussian-rasterization | `diff_surfel_rasterization` | 2DGS CUDA rasterizer |
| simple-knn | `simple_knn` | KNN distances for Gaussian init |

Runtime: LPIPS VGG weights downloaded on first eval (`lpipsPyTorch/`).

---

## 5. Required checkpoint files and datasets

### Local clone

| Asset | Present? |
|-------|----------|
| `data/` | **No** (gitignored) |
| `output/` | **No** |
| Pretrained GS-ProCams models | **No** |

### External downloads required

| Dataset | Source | README |
|---------|--------|--------|
| Real-world / compensation | SharePoint link | L3, L32 |
| Synthetic Nepmap | https://github.com/yoterel/nepmap | L32 |

Training writes `{model_path}/point_cloud/iteration_{N}/point_cloud.ply` and `{model_path}/procams/iteration_{N}/procams.ckpt`.

---

## 6. Training required vs pretrained inference

| Mode | Available? |
|------|------------|
| Out-of-box pretrained inference | **No** |
| Train from scratch | Yes — default **20,000** iterations |
| Load trained model | Yes — `render.py` / `compensate.py` at `--iteration` (default 20000) |

---

## 7. Exact executable entry points

### Install (`README.md` L17–28 — exact)

```bash
git clone https://github.com/RealQingyue/GS-ProCams.git
conda env create -f environment.yml
conda activate gs-procams
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization -e submodules/simple-knn
```

### Pipeline scripts (`README.md` L35–42)

```bash
bash ./scripts/synthetic.sh
bash ./scripts/real-world.sh
bash ./scripts/compensate.sh
```

### Documented Python commands (`README.md` L80–105)

```bash
python scripts/registrate.py -r 'data/compensation' -s example
python train.py -r 'data/compensation' -s example -m 'output/example'
python scripts/registrate.py -r 'data/compensation' -s example --view_id 26
python render.py -r 'data/compensation' -s example -m 'output/example' -o 'output/example/render' --views 26
python compensate.py -r 'data/compensation' -s example -m 'output/example' --view_id 26
```

### Key CLI modules

| Entry | Role |
|-------|------|
| `train.py` | Joint scene + projector training |
| `render.py` | Relight / simulate camera views |
| `compensate.py` | Radiometric compensation |
| `scripts/registrate.py` | COLMAP registration + masks |
| `evaluate.py` | Batch metrics |

**Note:** Shell scripts often pass `-s` with a full path and omit `-r` (e.g. `real-world.sh` L81, `compensate.sh` L65, `synthetic.sh` L71), which mismatches `train.py`’s expected `join(root, "setups", setup)`. Prefer the README `-r` / `-s` form.

---

## 8. Exact input formats

### Real-world / compensation layout (`README.md` L51–76)

```text
<data/compensation>/
├── patterns/{calib,ref,test,train}/
└── setups/<setup>/views/<NN>/cam/raw/{calib,ref,test,train}/
```

### Naming (`scene/dataset_readers.py`)

- Train camera images: `img_1{idx:03d}.png` (e.g. `img_1001.png`)
- Pattern stems must match camera image stems
- Ref black: `img_0001`
- Mask: `mask/mask.png` per view
- COLMAP: PINHOLE or SIMPLE_PINHOLE only
- Default registrate resolution: **800×800** cam and proj

### Nepmap synthetic

- Detected via `{root}/setups/{setup}/img2tex.json`
- Uses `transforms.json` with `K_cam`, `K_proj`, Blender matrices

---

## 9. Exact output formats

| Output | Format |
|--------|--------|
| `point_cloud.ply` | Gaussian + PBR attributes |
| `procams.ckpt` | PEModel + optimizer state |
| `cameras.json`, `projector.json` | Intrinsics/extrinsics |
| `render` outputs | Relit PNGs, depth, normals, albedo |
| `compensate` outputs | `patterns/{i:03d}.png` (compensated projector), `imgs/{i:03d}.png`, `compensate.log` |
| Metrics | `metrics.json`, `fps.json`, eval JSON |

---

## 10. Camera assumptions

- Pinhole, undistorted (COLMAP PINHOLE / SIMPLE_PINHOLE)
- Default capture size in registrate: 800×800
- Static pose per viewpoint
- RGB `[0,1]`, 3-channel
- Per-view occlusion mask from white−black refs
- OpenCV / 3DGS world↔view convention

---

## 11. Projector assumptions

- Modeled as pinhole camera (`scene/projector.py`)
- Single global projector; COLMAP view named `calib` by default
- Default resolution 800×800 in registrate
- Radiometric `PEModel`: learnable gamma, gain, optional 5×5 PSF
- Patterns: RGB tensors `(3, H, W)` in `[0,1]`

---

## 12. Calibration assumptions

- COLMAP SfM on camera+projector calib pattern captures (`scripts/registrate.py`)
- Principal point refinement enabled
- No separate structured-light geometric calibration
- Novel view: one calib capture + image registrator

---

## 13. Number of images / frames / views / patterns

| Setting | Default |
|---------|---------|
| Training viewpoints | 25 |
| Pattern pool | 100 (≈4 train pairs/view + ref) |
| Calib captures | 1 per training view + projector |
| Compensation script views | 26, 27, 28 (`scripts/compensate.sh`) |

---

## 14. Camera intrinsics must already be known?

**No** for real-world COLMAP path (estimated).  
**Yes** for Nepmap (`K_cam` in `transforms.json`).

---

## 15. Projector intrinsics must already be known?

**No** for real-world (COLMAP `calib` view).  
**Yes** for Nepmap (`K_proj`).

---

## 16. Camera–projector extrinsics must already be known?

**No.** Poses estimated jointly in COLMAP’s shared world frame.

---

## 17. Geometric correction produced?

**Implicit only** (depth-based `prj2cam_grid` inside the renderer).  
**No** exported homography, mesh warp, or geometric compensation file.

---

## 18. Radiometric compensation produced?

**Yes.** `compensate.py` optimizes full projector RGB patterns (Adam + Huber, 100 iters per desired image) using trained BRDF + PEModel.

---

## 19. Projector-space UV warp or LUT produced?

Internal `prj2cam_grid` / `cam2prj_grid` for `grid_sample` — **not exported** as `.npy`/LUT. Output is compensated RGB PNGs.

---

## 20. Usable in real time?

| Stage | Real-time? |
|-------|------------|
| Forward render | Possibly; FPS measured |
| Compensation | **No** (100 Adam iters per image) |
| Training | **No** (20k iters) |

---

## 21. Expected execution time and memory

- Training time logged to `extra_log.json` in minutes — no fixed benchmark
- Shell scripts poll `nvidia-smi`; **no minimum VRAM stated**
- Default 20,000 iterations

---

## 22. Missing code, undocumented assets, reproduction blockers

1. **Datasets not in clone** — must download SharePoint + Nepmap
2. **No pretrained checkpoints**
3. **COLMAP required** externally
4. **Shell script `-r`/`-s` misuse** vs README form
5. **`compensate.py` default desired path** uses `cam/desire/test` (L36) while README says `cam/desired/test` (L102)
6. DiffMorpher/CDC/LAMA mentioned but not implemented
7. Hardcoded `.cuda()` — no MPS/CPU training path

---

## 23. License restrictions

`LICENSE.md`: **Academic / non-profit / non-commercial research use only.** Derivative of 3D Gaussian Splatting (Inria/MPII non-commercial terms). Citation required for publications.

`README.md` L117–118 reiterates non-commercial research use.

---

## 24. Components reusable independently

| Component | Notes |
|-----------|-------|
| `diff_surfel_rasterization` | Standalone 2DGS CUDA rasterizer |
| `simple_knn` | Standard 3DGS KNN |
| `scene/projector.py` PEModel | Gamma/gain/PSF (CUDA/PyTorch) |
| `scripts/registrate.py` | COLMAP + mask helpers (partial) |
| `lpipsPyTorch/` | Self-contained LPIPS |
| Eval harness | Tied to their directory layout |

---

## macOS / CPU readiness

**No.** Requires NVIDIA CUDA 11.8, cu118 PyTorch wheels, and CUDAExtension builds. Not runnable on Apple Silicon MPS for train/render/compensate.
