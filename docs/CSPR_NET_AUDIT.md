# CSPR-Net Audit

**Repository:** `CSPR-Net/`  
**Remote:** https://github.com/Hao-Laboratory/CSPR-Net  
**Audit type:** Documentation only (no implementation changes)  
**Local status:** Data directories and pretrained `.pth` weights are present.

---

## 1. Research objective

Compute a **projector-space pre-warp** so that when a projector displays it on a **curved (non-planar) surface**, the **camera view appears geometrically undistorted**.

Method: two MLPs (`CoordinateNet`) learn bidirectional camera↔projector coordinate maps with self-supervised cycle consistency. Validated against ray-traced ground truth and real captured photos.

Evidence: `README.md` L3–9.

---

## 2. Paper and implementation correspondence

| Item | Finding |
|------|---------|
| Paper / arXiv / DOI | **None** in repo |
| Citation | None |
| Correspondence | README describes method; no publication link |

Evidence: `README.md` (no citation block); no `arxiv`/`doi` references in Python sources.

---

## 3. Supported operating systems and hardware

| Aspect | Finding | Evidence |
|--------|---------|----------|
| OS | Unspecified; portable Python | `README.md` L68–82 |
| GPU | Optional | `DEVICE = cuda if available else cpu` — `deep_learning_simulation_for_gradient.py:26`, `deep_learning_exp_for_gradient.py:27` |
| Display | Needed for PyVista `plotter.show()` | `qurdric_transfer.py` visualization path |

---

## 4. Required Python, CUDA, PyTorch and compiled dependencies

**README install (exact):**

```bash
pip install numpy opencv-python matplotlib pyvista torch scikit-image Pillow
```

| Dependency | Role |
|------------|------|
| numpy, opencv-python, matplotlib, torch, scikit-image, Pillow | Core |
| pyvista | Optional 3D viz in `qurdric_transfer.py` |

- No Python version pin, no `requirements.txt`, no CUDA pin.
- No custom CUDA extensions; standard wheels only.

---

## 5. Required checkpoint files and datasets

### Present locally (no download needed)

| Asset | Path |
|-------|------|
| Sim patterns + distorted views | `sim_data/` |
| Real captures + patterns | `real_data/` |
| Ray-tracing GT | `data/` |
| Sim pretrained nets | `neural_results_simulation/net_c2p.pth`, `net_p2c.pth` |
| Exp pretrained nets | `neural_results_exp/net_c2p.pth`, `net_p2c.pth` |

### Missing unless generated

- `data/generated_meshes.png` (from `generate_ununiform_mesh.py`)
- `neural_results_exp/loss_metrics.csv` (from training)

---

## 6. Training required vs pretrained inference

| Mode | Available |
|------|-----------|
| Pretrained inference | **Yes** — weights bundled |
| Train from scratch | Optional (`ITERS = 6000`) |
| Fine-tune (exp) | Supported via `Config.PRETRAINED_C2P/P2C` (default `None`) |

Defaults: simulation → `inference`; experimental → `inference_custom`.

---

## 7. Exact executable entry points

| Script | CLI | Notes |
|--------|-----|-------|
| `generate_ununiform_mesh.py` | none | Not listed in README run section |
| `qurdric_transfer.py` | none | README: `python qurdric_transfer.py` |
| `deep_learning_simulation_for_gradient.py` | `train` \| `inference` | Default `inference` (L508) |
| `deep_learning_exp_for_gradient.py` | `train` \| `inference` \| `inference_custom [path]` | Default `inference_custom` (L828); README omits `inference_custom` |

---

## 8. Exact input formats

| Input | Format | Size | Evidence |
|-------|--------|------|----------|
| Projector patterns | RGB PNG `{1,2,3}_pro.png` | 1920×1080 | `generate_ununiform_mesh.py`; `Config.H_PROJ,W_PROJ` |
| Sim camera views | RGB PNG `*_image_distorted.png` | 3072×1728 | sim Config L29–30 |
| Real captures | JPEG `*_image_distorted_origin.jpeg` | 3072×2048 → crop `[160:1888, 0:3072]` → 3072×1728 | exp `preprocess_and_crop` |
| Mask source | `red_distorted_origin.jpeg` | same crop; red channel > 120 | exp `create_mask` |
| Custom infer image | JPEG/PNG path | resized to projector res | `inference_custom` |

Tensors: `[B, 3, H, W]`, float `[0, 1]`, RGB. Training uses 480×270 (`TRAIN_SCALE=0.25`).

---

## 9. Exact output formats

| Output | Description |
|--------|-------------|
| `pre_warped.png` | Main deliverable: projector pre-warp (1920×1080) |
| `net_c2p.pth`, `net_p2c.pth` | MLP state dicts |
| `fake_cam.png`, `fake_proj.png` | Cycle visualizations |
| `camera_mask.png`, `Final_Mask_on_Projector_Space.png` | Masks |
| Ray-trace PNGs in `data/` | Distorted / prewarped / verified GT |

No standalone float LUT file on disk.

---

## 10. Camera assumptions

**Neural path:** Fixed resolution 3072×1728; fixed pose across captures; no explicit pinhole required.

**Ray-tracing path** (`qurdric_transfer.py:347–375`):

| Parameter | Value |
|-----------|-------|
| Resolution | 3072×1728 |
| Focal length | 10.0 mm |
| Sensor | `"1/1.8"` |
| Principal point | image center |
| Position | `[0, -7, 1]` |
| Model | Pinhole K, no distortion coeffs |

---

## 11. Projector assumptions

**Neural:** 1920×1080; full-frame mask (`ones`).

**Ray-tracing** (`qurdric_transfer.py:377–388`):

| Parameter | Value |
|-----------|-------|
| Resolution | 1920×1080 |
| Throw ratio | 1.2 → fx=fy=2304 |
| Offset | 0% |
| Position | `[0, -4.5, 2]` |

---

## 12. Calibration assumptions

| Aspect | Ray-tracing | Neural |
|--------|-------------|--------|
| Surface | Known quadric (cylinder R=2.0) | Implicit in images |
| Intrinsics / extrinsics | Hardcoded | Not required; fixed setup assumed |
| Patterns | 3 colored grids (+ red for mask in real) | Same |

---

## 13. Number of images / frames / patterns

- **3** pattern pairs for training (`TRAIN_PAIRS`)
- Real: **+1** red capture for mask
- Static stills only — no video / multi-view sequence

---

## 14. Camera intrinsics must already be known?

| Pipeline | Required? |
|----------|-----------|
| Neural | **No** |
| Ray-tracing GT | **Yes** (or derived from f_mm + sensor) |

---

## 15. Projector intrinsics must already be known?

| Pipeline | Required? |
|----------|-----------|
| Neural | **No** |
| Ray-tracing | **Yes** (throw ratio + resolution) |

---

## 16. Camera–projector extrinsics must already be known?

| Pipeline | Required? |
|----------|-----------|
| Neural | **No** (fixed unknown pose) |
| Ray-tracing | **Yes** (hardcoded) |

---

## 17. Geometric correction produced?

**Yes.** Primary output: `pre_warped.png` (projector-space image that should appear rectified in the camera). Ray-tracing also writes `*_image_prewarped.png` in `data/`.

---

## 18. Radiometric compensation produced?

**No.** Losses are L1 + Sobel gradient + cycle + smoothness + mask — geometric/photometric alignment only, not per-pixel gain maps.

---

## 19. Projector-space UV warp or LUT produced?

| Form | Produced? |
|------|-----------|
| Dense float maps in memory | Sim ray-trace only (`cam_to_proj_map_*`, `proj_to_cam_map_*`) — **not saved** |
| Neural field | Implicit in `.pth` |
| Exportable LUT (`.npy` etc.) | **No** |
| PNG pre-warp | **Yes** |

Inference uses `F.grid_sample`, not exported `cv2.remap` maps.

---

## 20. Usable in real time?

**No.** Offline train (6000 iters); HD inference evaluates MLP on millions of coordinates (exp chunks at 200000). No streaming loop.

---

## 21. Expected execution time and memory

| Clue | Value | Evidence |
|------|-------|----------|
| Training iters | 6000 | `Config.ITERS` |
| Train res | 480×270 | `TRAIN_SCALE` |
| HD grids | 1728×3072 + 1080×1920 | `save_hd_results` |
| OOM mitigation | chunk 200k | exp ~L494 |
| Checkpoint size | ~275 KB each | local files |

No official VRAM/time benchmarks in README. Inference prints elapsed seconds.

---

## 22. Missing code, undocumented assets, reproduction blockers

1. No `LICENSE` file
2. No version pins
3. No paper
4. `generate_ununiform_mesh.py` undocumented in README run section
5. Exp default `inference_custom` undocumented
6. Hardcoded resolutions — new hardware needs code edits
7. PyVista `show()` may block headless runs
8. Dense maps not persisted from ray-tracing

---

## 23. License restrictions

**Unknown / no license file.** Treat as all-rights-reserved until authors clarify (Hao-Laboratory).

---

## 24. Components reusable independently

| Component | Location | Reuse |
|-----------|----------|-------|
| `CoordinateNet` | both `deep_learning_*.py` | Small residual MLP warp |
| `GradientLoss` | both neural scripts | Sobel edge loss |
| `NeuralWarper` | both neural scripts | Train/infer loop |
| Ray-tracing core | `qurdric_transfer.py` | Quadric intersect + maps |
| Pattern generator | `generate_ununiform_mesh.py` | Color grids + red mask |
| Mask extraction | `create_mask()` | Contour / red threshold |

---

## macOS / CPU readiness

**Yes for inference:** CUDA optional; weights bundled. Training on CPU is slow. PyVista optional for viz only.

Recommended (derived from code + README):

```bash
cd CSPR-Net
pip install numpy opencv-python matplotlib pyvista torch scikit-image Pillow
python deep_learning_simulation_for_gradient.py inference
python deep_learning_exp_for_gradient.py inference
python deep_learning_exp_for_gradient.py inference_custom ./real_data/test.jpg
```
