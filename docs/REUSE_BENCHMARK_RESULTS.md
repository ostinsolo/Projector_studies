# Reuse Benchmark Results

Executed 2026-07-17. No physical projection. Immutable synthetic run untouched.

## Environment

| Item | Value |
|------|-------|
| Host Python OpenCV | 5.0.0 (no `structured_light` in default wheel) |
| Prototype venv | `prototypes/external_reuse/.venv` with `opencv-contrib-python-headless` 5.0.0 |
| Production package | `procam-calibration` (editable) |

## 1. kamino410 sample (original data, adapted for OpenCV 5)

Adapter: `prototypes/external_reuse/adapters/run_kamino410_sample.py`  
Log: `prototypes/external_reuse/logs/kamino410_sample.log`  
Result: `prototypes/external_reuse/results/kamino410_sample.json`

| Metric | Value |
|--------|-------|
| Status | **PASS** after API shim + corner-shape fix (upstream unmodified) |
| Camera RMS | 0.316 |
| Projector RMS | 0.203 |
| Stereo RMS | **0.311** |
| Failure without adapter | `structured_light_GrayCodePattern` missing; then `IndexError` on OpenCV 5 corners |

**Applicability to our two-plane task:** sample uses **multi-pose chessboard + Gray-code** on a single planar board pose set. It does **not** discover two unknown walls, estimate a fold seam, or produce piecewise content pre-warp for a fixed desired camera rectangle.

## 2. Our two-H vs single-H on synthetic two-plane data

Adapter: `prototypes/external_reuse/adapters/bench_two_plane_vs_single_h.py`  
Result: `prototypes/external_reuse/results/two_plane_vs_baselines.json`

| Case class | Result |
|------------|--------|
| Full synthetic suite replay | **14/14 PASS** |
| One-plane control | model=`one_plane` |
| Two-plane cases | two-H selected; assignment ≥0.92; exclusive masks |
| Single-H on genuine two-plane | p95 often **hundreds of px** (fails as expected) |
| Runtime per fit | ~2–4 ms |

External full-intrinsic pipelines (**Cho**, **kamino stereo**, **Huang BA**) were **not applicable** to sparse synthetic ChArUco two-wall observations without their capture stacks — recorded as N/A with reason in JSON.

## 3. Attempts that could not run as drop-in

| Candidate | Attempt | Outcome |
|-----------|---------|---------|
| Cho/Kim VISAPP 2025 | Locate code + run | **No code** |
| Jin et al. JCDE 2026 | Locate code + PDF | **No code**; open PDF blocked |
| Benjumea 2025 | Inspect MATLAB entrypoints | Requires MATLAB + custom `.mat` digital-feature capture; no wall-fold demo |
| RoomAlive | Build | Windows + Kinect — skipped (H) |
| GS-ProCams | Train/infer | CUDA + noncommercial — out of milestone |
| single-shot-pro-cam-calib | MATLAB GUI | MATLAB mandatory — skipped |
| procamcalib | — | GPL — not integrated |

## 4. Accuracy / stability / failure summary

| System | Accuracy evidence | Stability | Failure vs our target |
|--------|-------------------|-----------|------------------------|
| Ours (two-H) | Synthetic suite pass; designed for ≤5/12 gates | Fast, deterministic canonicalize | Physical pending (paused for audit) |
| kamino410 | Stereo RMS ~0.31 on sample | Needs contrib + API shim | Wrong scene model (board poses ≠ two walls) |
| Cho/Kim (paper) | Strong PMP 3D claims | Unknown without code | Custom target + phase; no pre-warp pipeline |
| Jin et al. (paper) | 3.78–8.53 px reported | Semi-auto | Manual reference; no code; px may miss our median gate |

## Conclusion for benchmarks

Executed evidence supports **retaining the current two-plane architecture**. The only runnable MIT structured-light baseline (kamino410) validates Gray-code stereo **components**, not a replacement base.
