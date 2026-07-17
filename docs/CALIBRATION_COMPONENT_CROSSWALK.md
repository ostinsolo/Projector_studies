# Calibration Component Crosswalk

| Feature | CSPR-Net file and symbol | GS-ProCams file and symbol | Input | Output | Coordinate system | Required for MVP | Reuse / adapt / exclude | Evidence |
|---------|--------------------------|----------------------------|-------|--------|-------------------|------------------|-------------------------|----------|
| Pattern generation | `generate_ununiform_mesh.py` (`generate_*_grid`, `main`) | Dataset `patterns/{calib,ref,train,test}/` (external assets; no generator in-repo) | Resolution 1920×1080 (CSPR) | RGB PNG patterns | Projector framebuffer pixels | Yes | **Adapt** CSPR idea; **replace** with Gray-code for automatic dense decode | CSPR README L28–32; GS README L51–57 |
| Image capture / dataset loading | Bundled `real_data/`, `sim_data/`; `load_image_tensor`, `preprocess_and_crop` | `scene/dataset_readers.py` (`readColmapSceneInfo*`, Nepmap loaders); `utils/image_utils.loadImage` | Paths / COLMAP tree | Torch `(3,H,W)` `[0,1]` | Camera image pixels (after optional crop) | Yes | **Replace** with iPhone still protocol | CSPR exp L179–198; GS `dataset_readers.py` |
| Camera calibration | `get_camera_matrices_physical` (sim only; not from images) | `scripts/registrate.py` → COLMAP; load K from sparse model | f_mm/sensor or calib photos | K, poses | Camera intrinsics (pixels) | Later / optional for planar H | **Exclude** for flat H MVP; GS COLMAP **exclude** until multi-view | CSPR `qurdric_transfer.py:21–66`; GS `registrate.py` |
| Projector calibration | `get_projector_matrices` (throw ratio; sim only) | COLMAP projector-as-camera (`calib` view); `Projector` / `ProjectorInfo` | throw ratio or calib pattern | K, R, T | Projector intrinsics (pixels) | Later / optional | **Exclude** for flat H MVP | CSPR L78–105; GS `scene/projector.py:10–45` |
| Distortion correction | None (ideal pinhole / absorbed in MLP) | PINHOLE only — no Brown model | — | — | distorted→undistorted **not modeled** | Document assumption | **Exclude** until measured; do not silently ignore if iPhone distortion large | CSPR ray-trace; GS `dataset_readers.py` PINHOLE check |
| Correspondence calculation | Ray-trace `cam_to_proj_map_*`; neural `CoordinateNet` `net_c2p`/`net_p2c` | Depth unproject + `warp_points` → `prj2cam_grid` / `cam2prj_grid` | Images or 3D surfels | Dense maps or NDC grids | See COORDINATE_SYSTEMS.md | Yes | **Replace** with SL decode for MVP; neural/GS grids **exclude** from first path | CSPR L399–415, `CoordinateNet`; GS `procams_render.py:264–267` |
| Geometry estimation | Implicit MLP or known quadric | 2D Gaussian surfels + COLMAP poses | Captures / SfM | Scene + poses | World / view | Planar H for MVP | **Replace** with homography (flat); dense map later | CSPR `CoordinateNet`; GS `GaussianModel` |
| Radiometric estimation | None | `PEModel` gamma/gain/PSF | Patterns + captures | `procams.ckpt` | Linear/sRGB projector response | No (MVP geometric) | **Exclude** until after geometric MVP | GS `scene/projector.py:82–109` |
| Warp generation | `F.grid_sample` → `pre_warped.png` in `save_hd_results` | Compensated RGB via `Compensater.compensate` (radiometric); geometric via render | Desired content / desired camera | Projector PNG | Projector←camera sampling | Yes (geometric) | **Adapt** CSPR pre-warp concept; **implement** via H/remap in INTEGRATION_ROOT | CSPR exp L525–526; GS `compensate.py` |
| Rendering / display | Implicit (user projects PNG); sim `cv2.remap` | `procams_render.render` BRDF+pattern | Pattern tensor | Camera RGB sim | World→cam with projector light | Validation display | OpenCV fullscreen in integration | GS `procams_render.py:259–271` |
| Validation | `calculate_metrics` SSIM/PSNR/RMSE vs GT | `evaluate.py`, `training_report` PSNR/SSIM/LPIPS | Pred / GT images | Scalars | Camera image | Yes | **Adapt** metrics; add geometric corner/line errors | CSPR `calculate_metrics`; GS `evaluate.py` |
| Metrics | SSIM, PSNR, RMSE (sim/exp) | PSNR, SSIM, LPIPS, FPS | Images | JSON/logs | Camera | Yes | Extend with reprojection stats | CSPR L362+; GS eval |
| Checkpoint loading | `NeuralWarper.load_models` → `net_*.pth` | `GaussianModel` PLY + `Projector.load_ckpt` | Paths | Weights | — | No for classical MVP | CSPR OK for experiment; GS needs train | CSPR `load_models`; GS `projector.py:77–80` |
| Training-only components | `NeuralWarper.train`, losses, 6000 iters | `train.py` / densification / 20k iters | Train pairs | Updated weights | — | No | **Exclude** from MVP critical path | Both train entry points |
| Inference-only components | `run_inference`, `inference_custom` | `render.py`, `compensate.py` | Weights + images | PNG / logs | As above | CSPR optional | Classical path preferred | CSPR L816+; GS CLI |
| Mask of illuminated region | `create_mask` (contour / red threshold) | white−black → `mask/mask.png` in `registrate.py` | Red or ref images | Binary mask | Camera pixels | Yes | **Adapt** white−black | CSPR exp L200–220; GS registrate mask |
| Closed-loop residual correction | Absent | Adam compensate loop (radiometric, simulated) | Desired image | Pattern | Camera desired ← projector | Later (M7) | **Replace** with physical residual loop in INTEGRATION_ROOT | GS `procams_compensater.py:116+` |

## MVP integration decisions

| Decision | Choice |
|----------|--------|
| Core correspondence | Classical structured light in `INTEGRATION_ROOT` (neither repo provides Gray-code decode) |
| Flat-wall geometry | Homography from automatic correspondences |
| Pre-warp delivery | Projector PNG + lossless float maps |
| CSPR-Net role | Reference pre-warp I/O; optional curved later |
| GS-ProCams role | Radiometry / multi-view later; CUDA blocker now |
| External ML models | **Forbidden** per project scope |
