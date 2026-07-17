# Component Crosswalk

Comparison of CSPR-Net, GS-ProCams, and classical CV for the automatic geometric projection-mapping MVP. Decision rule: prefer classical structured light when it solves a stage more reliably than a neural model.

| Feature | CSPR-Net implementation | GS-ProCams implementation | Classical CV alternative | Required for MVP | Reuse / adapt / replace | Evidence: file and line or symbol |
|---------|-------------------------|---------------------------|--------------------------|------------------|-------------------------|-----------------------------------|
| Project calibration / correspondence patterns | Color grid PNGs via `generate_ununiform_mesh.py` (3 densities + solid red) | `patterns/{calib,ref,train,test}/` RGB PNGs; COLMAP calib texture | Gray-code / binary structured light + black/white refs | Yes | Replace with Gray-code SL for dense, deterministic cam↔proj maps; adapt CSPR pattern script only for visual tests | CSPR: `generate_ununiform_mesh.py`; GS: `README.md` L51–57, `scripts/registrate.py` |
| Capture camera images of projections | Bundled `real_data/*_image_distorted_origin.jpeg` (3072×2048) | Multi-view `views/NN/cam/raw/{calib,train,test}` | Fixed iPhone stills (USB import or tethered) | Yes | Replace — new capture path for iPhone | CSPR: `Config.TRAIN_PAIRS` in `deep_learning_exp_for_gradient.py:18–22`; GS: `README.md` L58–75 |
| Camera intrinsics | Neural: not used; ray-trace: physical f_mm + sensor → K | COLMAP PINHOLE / SIMPLE_PINHOLE | Optional for planar homography; useful later for 3D | No (flat-wall MVP) | Replace later if moving to 3D | CSPR: `qurdric_transfer.py:363–373`; GS: `scene/dataset_readers.py` PINHOLE branch |
| Projector intrinsics | Neural: not used; ray-trace: throw ratio → K | COLMAP projector-as-camera (`calib` view) | Optional for planar H; needed for full ProCam 3D | No (flat-wall MVP) | Replace later | CSPR: `get_projector_matrices` / `qurdric_transfer.py:385–388`; GS: `scene/projector.py`, `dataset_readers.py` calib view |
| Camera–projector extrinsics | Neural: implicit fixed setup; ray-trace: hardcoded poses | Joint COLMAP world frame | Not required for planar homography from correspondence | No (flat-wall MVP) | Replace later for non-planar 3D | CSPR: `CAM_C`/`PROJ_C` `qurdric_transfer.py:365–379`; GS: `scripts/registrate.py` |
| Dense cam↔proj correspondence | Ray-trace maps in memory; neural `CoordinateNet` residual field | Depth-based `prj2cam_grid` / `cam2prj_grid` inside renderer | Decode Gray-code → per-pixel projector (u,v) | Yes | Replace with classical SL decoder | CSPR: `cam_to_proj_map_*` `qurdric_transfer.py:399–415`; `CoordinateNet`; GS: `procams_render.py` grid_sample path |
| Planar geometric warp (homography) | Not explicit — learned nonlinear map | Implicit via 3D Gaussians + depth | `cv2.findHomography` + `cv2.warpPerspective` / remap | Yes (flat wall) | Replace — classical H is MVP core | OpenCV; neither repo exports H |
| Non-planar / curved geometric pre-warp | Primary goal — `pre_warped.png` via MLP | Implicit geometry in Gaussians; no geometric pre-warp export | Mesh / depth + projective texture, or dense remap LUT | Later (cube / curved) | Adapt CSPR ideas later; not MVP | CSPR: `NeuralWarper.save_hd_results`; GS: no geometric LUT |
| Exportable UV LUT / remap tables | Not saved (maps in RAM only; nets in `.pth`) | Not exported | Save `map_x`, `map_y` float32 `.npy` + apply `cv2.remap` | Yes | Replace — MVP must write LUT files | CSPR: maps not written; GS: grids internal |
| Radiometric compensation | No | Yes — `compensate.py` Adam-optimized patterns + PEModel | Optional: per-channel gain / black level | No (MVP is geometric) | Defer; reuse GS concepts only if radiometry needed later | GS: `compensate.py`, `procams_compensater.py`; CSPR: photometric losses only |
| Mask of illuminated region | Contour / red-channel threshold | White−black ref difference → `mask.png` | Threshold or white−black | Yes | Adapt either; classical white−black preferred | CSPR: `create_mask`; GS: `registrate.py` mask generation |
| Real-time streaming | No | Render may be fast; compensate not real-time | Offline stills first; streaming later | No for first MVP | Replace later | Audits §20 |
| Pretrained weights / ready inference | Yes — bundled `.pth` + data | No — download data + train 20k | N/A | No | CSPR reusable for experimentation only | CSPR: `neural_results_*/net_*.pth`; GS: no checkpoints |
| CUDA / GPU requirement | Optional (CPU OK) | Required (cu118 + extensions) | CPU fine for SL + H | Prefer CPU for MVP | Replace GS for MVP path | CSPR: `DEVICE`; GS: `README.md` L11–13 |
| License for product use | Unknown (no LICENSE) | Non-commercial research only | OpenCV BSD / own code | Yes (clarity) | Prefer own classical code for product path | CSPR: none; GS: `LICENSE.md` |
| Multi-view scene reconstruction | No | Yes — 25 views default + COLMAP | Optional SfM later | No | Defer GS / COLMAP | GS: `arguments` `num_view=25` |
| Simulator / digital twin | Ray-trace quadric (`qurdric_transfer.py`) | Full GS ProCam forward model | Simple plane projector sim for unit tests | Nice-to-have | Adapt CSPR ray-trace for offline tests | CSPR: `qurdric_transfer.py`; GS: `procams_render.py` |

## Summary decisions for MVP

| Stage | Decision |
|-------|----------|
| Correspondence | **Classical Gray-code structured light** |
| Flat-wall warp | **Homography** from correspondence |
| Pre-warp delivery | **Projector PNG + float remap LUT** |
| CSPR-Net | Reference / optional later for curved surfaces |
| GS-ProCams | Out of critical path (CUDA, no geometric LUT, non-commercial) |
| Foundation models (DINO, SAM, etc.) | **Not required** — no gap they uniquely solve for flat-wall geometry |
