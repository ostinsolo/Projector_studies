# Production Pipeline Spec — Flat Wall

Status: authoritative for production. Experimental CSPR / GS paths are documented
elsewhere and are **not** part of this pipeline.

## Scope

| Item | Value |
|------|-------|
| Projector | one, fixed |
| Camera | one fixed iPhone |
| Surface | planar wall |
| Method | automatic ChArUco + planar homography |
| Output | projector-space pre-warp + validation report |

## One command

```bash
procam-calibrate auto-wall --run-dir <NEW_RUN_DIR>
```

Default `--pipeline homography`. Do not pass `--pipeline cspr` for production.

One-shot equivalent:

```bash
procam-calibrate flatwall-homography --run-dir <NEW_RUN_DIR>
```

## Authoritative stages

1. Discover display (wired ProjFB, preferably 1920×1080) and iPhone camera  
2. Display ChArUco calibration board (shared `charuco.py`)  
3. Capture CamImg  
4. Detect known projector corner IDs in CamImg  
5. Estimate `H_cam_from_proj` (ProjFB → CamImg)  
6. Invert to `H_proj_from_cam` (CamImg → ProjFB)  
7. **Define and save desired camera-space target** (independent of corrected capture)  
8. Generate projector-space pre-warp  
9. Display corrected validation board  
10. Capture corrected board  
11. Compare detections to the saved desired target  
12. Save matrices, pre-warp, overlays, metrics, report  

## Shared module

All board / detect / match / H / validation ownership:

`procam-calibration/procam_calibrate/charuco.py`

Canonical board: `DICT_4X4_50`, squares `(10, 7)`, squareLength `1.0`,
markerLength `0.7`, margin `20`, RANSAC threshold `3.0`.

## Coordinates

| Name | Direction |
|------|-----------|
| `H_cam_from_proj` | ProjFB pixels → CamImg pixels |
| `H_proj_from_cam` | CamImg pixels → ProjFB pixels |
| `desired_target` | CamImg pixels per ChArUco corner ID |
| `pre_warped.png` / `prewarp_*.png` | ProjFB |

See `docs/COORDINATE_SYSTEMS.md` and `docs/METRIC_DEFINITIONS.md`.

## State machine (production)

```
PREFLIGHT
→ DISCOVER_CAMERA
→ DISCOVER_PROJECTOR
→ HOMOGRAPHY_CALIBRATE
→ MEASURE_RESIDUAL   # includes independent-target measure / optional ROI recapture
→ FINAL_REPRODUCTION
→ DONE
```

Blocked states: `BLOCKED_EXTERNAL`, `ACCEPTANCE_FAILED`.

**Not in production:** `FIT_CSPR_SCENE`, neural train/infer, GS training.

## Artifacts (minimum)

```text
runs/<run_id>/
  homography_baseline/
    patterns/charuco_calib.png
    captures/charuco_uncorrected.png
    captures/charuco_corrected.png
    proj_corner_ids.json
    board_spec.json
    H_cam_from_proj.npy
    H_proj_from_cam.npy
    desired_target.json
    prewarp_charuco.png
  calibration/homography/          # production copies
  prewarps/prewarp_best.png
  metrics.json
  report.md
```

## Out of scope

- CSPR-Net training or inference for flat wall  
- GS-ProCams as a dependency  
- Non-planar surfaces  
- New ML models  
