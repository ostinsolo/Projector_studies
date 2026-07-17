# Two-Plane Physical Runbook

## Status prerequisite

For **two-wall geometry validation**: `TWO_PLANE_STATE.md` ≥ `SOFTWARE_READY_FOR_PHYSICAL_VALIDATION`.

For **oblique maximum video + ceiling exclusion**: `TWO_PLANE_STATE.md` must read **SOFTWARE_READY_FOR_OBLIQUE_VIDEO_VALIDATION** (or later).

Do not start until projector + Continuity Camera + two-wall fold are placed. Read `docs/PHYSICAL_OBLIQUE_SETUP.md` for oblique placement.

## Placement checklist

1. Two rigid matte light surfaces / walls, clear vertical corner  
2. Interior angle ~60–100°  
3. Fold in the **centre third** of the projector image (preferred)  
4. Meaningful light on **both** planes  
5. Ceiling may appear in the upper projector coverage — that is OK  
6. Projector fixed (oblique OK); iPhone fixed at **audience** position  
7. Full projection + fold visible in camera  
8. Room dark enough for ChArUco  
9. **Disable** macOS display mirroring; use extended display  
10. Confirm external display is 1920×1080 framebuffer  

## Exact command (new run directory)

```bash
cd /Users/ostino/Documents/Code/Projector_studies
procam-calibrate auto-two-plane \
  --run-dir procam-test-data/runs/two_plane_oblique_$(date +%Y%m%d_%H%M%S) \
  --mode physical \
  --setup-confirmed
```

Use a **new** run directory. Never overwrite `two_plane_synthetic_20260717` or flat-wall absolute-pass runs.

## After calibration — diagnostic then user video

```bash
procam-calibrate play-video \
  --calibration-run <RUN_DIR> \
  --diagnostic \
  --loop

procam-calibrate play-video \
  --calibration-run <RUN_DIR> \
  --input <VIDEO_FILE> \
  --loop \
  --video-fit contain \
  --alignment camera
```

## What the system will do

Empty-scene → architecture prior → multi-frame ChArUco → active/excluded plane classification (ceiling peel) → dual-H → seam → wall–ceiling boundary → usable/exclusion masks → maximum inscribed video rectangle → piecewise video remap bundle → pre-warp validation → metrics / refine → report.

## User must not

Capture manually, click points, draw the seam, edit JSON, move devices after start, or run ad-hoc Python.

## Expected artifacts

See `docs/TWO_PLANE_ARCHITECTURE.md`, plus:

- `excluded_geometry/`
- `video_region/`
- `video_playback/`
- `diagnostic_video/`
