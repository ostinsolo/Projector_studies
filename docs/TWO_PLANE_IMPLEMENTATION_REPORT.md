# Two-Plane Implementation Report

Date: 2026-07-17  
Architecture decision: **CONTINUE_CURRENT_ARCHITECTURE** (extended, not replaced)

## Implemented stages

| Stage | Implementation |
|-------|----------------|
| CAPTURE_EMPTY_SCENE | `auto_two_plane.step_capture_empty_scene` |
| ANALYSE_ARCHITECTURE | `architecture_analysis.analyse_architecture` |
| PROPOSE_TARGET_REGION | `propose_target_region` + SM stage |
| Calibration sequence | Multi-frame ChArUco (`two_plane_observations`) |
| DETECT_ACTIVE_AND_EXCLUDED_PLANES | `excluded_geometry.classify_active_and_excluded_planes` |
| Plane assign / dual-H | Ceiling-aware peel then `fit_two_homographies` fallback |
| Seam + verify | `estimate_straight_seam` + arch prior verify |
| ESTIMATE_WALL_CEILING_BOUNDARY | `estimate_wall_ceiling_boundary` |
| BUILD_USABLE_PROJECTION_MASK | `build_usable_and_exclusion_masks` |
| OPTIMISE_MAXIMUM_VIDEO_REGION | `video_region.maximum_inscribed_rectangle` |
| BUILD_PIECEWISE_VIDEO_WARP | `video_playback.prepare_playback_calibration` |
| Piecewise prewarp / validate / refine | `two_plane.py` + `two_plane_refine` |
| play-video CLI | `procam-calibrate play-video` |

## New modules (oblique video)

- `excluded_geometry.py` — active walls vs excluded ceiling
- `video_region.py` — max inscribed aspect rectangle
- `video_playback.py` — remap, diagnostic, projector play
- `synthetic_oblique_video.py` — ceiling / max-rect suite
- `tests/test_oblique_video.py`

## Docs

- `docs/CEILING_EXCLUSION.md`
- `docs/MAXIMUM_VIDEO_REGION.md`
- `docs/VIDEO_PLAYBACK.md`
- `docs/VIDEO_DIAGNOSTIC_VALIDATION.md`
- `docs/PHYSICAL_OBLIQUE_SETUP.md`
- Updated runbook + `TWO_PLANE_STATE.md`

## Tests (executed)

`pytest tests/` → **78+ passed** (flat-wall + two-plane + architecture + oblique video + CLI)

Synthetic: core **14/14**, extended **25/25**, oblique video suite **all_pass**.

## Adversarial self-review (2 passes)

1. Ceiling must never become a third artistic content plane — classification note + model checks.  
2. Max rectangle must not be a naive bounding box; contain ≠ stretch; leakage self-check on diagnostic prewarp.

## Not claimed

Physical **VALIDATED_OBLIQUE_VIDEO** — hardware run not executed in this session. Terminal software state: `SOFTWARE_READY_FOR_OBLIQUE_VIDEO_VALIDATION`.
