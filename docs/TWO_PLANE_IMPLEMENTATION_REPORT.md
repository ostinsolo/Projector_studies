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
| Correspondences | Unique `pattern:id` keys |
| Plane count / assign | `fit_two_homographies` + canonicalize |
| Seam + verify | `estimate_straight_seam` + arch prior verify |
| Masks / desired / prewarp | `two_plane.py` + SM |
| Validate / refine | metrics + `two_plane_refine` |
| CLI modes | `--mode synthetic\|physical` |

## New modules

- `architecture_analysis.py`
- `coordinate_conventions.py`
- Extended suite in `synthetic_two_plane.run_extended_synthetic_suite`

## Tests (executed)

`pytest tests/` → **71 passed** (includes flat-wall + two-plane + architecture + CLI)

Synthetic: core **14/14**, extended **25/25**.

## Not claimed

Physical VALIDATED — hardware run not executed in this session.
