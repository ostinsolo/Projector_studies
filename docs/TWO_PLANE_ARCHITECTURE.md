# Two-Plane Architecture

Separate from the validated single-plane flat-wall production path.

## Goal

Automatically correct one projected image spanning two connected planar surfaces
(e.g. two walls meeting at a corner), viewed by one fixed iPhone.

## Pipeline

```
PREFLIGHT / device discovery / physical preflight patterns
→ multi-frame ChArUco sequence (unique pattern_index:corner_id keys)
→ FIT_SINGLE_H → FIT_TWO_H → model selection
→ seam + exclusion band + exclusive masks
→ independent desired_target_two_plane
→ piecewise forward pre-warp (no feathering until seam gate)
→ uncorrected / corrected validation bursts
→ per-plane + seam + global metrics
→ coupled residual refinement (≤4 iters) or accept
```

## CLI

```bash
procam-calibrate auto-two-plane --run-dir <DIR> --mode synthetic   # no hardware
procam-calibrate auto-two-plane --run-dir <DIR> --mode physical    # projector + iPhone
# after USER_ACTION_REQUIRED setup:
procam-calibrate auto-two-plane --run-dir <DIR> --mode physical --setup-confirmed
```

Deprecated aliases: `--synthetic-only`, `--allow-physical` (conflicts rejected).

## Modules

| Module | Role |
|--------|------|
| `two_plane.py` | fit, seam, masks, desired target, metrics, acceptance |
| `two_plane_observations.py` | multi-frame observation keys + pattern sequence |
| `two_plane_refine.py` | coupled homography-only refinement + rollback |
| `auto_two_plane.py` | physical state machine |
| `synthetic_two_plane.py` | synthetic suite |

## Matrices

- `H_cam_from_proj_plane_A` / `_B`
- `H_desired_cam_from_source` (predetermined; not from corrected capture)
- `H_proj_from_source_plane_A` / `_B` (OpenCV forward point maps)

## State isolation

- Run state: `run_dir/two_plane_state.json`
- Live status: `run_dir/TWO_PLANE_STATE_LIVE.md` (optional `--two-plane-state-path`)
- Never writes root `PROJECT_STATE.md` unless explicitly configured (CLI default: no)

## Non-goals

- CSPR / GS-ProCams
- Soft alpha blending as a substitute for geometry
- Modifying flat-wall production state or immutable runs
