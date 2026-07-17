# Two-Plane Physical Runbook

## Status prerequisite

`TWO_PLANE_STATE.md` must read **SOFTWARE_READY_FOR_PHYSICAL_VALIDATION** (or later).

Do not start until projector + Continuity Camera + two-wall fold are placed.

## Placement checklist

1. Two rigid matte light surfaces / walls, clear vertical corner  
2. Interior angle ~60–100°  
3. Fold in the **centre third** of the projector image  
4. Meaningful light on **both** planes  
5. Projector fixed; iPhone fixed  
6. Full projection + fold visible in camera (offset OK)  
7. Room dark enough for ChArUco  
8. **Disable** macOS display mirroring; use extended display  
9. Confirm external display is 1920×1080 framebuffer  

## Exact command

```bash
cd /Users/ostino/Documents/Code/Projector_studies
procam-calibrate auto-two-plane \
  --run-dir procam-test-data/runs/two_plane_phys_$(date +%Y%m%d_%H%M%S) \
  --mode physical \
  --setup-confirmed
```

Use a **new** run directory. Never overwrite `two_plane_synthetic_20260717` or flat-wall absolute-pass runs.

## What the agent/system will do

Empty-scene capture → architecture prior → preflight patterns → multi-frame ChArUco → one/two-H → seam verify → piecewise pre-warp → uncorrected/corrected validation bursts → metrics → coupled refine ≤4 → final report.

## User must not

Capture manually, click points, draw the seam, edit JSON, move devices after start, or run ad-hoc Python.

## Expected artifacts

See `docs/TWO_PLANE_ARCHITECTURE.md` artifact list under the new `run_dir/`.
