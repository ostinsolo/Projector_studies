# Two-Plane Architecture

**Decision:** CONTINUE_CURRENT_ARCHITECTURE (extended with architectural priors)

## Pipeline (production)

```
CAPTURE_EMPTY_SCENE
→ ANALYSE_ARCHITECTURE (prior)
→ PROPOSE_TARGET_REGION
→ VERIFY_TWO_PLANE_SETUP (preflight patterns)
→ GENERATE / PROJECT / CAPTURE calibration (multi-frame ChArUco)
→ EXTRACT_CORRESPONDENCES
→ FIT_SINGLE_H → TEST_MULTI_PLANE → FIT_TWO_H → ASSIGN_SURFACES
→ ESTIMATE_SEAM → VERIFY_SEAM (prior vs correspondence)
→ BUILD_MASKS → DEFINE_DESIRED_TARGET → GENERATE_PIECEWISE_PREWARP
→ uncorrected / corrected validation bursts
→ MEASURE → DIAGNOSE → REFINE (coupled) → ACCEPT_OR_ROLLBACK
→ FINAL_REPRODUCTION → DONE
```

## Modules

| Module | Role |
|--------|------|
| `architecture_analysis.py` | Classical lines/VP/seam prior/target region |
| `two_plane_observations.py` | Multi-frame observation keys |
| `two_plane.py` | Dual-H, seam, masks, desired, metrics |
| `two_plane_refine.py` | Coupled residual refine |
| `coordinate_conventions.py` | Naming + composition guards |
| `auto_two_plane.py` | Physical/offline state machine |
| `synthetic_two_plane.py` | Core 14 + extended synthetic suite |

## CLI

```bash
procam-calibrate auto-two-plane --run-dir <DIR> --mode synthetic
procam-calibrate auto-two-plane --run-dir <DIR> --mode physical --setup-confirmed
```

## Non-goals

CSPR, GS-ProCams, full PMP BA, replacing flat-wall production.
