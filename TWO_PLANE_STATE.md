# TWO_PLANE_STATE

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

Do not modify or weaken flat-wall production artifacts from two-plane work.

## Two-plane piecewise homography

**USER_ACTION_REQUIRED**

| Phase | Status |
|-------|--------|
| Two-homography model fit | implemented |
| Seam model (straight) + exclusion band | implemented |
| Exclusive masks + piecewise pre-warp | implemented |
| Multi-frame observation keys | implemented |
| Physical state machine | implemented |
| Coupled residual refinement | implemented |
| Synthetic acceptance gates | **PASS** (14/14) |
| CLI `--mode synthetic\|physical` | implemented |
| Physical setup | **USER_ACTION_REQUIRED** |
| Physical capture / validation | pending user hardware placement |

## Commands

```bash
# software only (no devices)
procam-calibrate auto-two-plane --run-dir <NEW_RUN_DIR> --mode synthetic

# physical (stops at setup request until confirmed)
procam-calibrate auto-two-plane --run-dir <NEW_RUN_DIR> --mode physical

# after user confirms fold / devices placed
procam-calibrate auto-two-plane --run-dir <NEW_RUN_DIR> --mode physical --setup-confirmed
```

Immutable synthetic run: `procam-test-data/runs/two_plane_synthetic_20260717`
