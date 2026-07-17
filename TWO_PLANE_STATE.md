# TWO_PLANE_STATE

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

Do not modify or weaken flat-wall production artifacts from two-plane work.

## Two-plane piecewise homography

**RUNNING** (software / synthetic phase)

| Phase | Status |
|-------|--------|
| Two-homography model fit | implemented |
| Seam model (straight) | implemented |
| Piecewise pre-warp | implemented |
| Synthetic acceptance gates | **PASS** |
| Physical setup | **blocked** until software gates remain green |
| Physical capture / validation | not started |

## Command

```bash
procam-calibrate auto-two-plane --run-dir <NEW_RUN_DIR> --synthetic-only
```

Physical projection/capture is not requested until synthetic gates pass and a
setup request is issued explicitly.
