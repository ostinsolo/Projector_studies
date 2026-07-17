# PROJECT_STATE

## Flat-wall MVP

**DONE / VALIDATED / ABSOLUTE PASS**

## Absolute-accuracy enhancement

**COMPLETE**

## Production

ChArUco + planar homography + optional planar residual refinement

## Best physical runs

| Role | Run ID |
|------|--------|
| Absolute-pass calibrate | `flatwall_abs_20260717_165506` |
| Absolute-pass refine | `flatwall_abs_refine_20260717_165533` |

## Best corrected results

| Metric | Value |
|--------|-------|
| target_median_err_px | **0.34** |
| target_p95_err_px | **0.64** |

See `ACCURACY_STATE.md` for full before/after and residual diagnosis.

## Two-plane piecewise homography

**USER_ACTION_REQUIRED**

- Flat-wall status above is unchanged and must not be replaced by two-plane state
- Command: `procam-calibrate auto-two-plane --run-dir <NEW_RUN_DIR> --mode physical`
- State: `TWO_PLANE_STATE.md`
- Synthetic suite: **PASS** (`docs/TWO_PLANE_SYNTHETIC_RESULTS.md`)
- Physical setup: **USER_ACTION_REQUIRED** (`docs/TWO_PLANE_PHYSICAL_SETUP_REQUEST.md`)

## Experimental

| Component | Status |
|-----------|--------|
| CSPR-Net | experimental non-planar only |
| GS-ProCams | `REUSABLE_COMPONENTS_ONLY` |
