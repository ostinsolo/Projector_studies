# PROJECT_STATE

## Flat-wall MVP

**DONE / VALIDATED**

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

## Experimental

| Component | Status |
|-----------|--------|
| CSPR-Net | experimental non-planar only |
| GS-ProCams | `REUSABLE_COMPONENTS_ONLY` |
