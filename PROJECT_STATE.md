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

**SOFTWARE_READY_FOR_PHYSICAL_VALIDATION** · architecture **CONTINUE_CURRENT_ARCHITECTURE**

- Flat-wall status above is unchanged and must not be replaced by two-plane state
- Reuse audit: `docs/2025_2026_REUSE_AUDIT.md` (complete — no external drop-in)
- Architectural priors + extended synthetic suite integrated
- State: `TWO_PLANE_STATE.md`
- Synthetic: core 14/14 + extended 25/25; package tests 71 passed
- Physical: awaiting hardware — `docs/TWO_PLANE_PHYSICAL_RUNBOOK.md`

## Experimental

| Component | Status |
|-----------|--------|
| CSPR-Net | experimental non-planar only |
| GS-ProCams | `REUSABLE_COMPONENTS_ONLY` |
