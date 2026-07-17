# Absolute Accuracy Plan

Goal: drive the validated flat-wall homography pipeline from relative acceptance
to absolute target thresholds without changing the MVP method.

## Absolute thresholds

- corrected target median ≤ **5** camera px
- corrected target p95 ≤ **12** camera px
- coverage gate pass (expected-visible mask)
- geometry gate pass (axis / orthogonality)

## Scope

In scope:

1. Consistent `evaluation_ids` for all production target metrics
2. Residual field diagnosis (homography-only models)
3. Persist exact forward pre-warp `H_proj_from_source_current`
4. Conservative residual homography refine with strengths + rollback
5. Burst capture variability
6. State-machine integration (not CSPR)

Out of scope (deferred):

- Non-planar / two-plane scenes
- CSPR-Net refine
- GS-ProCams

## Pipeline

```
HOMOGRAPHY_CALIBRATE
  → MEASURE_CORRECTED_TARGET
  → DIAGNOSE_PLANAR_RESIDUAL
  → REFINE_HOMOGRAPHY
      PROJECT_REFINED → CAPTURE_REFINED → MEASURE_REFINED → ACCEPT_OR_ROLLBACK
  → FINAL_REPRODUCTION
```

If absolute gate already passes after calibrate, refine is skipped.

## Update rule

Do not lerp homography coefficients. Interpolate projector-space control points
between current and full-candidate pre-warps, then refit H.

Candidate strengths: 1.0, 0.75, 0.5, 0.25. Max 4 physical iterations.
Accept only when median **and** p95 improve beyond measured jitter, coverage
passes, and axis/ortho do not regress beyond 0.25° / 0.5°.

## Termination

| Code | Meaning |
|------|---------|
| A | Absolute pass |
| B | Converged physical limit (document; do not fake absolute) |
| C | Failure (detection / stale / convention / repeated regression) |

## Commands

```bash
procam-calibrate auto-wall --run-dir <NEW_RUN_DIR>
procam-calibrate remeasure-v2 --run-dir <RUN_DIR>
procam-calibrate refine-homography --run-dir <RUN_DIR>
```
