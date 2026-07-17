# Refinement Results

## Distinction of outcomes

| Layer | Result |
|-------|--------|
| MVP validation | **VALIDATED** (immutable live + canonical) |
| Relative acceptance | **PASS** on MVP live (`relative_gate_pass: true`) |
| Absolute accuracy | **PASS** on new physical runs |
| Measurement variability | refine burst median jitter ≈ **0.29 px** |
| Best physical corrected | median **0.34 px**, p95 **0.64 px** |

## Experiment A — fresh calibrate (absolute pass)

Run: `procam-test-data/runs/flatwall_abs_20260717_165506`

- Path: `auto-wall` → `HOMOGRAPHY_CALIBRATE` → absolute gate already met → `FINAL`
- Corrected: median **0.43**, p95 **0.81**
- Coverage: 29/33 expected-visible (fraction 0.88)
- Forward matrix saved: `homography_baseline/H_proj_from_source_current.npy`

Refine loop not entered (already absolute-pass).

## Experiment B — residual refine from live baseline

Run: `procam-test-data/runs/flatwall_abs_refine_20260717_165533`  
Prepared by copying immutable `flatwall_live_20260717_154312` (source untouched).

| Iteration | Strength | Accepted | median | p95 | reason |
|-----------|----------|----------|--------|-----|--------|
| 0 | — | baseline | 20.76 | 24.61 | class=translation |
| 1 | 1.0 | **yes** | **0.34** | **0.64** | accepted (jitter 0.29 px) |

Termination: **A — ABSOLUTE_PASS**

Artifacts:

- `homography_refinement/refinement_result.json`
- `homography_refinement/diagnosis/` (residual vectors / overlay / histogram)
- `homography_refinement/H_proj_from_source_best.npy`
- `homography_refinement/prewarp_best.png`
- `homography_refinement/iter_01/burst_*` (timestamps + variability JSON)

## Experiment C — aborted first calibrate (coverage mask)

Run: `flatwall_abs_20260717_165403`  
Corrected numeric errors were already sub-pixel, but coverage used full 54 IDs
because expected-visible was not passed into live measure. Fixed in production
path; superseded by Experiment A.

## Immutable runs (unchanged originals)

- `flatwall_20260717_141610`
- `flatwall_live_20260717_154312`

Recalculated only under each run’s `analysis_remeasure_v2/`.
