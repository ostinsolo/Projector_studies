# Two-Plane Metrics

## Model selection

| Metric | One-plane accept |
|--------|------------------|
| median reprojection | ≤ 2 px |
| p95 reprojection | ≤ 5 px |

If one H fails these, evaluate the two-H model.

## Two-H validity

| Requirement | Threshold |
|-------------|-----------|
| Points per plane | ≥ 12, well spread |
| Convex-hull area | > 0 |
| Spatial contiguity | ≥ 90% NN-connected |
| Interleaving | rejected (few label flips along centroid axis) |
| Robust error vs one-H | two-H < 0.55 × one-H robust error |

## Per-plane fit

| Metric | No noise | With expected noise |
|--------|----------|---------------------|
| median reprojection | ≤ 1 px | ≤ 2 px |

## Physical target accuracy (absolute gate)

| Scope | median | p95 |
|-------|--------|-----|
| Plane A / B independently | ≤ 5 cam px | ≤ 12 cam px |
| Combined global | ≤ 5 cam px | ≤ 12 cam px |
| Seam mismatch | ≤ 3 cam px | ≤ 6 cam px |
| Coverage per plane | ≥ 0.80 vs expected-visible | |

Corrected must beat uncorrected; improvement must exceed burst jitter.
Empty/invalid measurements must not report zero error.

## Seam (geometry)

| Metric | Target |
|--------|--------|
| median seam mismatch (H_A vs H_B on seam) | ≤ 3 camera px |
| p95 seam mismatch | ≤ 6 camera px |

Feathering is not used until geometric seam alignment passes.

Observation keys use `"<pattern_index>:<corner_id>"` — never merge repeated
ChArUco IDs across patterns.

## Assignment

| Metric | Target |
|--------|--------|
| point→plane accuracy (synthetic, known GT) | ≥ 95% (noise-free); ≥ 92% with mild noise/outliers |

## Masks

- No overlapping double-projection pixels
- No unassigned holes in the projector framebuffer
