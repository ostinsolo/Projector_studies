# Two-Plane Validation

## Independent desired target

`desired_target_two_plane.json` is fixed **before** analysing the corrected capture. Metrics use one `evaluation_keys` population.

## Gates

| Metric | Threshold |
|--------|-----------|
| Per-plane + global median | ≤ 5 cam px |
| Per-plane + global p95 | ≤ 12 cam px |
| Seam median / p95 | ≤ 3 / ≤ 6 cam px |
| Coverage | ≥ 0.80 |
| Corrected vs uncorrected | must improve |
| Improvement vs burst jitter | required |

Empty/invalid measurements must not report zero error.

## Refinement

Coupled A+B candidates at strengths 1.0/0.75/0.5/0.25 via control-point interpolation; joint rollback; max 4 physical iters.
