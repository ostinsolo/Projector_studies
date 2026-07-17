# Seam Estimation

## Sources

1. **Architectural prior** — long vertical lines from empty-scene analysis (`architecture_analysis.py`)
2. **Correspondence seam** — boundary between plane A/B assignments (`estimate_straight_seam`)

## Correspondence seam (authoritative)

- Midpoint along A↔B centroid axis
- Axis-aligned normal (vertical or horizontal fold MVP)
- Offset optimized so H_A and H_B agree along the line
- Bootstrap stability + seam exclusion band for refitting

## Prior verification

`verify_seam_with_architecture_prior` records agreement; does **not** override correspondence.

## Masks

Exclusive projector masks: `A ∩ B = ∅`, `A ∪ B = full framebuffer`. No feathering until geometric seam gate passes.
