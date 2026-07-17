# Piecewise Pre-Warp

## Forward matrices

For each plane k:

`H_proj_from_source_plane_k = inv(H_cam_from_proj_k) @ H_desired_cam_from_source`

## Composition

1. Warp source with plane A forward H  
2. Apply exclusive mask A  
3. Warp source with plane B forward H  
4. Apply exclusive mask B  
5. Sum (no overlap / no holes)

Artifacts: `prewarps/H_proj_from_source_plane_*.npy`, `prewarp_plane_*.png`, `prewarp_piecewise.png`, forward metadata JSON.
