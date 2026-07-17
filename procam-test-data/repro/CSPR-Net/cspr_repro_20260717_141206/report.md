# CSPR-Net M1 Reproduction Report

## Status: PASS

CSPR-Net inference reproduced unchanged on macOS arm64 CPU.

## Evidence

- Weights present and loaded (`Models loaded successfully`)
- Generated projector-space pre-warps:
  - `prewarps/sim_pre_warped.png`
  - `prewarps/exp_pre_warped.png`
  - `prewarps/exp_pre_warped_test.png`
- Simulation pre-warp metrics vs ray-trace GT: SSIM 0.9243, PSNR 22.74

## Runtime

- Sim inference ~19.4s wall (CPU)
- Exp inference ~8.1s wall (CPU, chunked)

## Conclusion

Geometric pre-warp path works with bundled checkpoints. Ready for integration wrapping.
