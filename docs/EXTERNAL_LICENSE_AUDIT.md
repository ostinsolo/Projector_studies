# External License Audit

Purpose: prevent incompatible code from entering production `procam-calibration/`.

| Source | License | Production use? | Notes |
|--------|---------|-----------------|-------|
| kamino410/procam-calibration | MIT | **Allowed** (attribution) | Prefer reimplement OpenCV GrayCode against MIT ideas; keep upstream out of tree |
| YCAMInterlab/ProCamToolkit | MIT | **Allowed** (ideas/snippets with attribution) | Do not vendor entire oF tree |
| microsoft/RoomAliveToolkit | MIT | Allowed but **hardware incompatible** | Kinect/Windows |
| bytedeco/procamcalib | **GPL-2.0** | **No** (without dual-license decision) | Contaminating risk if linked |
| SARndbox | GPL-2 | **No** | |
| Magic-Sand | GPL-2 | **No** | |
| BingyaoHuang/single-shot-pro-cam-calib | Academic **noncommercial** | **No** for general production | Research-only |
| GS-ProCams | Academic noncommercial + 3DGS restrictions | **No** for production milestone | Also CUDA |
| ebertobenjumea/…Digital-Features | **Missing LICENSE** | **No** until authors clarify | Do not copy |
| Jin / Cho papers | Paper copyright / CC-BY-NC (Jin abstract notes CC BY-NC 4.0 on RG) | Algorithm reimplementation from paper text is legally distinct from copying code; **no code available** | |

## Decision for this repository

- Do **not** introduce GPL or noncommercial-academic code into production.
- Prototype adapters may **call** MIT upstream from `research_external/` without copying into `procam_calibrate/`.
- Any future Gray-code densification should use **OpenCV contrib APIs** + our own thin wrapper (BSD/Apache-compatible OpenCV), not vendored GPL apps.
