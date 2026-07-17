# External Repository Matrix

Audit date: 2026-07-17. Clones under `research_external/` (see `CLONE_MANIFEST.md`).

| ID | Repository | Paper/project | Tip commit | License | Lang | OS | Depth cam | CUDA | MATLAB | Win/Kinect | Manual pts | Multi-plane | Auto unknown planes | Auto seam | Dense corr | Distortion | Radiometric | Class |
|----|------------|---------------|------------|---------|------|----|-----------|------|--------|------------|------------|-------------|---------------------|-----------|------------|------------|-------------|-------|
| R1 | [kamino410/procam-calibration](https://github.com/kamino410/procam-calibration) | Moreno & Taubin 2012 | `09e58d5` 2022-03 | MIT | Python | any + OpenCV contrib | No | No | No | No | Chessboard auto | Multi-pose board only | No | No | Gray-code | Yes (stereo) | No | **D** |
| R2 | [BingyaoHuang/single-shot-pro-cam-calib](https://github.com/BingyaoHuang/single-shot-pro-cam-calib) | T-ASE 2020 / ISMAR 2018 | `cd7fda6` 2021-05 | Academic noncommercial | MATLAB | Win/MATLAB | Optional RealSense | No | **Yes** | No | SL auto | Multi-pose | No | No | Color SL | Yes + BA | No | **E**/H |
| R3 | [microsoft/RoomAliveToolkit](https://github.com/microsoft/RoomAliveToolkit) | RoomAlive MSR | `51a39f1` 2022-11 | MIT | C# | **Windows** | **Kinect required** | No | No | **Yes** | Toolkit UI | Room surfaces via depth | Depth mesh | N/A | Depth+proj | Yes | Partial | **H** |
| R4 | [YCAMInterlab/ProCamToolkit](https://github.com/YCAMInterlab/ProCamToolkit) | YCAM / mapamok | `09e17be` 2013-10 | MIT | C++/oF | macOS possible | No | No | No | No | Often model/manual | Scene model | Limited | No | Gray-code apps | Partial | No | **K**/C |
| R5 | [bytedeco/procamcalib](https://github.com/bytedeco/procamcalib) | ProCamCalib | `0361086` 2023-06 | **GPL-2.0** | Java | Cross | No | No | No | No | Chessboard | Multi-pose | No | No | Patterns | Yes | No | **J** |
| R6 | [KeckCAVES/SARndbox](https://github.com/KeckCAVES/SARndbox) | AR Sandbox | `d1e51b5` 2019-06 | GPL-2 | C++ | Linux/mac | Kinect | No | No | Kinect | Sandbox calib | Terrain | Depth | No | Depth | — | No | **H**/J |
| R7 | [thomwolf/Magic-Sand](https://github.com/thomwolf/Magic-Sand) | Magic Sand | `0d26ba8` 2019-09 | GPL-2 | C++/oF | Win/mac | Kinect | No | No | Kinect | Sandbox | Terrain | Depth | No | Depth | — | No | **H**/J |
| R8 | [ebertobenjumea/Calibration-Multimodal…](https://github.com/ebertobenjumea/Calibration-Multimodal-System-Digital-Features) | Applied Optics 2025 | `c86d6c5` 2025-07 | **No LICENSE file** | MATLAB+Py | MATLAB | Aux cam + mirror + screen | No | **Yes** | No | Digital features | Not wall fold | No | No | SL/digital | Yes | No | **H**/C |
| R9 | `GS-ProCams/` (local) | ISMAR/TVCG 2025 | local | Academic noncommercial + 3DGS | Python | CUDA host | No | **Yes** | No | Often Win build | COLMAP etc. | Arbitrary via GS | Mesh-like | N/A | Learned | Implicit | Yes | **I**/J |
| R10 | `CSPR-Net/` (local) | CSPR-Net | local | see tree | Python | CUDA preferred | No | Preferred | No | No | Training pairs | Nonplanar | N/A | N/A | Neural | Implicit | Yes | **L** (out of milestone) |

## Classification key

A DROP_IN · B MAJOR_BASE · C REUSABLE_CALIBRATION · D REUSABLE_STRUCTURED_LIGHT · E REUSABLE_OPTIMIZATION · F REUSABLE_PLANE_OR_SEAM · G PAPER_ONLY · H HARDWARE_INCOMPATIBLE · I PLATFORM_INCOMPATIBLE · J LICENSE_INCOMPATIBLE · K OUTDATED · L NOT_RELEVANT

## Rejected / not adopted as base

No repository is class **A** or **B** for our exact target (macOS, Continuity Camera, two unknown connected walls, automatic seam, viewpoint pre-warp, ≤5/12 px gates without Kinect/CUDA/MATLAB/GPL).

## Reusable modules (if ever hybridized)

| Source | Module | Caveat |
|--------|--------|--------|
| kamino410 | Gray-code encode/decode + local homography subpixel corr | Needs `opencv-contrib`; multi-pose chessboard; OpenCV 5 API shim |
| ProCamToolkit | Gray-code decode concepts (MIT) | Ancient oF stack; port ideas only |
| OpenCV contrib | `structured_light.GrayCodePattern` / sinusoidal | Prefer stock OpenCV over vendoring GPL apps |
