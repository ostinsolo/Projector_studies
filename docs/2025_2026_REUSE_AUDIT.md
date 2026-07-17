# 2025–2026 Reuse Audit — Two-Plane Projector–Camera Calibration

**Final decision: CONTINUE_CURRENT_ARCHITECTURE**

Audit completed: 2026-07-17  
Physical two-plane run: **not started** during this audit  
Existing synthetic + physical software: **preserved**

Companion documents:

| Doc | Content |
|-----|---------|
| [EXTERNAL_REPOSITORY_MATRIX.md](EXTERNAL_REPOSITORY_MATRIX.md) | Repo forensics + classification |
| [RECENT_PAPER_METHOD_MATRIX.md](RECENT_PAPER_METHOD_MATRIX.md) | Primary 2025–2026 papers |
| [TWO_PLANE_ARCHITECTURE_DECISION.md](TWO_PLANE_ARCHITECTURE_DECISION.md) | Design choice + rationale |
| [EXTERNAL_LICENSE_AUDIT.md](EXTERNAL_LICENSE_AUDIT.md) | License gate |
| [REUSE_BENCHMARK_RESULTS.md](REUSE_BENCHMARK_RESULTS.md) | Executed evidence |

## Mission result (one sentence)

No paper or repository provides a drop-in or major-base replacement for our macOS Continuity + two-unknown-wall piecewise-homography pipeline; retain the current architecture and import only small MIT/OpenCV structured-light pieces later if sparse ChArUco coverage fails.

## Search record (credible candidates)

Searches included: multi-plane procam, SAR automatic calibration, piecewise/surface-specific homography, structured light / PMP calibration, multi-H RANSAC, bundle adjustment procam, GitHub 2025/2026 procam, project pages, Zenodo/OSF (no Cho/Jin code found), KAIST VCLAB org, author GitHubs.

| Candidate | Disposition |
|-----------|-------------|
| Jin et al. JCDE 2026 | G — multi-surface H; semi-auto; **no code**; reported 3.78–8.53 px |
| Cho & Kim VISAPP 2025 + SNCS 2026 | G — strong BA+phase+multi-plane **target**; **no code** |
| Benjumea Applied Optics 2025 + GitHub | H/C — MATLAB screen/mirror multimodal |
| kamino410/procam-calibration | D — Gray-code+stereo; **ran** sample (RMS≈0.31) |
| BingyaoHuang/single-shot-pro-cam-calib | E/H — MATLAB noncommercial |
| RoomAliveToolkit | H — Kinect+Windows |
| ProCamToolkit | K/C — MIT graycode/mapamok; 2013 |
| bytedeco/procamcalib | J — GPL-2 |
| SARndbox / Magic-Sand | H/J — Kinect sandbox GPL |
| GS-ProCams (local) | I/J — CUDA + noncommercial |
| CSPR-Net (local) | L — out of milestone |

## Executed evidence (not README-only)

1. Cloned 8 upstream repos under `research_external/` (SHAs in `CLONE_MANIFEST.md`).  
2. Ran kamino410 **sample_data** via OpenCV-5 adapter → stereo RMS **0.311** (logs/results under `prototypes/external_reuse/`).  
3. Replayed our synthetic two-plane suite → **14/14 PASS**; single-H fails hard on two-plane scenes.  
4. Confirmed Cho/Jin code absence via project pages, GitHub org/search, DOI landing pages.  
5. Confirmed default macOS OpenCV wheel lacks `structured_light` (contrib required for GrayCode).

## Architecture answer

For two connected walls, fixed devices, one camera viewpoint, automatic operation, macOS CPU, low-pixel gates:

→ **Two (or more) planar homographies + automatic seam/masks + independent desired target + coupled H refine** is the correct production model.

Full metric PROCAM + phase PMP is a **different product** (3D scanner calibration). SAR multi-surface H papers validate the family but do not supply a better runnable stack.

## Next operational step

Resume physical two-plane setup (`USER_ACTION_REQUIRED` checklist in `docs/TWO_PLANE_PHYSICAL_SETUP_REQUEST.md`) when ready — **without** rewriting the pipeline for unavailable external bases.
