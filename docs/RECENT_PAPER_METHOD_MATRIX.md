# Recent Paper Method Matrix (2025–2026 primary sources)

## 1. Jin, Seo, Han — JCDE 2026 (DOI: 10.1093/jcde/qwag009)

| Item | Finding |
|------|---------|
| Title | Spatial augmented reality (SAR)-based 3D BIM visualization through semi-automated geometric calibration of a projector–camera system |
| Code located? | **No.** No GitHub/Zenodo/OSF linked from DOI, dblp, ResearchGate abstract, or author public repos inspected. Full PDF not openly downloadable in this audit (403/HTML gate). |
| Method (from abstract + secondary sources) | Single **reference-surface** alignment, then **automatic surface-specific homographies** via **marker detection**; explicitly **avoids full PROCAM** intrinsic/extrinsic calibration. |
| Multi-surface | Yes (single-volume and multi-volume mock-ups). |
| Seam / piecewise pre-warp | Not described as automatic fold seam for one spanning image; BIM visualization on mock-ups. |
| Reported error | **3.78–8.53 px** (~1.13–2.56 mm); calib 0.3–1.0 s. |
| Manual steps | **Semi-automated** — reference alignment required. |
| Fit to our gates | Pixel errors exceed our median ≤5 / seam ≤3 targets in the reported range; method family is close (multi-H) but workflow differs. |
| Class | **G PAPER_ONLY_NO_CODE** |
| Contact path | Authors: Yixuan Jin, JoonOh Seo (HK PolyU), SangUk Han — request code via journal/ResearchGate. |

## 2. Cho & Kim — VISAPP 2025 (DOI: 10.5220/0013145400003912)

| Item | Finding |
|------|---------|
| Title | Joint Calibration of Cameras and Projectors for Multiview Phase Measuring Profilometry |
| Project page | https://vclab.kaist.ac.kr/visapp2025p2/ (PDF + supplemental verified) |
| Code located? | **No.** KAIST-VCLAB org has many repos; **none** published for this paper. GitHub search for multiview PMP Cho returned no matching implementation. |
| Method (PDF inspected) | Custom **static multi-planar target** with ArUco; plane equations; multi-plane homographies; **phase-shift** (freqs 1/4/16/64) + unwrap; **target-aware bundle adjustment**; joint camera/projector **intrinsics, extrinsics, 6th-order radial distortion**. |
| Automatic | No manual target motion; still requires their **fabricated target** in FOV, not unknown room walls. |
| Goal | Accurate **3D PMP fusion** across views — not camera-viewpoint geometric correction of a spanning content image on two walls. |
| Class | **G PAPER_ONLY_NO_CODE** (math rich; hardware/target mismatch) |
| Contact | `{hjcho,minhkim}@vclab.kaist.ac.kr` |

## 3. Cho & Kim — SN Computer Science 2026 (DOI: 10.1007/s42979-026-05176-1)

| Item | Finding |
|------|---------|
| Relation | Extended/unified journal treatment of the VISAPP static multiview PMP calibration. |
| Code | **No** public code found (same lab/project page). |
| Class | **G PAPER_ONLY_NO_CODE** |

## 4. Benjumea et al. — Applied Optics 2025 (DOI: 10.1364/ao.569536)

| Item | Finding |
|------|---------|
| Title | Calibration of multimodal 3D structured-light systems using digital features |
| Code | https://github.com/ebertobenjumea/Calibration-Multimodal-System-Digital-Features (`c86d6c5`, 2025-07-21) — **verified clone** |
| Stack | Primarily **MATLAB** + incomplete Python helpers; `.mat` datasets; screen + mirror + auxiliary camera |
| License | **No LICENSE file** in repo — treat as all-rights-reserved until clarified |
| Fit | Multimodal SL **device** calibration, not two-wall projection mapping |
| Class | **H** / **C** (OpenCV `calibrateCamera` snippets only) |

## Literature design signals (not adoption)

| Design choice | Papers favoring it | Needed for our two-wall viewpoint task? |
|---------------|-------------------|----------------------------------------|
| Multiple planar homographies | Jin 2026; Cho 2025 (on target faces) | **Yes** — we already do this |
| Full K/R/t + distortion | Cho 2025/2026; classic stereo | **Optional** — helps lens-like residual; not required if viewpoint H meets gates |
| Explicit 3D plane equations | Cho | Helpful for metric 3D; **not required** for 2D camera-desired pre-warp |
| Gray-code dense correspondence | Moreno/Taubin lineage (kamino410) | Optional densification if sparse ChArUco fails |
| Phase-shifting PMP | Cho | Overkill for geometric wallpaper correction |
| Target-aware BA | Cho; Huang single-shot | Valuable if we had their target; not wall discovery |
| Surface mesh / GS | GS-ProCams; RoomAlive | Out of scope / CUDA / depth |

## Rejected paper-only claims

| Claim | Verdict |
|-------|---------|
| “Jin et al. replaces our pipeline” | Unverified without code; semi-manual reference step; reported px worse than our absolute gate |
| “Cho/Kim is production-ready for walls” | Requires custom multi-plane **target** + phase sequence; no code |
| “2025 multimodal digital features solves Continuity+projector fold” | Wrong hardware topology (screen/mirror/aux) |
