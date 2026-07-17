# Two-Plane Architecture Decision

**Decision: CONTINUE_CURRENT_ARCHITECTURE**

Date: 2026-07-17  
Status after audit: supersedes temporary `REUSE_AUDIT_RUNNING`  
Physical run remains paused only until this decision is accepted operationally; software base is unchanged.

## Exact target (recap)

macOS · 1920×1080 projector · iPhone Continuity Camera · fixed viewpoint · two connected planar surfaces · automatic display/capture · automatic one-H vs two-H · automatic assignment & seam · piecewise pre-warp · independent desired-target validation · residual refine · median ≤5 / p95 ≤12 / seam ≤3/6 px · no neural milestone.

## Critical design question

Should production use (1) multiple planar Hs, (2) full K/R/t, (3) 3D planes, (4) Gray-code, (5) phase-shift, (6) BA, (7) mesh, or (8) hybrid?

### Evaluation for two connected walls + one fixed camera viewpoint

| Approach | Verdict for this target |
|----------|-------------------------|
| 1. Multiple independent planar homographies | **Sufficient and preferred.** Camera-desired geometry for planar patches is a homography. Two connected walls ⇒ two Hs + seam masks. Matches Jin-style SAR multi-surface H thinking without their semi-manual reference step. |
| 2. Full camera/projector intrinsics/extrinsics | **Not required as base.** Useful later if lens-like residual dominates after H refine. Cho/Kim show the path but need their target + code. |
| 3. Explicit 3D plane equations | **Optional.** Needed for metric 3D fusion; our acceptance is in **camera pixels** for a predetermined desired 2D target. |
| 4. Gray-code dense correspondence | **Complementary, not replacement.** Validated via kamino410 sample (stereo RMS ~0.31) under opencv-contrib. Use only if multi-frame ChArUco coverage fails. |
| 5. Phase-shifting PMP | **Overkill** for geometric wallpaper correction; long sequences; Cho code unavailable. |
| 6. Target-aware BA | Valuable **on a known multi-plane target**; does not discover unknown room fold. |
| 7. Reconstructed mesh / GS | CUDA/noncommercial/depth stacks; **out of milestone**. |
| 8. Hybrid | **Only small components** (e.g. OpenCV GrayCode densify) if physical sparse coverage fails — still under CONTINUE_CURRENT, not ADOPT_EXTERNAL_BASE. |

## Why not ADOPT_EXTERNAL_BASE

No inspected repository covers Continuity Camera + automatic unknown two-wall discovery + seam + piecewise pre-warp + our validation/refine SM on macOS CPU without forbidden deps.

## Why not HYBRID_REUSE as primary decision

Hybrid would imply replacing core math stages now. Benchmarks show our two-H selection already beats single-H on synthetic two-plane data (suite 14/14). No external multi-plane seam/pre-warp module ran successfully against our observations. Optional Gray-code remains a **future contingency**, not a present replacement.

## Why not BLOCKED_BY_UNAVAILABLE_CODE

Cho/Kim and Jin et al. are interesting **paper-only** references, but they do not block shipping our architecture: Jin is methodologically close yet semi-manual and code-less; Cho solves a **different** problem (PMP device calibration on a fabricated target).

## Component comparison (summary)

| Component | Ours vs external |
|-----------|------------------|
| ChArUco multi-frame keys | **Ours stronger** for unique pattern:id obs |
| One-H vs two-H selection | **Ours stronger** for unknown walls (executed suite) |
| Seam + exclusive masks | **Ours stronger** — no external auto seam for this scene |
| Desired-target independence | **Ours stronger** / unique to our gate |
| Forward piecewise pre-warp | **Ours stronger** for viewpoint correction |
| Continuity + projector SM | **Ours stronger** (macOS path exists) |
| Dense Gray-code corr | **External stronger** (kamino410 / OpenCV) — complementary |
| Full distortion BA | **External/paper stronger** (Cho, Huang) — no runnable wall pipeline |
| Neural compensation | CSPR/GS — **out of scope** |

## Production directive

1. **Keep** `two_plane.py` / `auto_two_plane.py` / synthetic checkpoint.  
2. **Do not** vendor GPL or noncommercial academic trees into production.  
3. After physical setup, run the existing physical SM.  
4. If physical sparse coverage fails, consider OpenCV GrayCode densification as an additive module under a new run ID — without discarding two-H.

## Contact list (code requests — optional)

| Paper | Contact |
|-------|---------|
| Cho & Kim 2025/2026 | hjcho@vclab.kaist.ac.kr, minhkim@vclab.kaist.ac.kr |
| Jin, Seo, Han 2026 | via JCDE / ResearchGate author request |
