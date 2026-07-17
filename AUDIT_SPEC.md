You are auditing two local research repositories related to automatic
projector-camera calibration and projection compensation:

1. CSPR-Net
2. GS-ProCams

Do not modify their implementation during this first phase.

The practical target is an automatic projection-mapping prototype in which:

- an iPhone camera continuously observes the projector output;
- the projector displays calibration or correspondence patterns;
- the system estimates the mapping between projector pixels and camera pixels;
- the system generates a projector-space pre-warp;
- a rectangular test image appears geometrically correct on:
  1. a flat wall;
  2. an angled planar surface;
  3. a cube or simple non-planar object.

The final system may combine methods from both repositories, but you must not
assume that either repository directly solves the complete problem.

FIRST PHASE: AUDIT ONLY

Inspect both repositories completely before proposing implementation changes.

For each repository, document:

1. Research objective.
2. Paper and implementation correspondence.
3. Supported operating systems and hardware.
4. Required Python, CUDA, PyTorch and compiled dependencies.
5. Required checkpoint files and datasets.
6. Whether training is required or pretrained inference is available.
7. Exact executable entry points.
8. Exact input formats.
9. Exact output formats.
10. Camera assumptions.
11. Projector assumptions.
12. Calibration assumptions.
13. Number of images, frames, views or projected patterns required.
14. Whether camera intrinsics must already be known.
15. Whether projector intrinsics must already be known.
16. Whether camera-projector extrinsics must already be known.
17. Whether geometric correction is produced.
18. Whether radiometric compensation is produced.
19. Whether a projector-space UV warp or lookup table is produced.
20. Whether the output is usable in real time.
21. Expected execution time and memory.
22. Missing code, undocumented assets or reproduction blockers.
23. License restrictions.
24. Components that can be reused independently.

Create:

- docs/CSPR_NET_AUDIT.md
- docs/GS_PROCAMS_AUDIT.md
- docs/COMPONENT_CROSSWALK.md
- docs/INPUT_OUTPUT_CONTRACTS.md
- docs/REPRODUCTION_PLAN.md
- docs/IPHONE_CAPTURE_PLAN.md
- docs/MVP_ARCHITECTURE.md
- docs/BENCHMARK_PLAN.md
- docs/RISK_REGISTER.md
- PROJECT_STATE.md

COMPONENT_CROSSWALK.md must use this table:

Feature |
CSPR-Net implementation |
GS-ProCams implementation |
Classical CV alternative |
Required for MVP |
Reuse / adapt / replace |
Evidence: file and line or symbol

INPUT_OUTPUT_CONTRACTS.md must identify concrete tensor shapes, image dimensions,
coordinate conventions, camera coordinate systems, projector coordinate systems,
normalization ranges and file formats.

REPRODUCTION_PLAN.md must contain exact commands for installing and running each
repository unchanged. Do not invent commands. Derive them from the repository.

MVP_ARCHITECTURE.md must propose the smallest system that can produce an
automatic geometric warp on a flat wall using one projector and one iPhone.

Prefer classical structured-light correspondence, homography and camera-projector
geometry when they solve a stage more reliably than a neural model.

Do not add DINO, SAM, MoGe, EdgeTAM, BiRefNet or other models unless the audit
demonstrates a concrete missing requirement that they solve.

The iPhone may be connected through:
- a local video stream;
- a recorded video;
- still photographs;
- or a native capture application.

During the first MVP, use whichever input is easiest to make deterministic.
Do not begin with wireless real-time streaming if a recorded or USB capture
provides a more reproducible test.

At the end, provide:

1. Which repository can run immediately.
2. Which repository cannot run and why.
3. Which one is closest to the MVP.
4. Which components should be combined.
5. The smallest implementation milestone.
6. The exact first coding task.
7. Unresolved questions that require human input.

Stop after documentation. Do not implement the combined system yet.