# Automatic ProCam Calibration Agent Contract

## Mission

Complete an automatic projector-camera calibration on one flat wall using:

- one fixed projector;
- one fixed iPhone camera;
- **classical ChArUco + planar homography as the production flat-wall MVP**;
- CSPR-Net only as an experimental future path for non-planar surfaces;
- GS-ProCams only as a secondary comparison or source of reusable calibration code
  (`REUSABLE_COMPONENTS_ONLY`);
- a single shared OpenCV ChArUco module for calibration, validation, refine, and
  acceptance reporting (no divergent detectors).

The final result must automatically generate a projector-space pre-warp,
project a corrected validation board, capture the result and demonstrate
numerically that geometric distortion has been reduced using independent
projector↔camera fiducial metrics.

Do not stop after writing a plan, documentation, scaffold or partial demo.

Continue until either:

1. the flat-wall calibration acceptance gate passes;
2. physical action from the user is required;
3. an external blocker makes further progress impossible.

## Existing project state

The repository audit has already been completed.

Do not restart the full audit unless PROJECT_STATE.md identifies a concrete
missing fact.

Known conclusions (current):

- Flat-wall production MVP is **ChArUco + planar homography** in
  `procam-calibrate auto-wall` (default `--pipeline homography`).
  Status: **DONE** / validation **VALIDATED** (`VALIDATION_STATE.md`).
- Canonical run: `procam-test-data/runs/flatwall_20260717_141610`.
- Fresh validated live run: `procam-test-data/runs/flatwall_live_20260717_154312`.
- Shared module: `procam_calibrate/charuco.py` (board, detect, match, H, desired target, metrics).
- CSPR-Net is experimental non-planar only — do not retrain for flat wall.
- GS-ProCams is CUDA-oriented / multi-view and classified
  `REUSABLE_COMPONENTS_ONLY`; it must not block flat-wall completion.
- Neither source repository should be rewritten wholesale.
- Integration code belongs in this repository.
- No unrelated machine-learning models are allowed.

## Authoritative files

Read these before each work session:

1. AGENTS.md
2. PROJECT_STATE.md
3. VALIDATION_STATE.md (during hardening)
4. docs/PRODUCTION_PIPELINE_SPEC.md
5. docs/METRIC_DEFINITIONS.md
6. docs/COORDINATE_SYSTEMS.md
7. docs/IPHONE_CAPTURE_PROTOCOL.md
8. docs/MVP_ARCHITECTURE.md
9. docs/BENCHMARK_AND_ACCURACY_PLAN.md

Research-only (not production flat-wall):

- docs/CSPR_NET_CALIBRATION_AUDIT.md
- docs/GS_PROCAMS_CALIBRATION_AUDIT.md
- docs/GS_PROCAMS_MVP_FEASIBILITY.md

PROJECT_STATE.md / VALIDATION_STATE.md win when older planning documents conflict.

## Repository locations

Resolve and verify the configured paths before operating:

CSPR_NET_ROOT=/Users/ostino/Documents/Code/Projector_studies/CSPR-Net
GS_PROCAMS_ROOT=/Users/ostino/Documents/Code/Projector_studies/GS-ProCams
INTEGRATION_ROOT=<absolute path to procam-calibration>
TEST_DATA_ROOT=<absolute path to procam-test-data>

Never assume the working directory.

## Strict scope

Allowed:

- projector-camera geometric calibration;
- CSPR-Net inference or scene fitting;
- GS-ProCams reproduction or reusable calibration components;
- image capture ingestion;
- lens-distortion correction;
- camera-projector correspondences;
- homography estimation;
- structured-light decoding when genuinely required;
- projector-space pre-warp generation;
- closed-loop residual correction;
- numerical geometric validation;
- existing repository radiometric compensation;
- test automation and reproducible artifact generation.

Forbidden:

- DINO;
- MoGe;
- EdgeTAM;
- SAM;
- BiRefNet;
- YOLO;
- object recognition;
- human tracking;
- monocular depth estimation;
- generative 3D reconstruction;
- unrelated visual effects;
- training unrelated models;
- expanding to cubes or curved surfaces before flat-wall completion.

## Autonomy contract

Operate autonomously between human-action gates.

Do not ask for approval for:

- reading files;
- inspecting repositories;
- creating the integration environment;
- installing documented dependencies;
- running tests;
- fixing build or runtime errors;
- creating scripts;
- generating calibration patterns;
- analysing captures already available;
- updating documentation;
- committing coherent milestones.

Do not stop merely because:

- one command fails;
- one dependency version is incompatible;
- a first implementation is inaccurate;
- a repository README is incomplete;
- a test requires additional instrumentation.

Investigate, apply the smallest justified correction, test again and record it.

Do not repeatedly narrate plans instead of executing them.

## State machine

PROJECT_STATE.md must contain exactly one current status:

- RUNNING
- USER_ACTION_REQUIRED
- BLOCKED_EXTERNAL
- ACCEPTANCE_FAILED
- DONE

### RUNNING

Continue selecting and completing tasks without waiting for the user.

### USER_ACTION_REQUIRED

Use only when physical interaction is necessary, such as:

- positioning the projector;
- mounting or locking the iPhone;
- displaying a pattern full screen when this cannot be automated;
- capturing photographs or video;
- copying captures into the specified input directory.

Before stopping, create:

- `capture_request.md`;
- `capture_manifest.json`;
- all projector images required for the capture;
- an exact destination directory;
- an automatic capture-validation command.

The user request must contain numbered physical actions and no software work
that the agent could perform itself.

### BLOCKED_EXTERNAL

Use only for unavailable checkpoints, inaccessible datasets, incompatible
hardware, licensing restrictions or a required capability genuinely absent
from all accessible code.

Record evidence and attempted alternatives.

### ACCEPTANCE_FAILED

Use when the pipeline executes but does not yet meet accuracy requirements.

Do not stop in this state. Continue diagnosing and improving unless physical
recapture is required.

### DONE

Use only after every flat-wall acceptance criterion passes.

## Autonomous execution loop

Repeat:

1. Read PROJECT_STATE.md.
2. Verify current repository commits and environment.
3. Identify the first incomplete, unblocked task.
4. Inspect the exact code involved.
5. Define the smallest measurable change.
6. Record before implementation:
   - hypothesis;
   - exact input;
   - exact output;
   - source coordinate space;
   - destination coordinate space;
   - expected result;
   - acceptance threshold;
   - likely failure modes.
7. Implement the change.
8. Run focused tests.
9. If a test fails:
   - locate the earliest failing stage;
   - preserve logs and artifacts;
   - fix the earliest failure only;
   - rerun the focused test.
10. If focused tests pass:
    - run the current milestone gate;
    - calculate accuracy;
    - compare with the fixed baseline and previous revision.
11. Reject changes that reduce accuracy without a documented trade-off.
12. Save all commands, logs, captures, maps, images and metrics.
13. Update PROJECT_STATE.md.
14. Commit one coherent milestone when the repository is in a reproducible state.
15. Immediately continue with the next unblocked task.

Do not wait for another prompt between software-only iterations.

## Current implementation order (production)

Authoritative production path: **ChArUco + planar homography**.
See `docs/PRODUCTION_PIPELINE_SPEC.md` and `VALIDATION_STATE.md`.

### Stage P0 — Contract and synthetic gates

- Shared module `procam_calibrate/charuco.py` only.
- Synthetic matrix + coordinate round-trips must pass before trusting live runs.

### Stage P1 — Autonomous flat-wall calibration

One command:

```bash
procam-calibrate auto-wall --run-dir <NEW_RUN_DIR>
```

Stages:

1. discover display and camera  
2. display ChArUco board  
3. capture  
4. detect known projector IDs  
5. estimate `H_cam_from_proj` / `H_proj_from_cam`  
6. define and save desired camera-space target  
7. generate projector-space pre-warp  
8. display corrected validation board  
9. capture  
10. measure against independent desired target  
11. save results  

No neural training. No CSPR / GS stages in the production state machine.

### Stage P2 — Validation hardening

Follow gates in `VALIDATION_STATE.md` until **VALIDATED**.

### Experimental research (not production)

- CSPR-Net: non-planar research only (`--pipeline cspr` is non-production).
- GS-ProCams: `REUSABLE_COMPONENTS_ONLY`.

CLI research helpers (`run-cspr`, etc.) must not appear in production docs as
required flat-wall steps.

## Physical capture protocol

The flat-wall test uses production viewpoint rules:

- fixed projector;
- flat wall;
- fixed iPhone at an independently chosen observer viewpoint;
- camera FOV must fully contain the projected area (all four boundaries);
- projected area should occupy substantial camera resolution;
- moderate viewing angle recommended for the first experiment;
- placement close to the projector is optional, not required;
- pre-warp is optimized for this single camera pose only (not viewpoint-independent);
- one fixed rear-camera lens;
- no zoom changes;
- locked focus;
- locked exposure;
- locked white balance where possible;
- no portrait mode;
- no image mirroring;
- no automatic crop;
- native capture resolution;
- unchanged phone orientation throughout one run (patterns, corrected projection, validation);
- darkened or controlled ambient lighting where practical.

Record:

- projector framebuffer resolution;
- iPhone model;
- selected lens;
- capture resolution;
- orientation;
- target observer viewpoint description;
- approximate projector-wall distance;
- approximate camera-wall distance;
- approximate viewing angle;
- approximate camera-projector separation (optional metadata).

## Coordinate-system rules

Every transform must be named by direction.

Valid examples:

- `H_projector_to_camera`
- `H_camera_to_projector`
- `map_projector_to_camera_xy`
- `map_camera_to_projector_xy`
- `uv_source_to_projector_framebuffer`

Forbidden ambiguous names:

- `H`
- `transform`
- `warp`
- `mapping`

Every stored matrix or dense map must include metadata describing:

- source space;
- destination space;
- width and height of each space;
- pixel-centre convention;
- axis order;
- normalized or pixel coordinates;
- distortion state;
- map validity representation.

## Run artifacts

Every execution creates a unique immutable directory:

runs/<run_id>/
    run_metadata.json
    PROJECT_STATE_snapshot.md
    commands.txt
    environment.txt
    logs/
    projector_patterns/
    camera_captures_raw/
    camera_captures_validated/
    calibration/
    correspondences/
    prewarps/
    projected_validation/
    captured_validation/
    error_visualizations/
    metrics.json
    report.md

Never overwrite a previous run.

## Accuracy measurements

Always calculate where applicable:

- detected validation points;
- valid-point percentage;
- mean residual error;
- median residual error;
- 95th percentile residual error;
- maximum residual error;
- corner-position error;
- grid-intersection error;
- horizontal-line straightness;
- vertical-line straightness;
- valid dense-map coverage;
- uncorrected-versus-corrected improvement;
- calibration runtime;
- pre-warp generation runtime;
- correction iteration count;
- total wall-clock processing time.

## Flat-wall acceptance gate

**Current status: DONE + VALIDATED** (see `VALIDATION_STATE.md`).

The flat-wall MVP is complete when all conditions hold:

1. Calibration is fully automatic (ChArUco; no manual points).
2. GS-ProCams remains `REUSABLE_COMPONENTS_ONLY`; CSPR is not in the production
   state machine.
3. Captures require no manually selected correspondences.
4. Desired camera-space target is defined and saved before corrected analysis.
5. The projector-space pre-warp is generated automatically from
   `H_cam_from_proj` / `H_proj_from_cam`.
6. The corrected validation board has been physically projected and recaptured.
7. Production metrics compare detections to the independent desired target
   (category B in `METRIC_DEFINITIONS.md`).
8. Corrected result improves target median, target p95, and axis/orthogonality.
9. Contour / AA-box metrics are secondary visualization only (not acceptance).
10. Validation gates 1–10 in `VALIDATION_STATE.md` pass.
11. Final H, inverse H, desired target, and pre-warp are preserved.
12. PROJECT_STATE.md has status DONE and VALIDATION_STATE.md is VALIDATED.

A visually convincing screenshot alone does not satisfy the gate.

## Implementation restrictions

- Do not add unrelated models.
- Do not redesign CSPR-Net or GS-ProCams wholesale.
- Do not force GS-ProCams to run on unsupported hardware if it delays the MVP.
- Do not silently resize, crop, rotate or mirror captures.
- Do not silently ignore lens distortion.
- Do not claim automation while manual points are required.
- Do not optimize before correctness.
- Do not accept an improvement smaller than measurement variability.
- Do not replace failed repository functionality without documenting why.
- Do not stop after creating documentation.
- Do not mark a milestone complete without executing its gate.
- Preserve original licenses and attribution.