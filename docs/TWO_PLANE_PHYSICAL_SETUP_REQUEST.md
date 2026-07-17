# Two-Plane Physical Setup Request

## Status: BLOCKED

Synthetic software gates are **PASS**. This document is the exact physical
checklist for when the user is ready — **do not rearrange hardware until this
gate is intentionally opened**.

## User actions only

When unlocked, the user should only need to:

1. Position **two connected planar surfaces** (a corner / fold) in the projector field.
2. Keep the **projector fixed**.
3. Keep the **iPhone camera fixed**.
4. Confirm the **full projected region and the fold** are visible to the camera.

All pattern generation, display, capture, fitting, correction, validation, and
artifact creation remain automatic via:

```bash
procam-calibrate auto-two-plane --run-dir <NEW_RUN_DIR>
```

(Physical mode is currently refused with `--allow-physical` until this request
is explicitly approved in a follow-up.)

## Do not

- Move the projector or camera after calibration begins
- Soft-blend over an incorrect seam
- Overwrite flat-wall immutable runs or production state
