# PROJECT_STATE

## Status

**RUNNING**

## Current milestone

**Stage 1 — Reproduce CSPR-Net unchanged**

## Exact next action

Create clean venv; install README deps; run sim + exp inference; archive artifacts under `TEST_DATA_ROOT/repro/CSPR-Net/<run_id>/`.

## Paths

| Variable | Absolute path |
|----------|---------------|
| `CSPR_NET_ROOT` | `/Users/ostino/Documents/Code/Projector_studies/CSPR-Net` |
| `GS_PROCAMS_ROOT` | `/Users/ostino/Documents/Code/Projector_studies/GS-ProCams` |
| `INTEGRATION_ROOT` | `/Users/ostino/Documents/Code/Projector_studies/procam-calibration` |
| `TEST_DATA_ROOT` | `/Users/ostino/Documents/Code/Projector_studies/procam-test-data` |

## Current Git commits

| Repo | Commit |
|------|--------|
| CSPR-Net | `a2a0a28ac2022c876f1ac1e52620b44bfe96672a` |
| GS-ProCams | `dfb7f449710ce41a63369a447ca728e0cb220d97` |

## Environment

| Field | Value |
|-------|-------|
| OS | macOS 15.7.2 arm64 |
| CUDA | Not available |

## Completed work

- M0 audit documentation complete.

## Unresolved failures / blockers

- M1 in progress
- M2 GS expected CUDA/data blocker
- Physical capture not started

## Known assumptions

- CSPR-Net is primary geometric path (AGENTS.md).
- GS must not block flat-wall MVP.
- No unrelated ML models.
