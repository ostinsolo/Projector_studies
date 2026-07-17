# procam-calibration

Integration CLI for automatic projector–camera geometric calibration.

**Production flat-wall method:** ChArUco + planar homography  
(`procam-calibrate auto-wall`, default `--pipeline homography`).

**CSPR-Net / GS-ProCams:** research only — not required for flat-wall production.

## Setup

```bash
export INTEGRATION_ROOT=/Users/ostino/Documents/Code/Projector_studies/procam-calibration
export TEST_DATA_ROOT=/Users/ostino/Documents/Code/Projector_studies/procam-test-data

python3 -m venv .venv
source .venv/bin/activate
pip install -e "$INTEGRATION_ROOT"
pip install pytest   # for validation suite
```

## Production command

```bash
procam-calibrate auto-wall --run-dir "$TEST_DATA_ROOT/runs/<NEW_RUN_ID>"
```

Requires the projector as an **extended** macOS display (not mirrored) and an
iPhone Continuity / AVFoundation camera.

## Validation suite

```bash
cd "$INTEGRATION_ROOT"
python -m pytest tests/test_validation_suite.py -v
```

Canonical immutable reference run:

`procam-test-data/runs/flatwall_20260717_141610`

See `../VALIDATION_STATE.md` and `../docs/PRODUCTION_PIPELINE_SPEC.md`.
