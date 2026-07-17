# Failure Injection Results

Synthetic matrix from `tests/test_validation_suite.py` (seed=42).
Results JSON: `procam-calibration/tests/results/gate6_synthetic.json` (written on test run).

| Case | Expected | Observed | Notes |
|------|----------|----------|-------|
| identity | success | PASS | already-good path |
| horizontal_keystone | success | PASS | target median drops sharply |
| vertical_keystone | success | PASS | |
| combined_projective | success | PASS | |
| camera_rotation | success | PASS | |
| mirror | failure | PASS (rejected) | acceptance must not pass |
| mild_blur | success | PASS | same keystone + blur path |

Additional coordinate failure detectors (Gate 3):

| Injection | Result |
|-----------|--------|
| x/y swap | large desired-target error |
| stale 1280×720 ROI on 1920×1080 | raises |
| double-invert prewarp | mean diff vs ideal fails threshold |
| empty detection | `valid:false`, null metrics |

## Policy

Never loosen thresholds to force PASS. Failures must preserve artifacts under
`tests/results/`.
