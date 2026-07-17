# Validation Matrix

| Gate | Name | Status | Evidence |
|------|------|--------|----------|
| 1 | Contract / documentation consistency | PASS | `AGENTS.md`, `PRODUCTION_PIPELINE_SPEC.md`, stale-search clean for production instructions |
| 2 | Single implementation source | PASS | `tests/test_validation_suite.py` Gate2; module `charuco.py` |
| 3 | Coordinate / warp mathematics | PASS | `tests/results/gate3_roundtrip.json`; prewarp convention test |
| 4 | Metric correctness (independent target) | PASS | `METRIC_DEFINITIONS.md`; empty→null test; desired_target.json |
| 5 | Detection coverage | PASS | expected_visible from prewarp; canonical 27/29≈0.93; board-region quadrants |
| 6 | Synthetic failure injection | PASS | `FAILURE_INJECTION_RESULTS.md`; 14 tests include matrix |
| 7 | Canonical run replay | PASS | `tests/results/gate7_canonical_replay.json`; immutable hashes |
| 8 | Clean environment reproduction | PASS | `CLEAN_REPRODUCTION_REPORT.md` |
| 9 | Process / state-machine safety | PASS | lock rejection; atomic state; homography pipeline default |
| 10 | Fresh live regression | PASS | `procam-test-data/runs/flatwall_live_20260717_154312` |

Suite log: `procam-calibration/tests/results/pytest_gate2_9.txt`  
Clean env log: `procam-calibration/tests/results/clean_env_pytest.txt`
