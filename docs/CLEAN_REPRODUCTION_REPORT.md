# Clean Reproduction Report

## Environment

| Item | Value |
|------|-------|
| Python | 3.10.11 |
| venv | `procam-calibration/.venv_clean_validation` |
| OpenCV | 5.0.0 |
| NumPy | 2.2.6 |
| macOS | 15.7.2 (arm64) |
| Torch | not installed (not required for production homography) |

## Commands

```bash
python3 -m venv procam-calibration/.venv_clean_validation
source procam-calibration/.venv_clean_validation/bin/activate
pip install -U pip
pip install -e "procam-calibration[dev]"
procam-calibrate doctor
cd procam-calibration && python -m pytest tests/test_validation_suite.py -q
python -c "from pathlib import Path; from procam_calibrate.homography_baseline import remeasure_saved_captures; \
 r=remeasure_saved_captures(Path('procam-test-data/runs/flatwall_20260717_141610')); \
 assert r['comparison']['acceptance_pass']"
```

## Results

| Check | Result |
|-------|--------|
| CLI help | PASS |
| doctor | PASS (torch absent OK for homography) |
| unit/validation suite | **14 passed** |
| canonical replay acceptance | PASS (`target_median` 75.55 → 18.24 px) |

Evidence: `procam-calibration/tests/results/clean_env_pytest.txt`
