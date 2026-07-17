# TWO_PLANE_STATE

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

## Two-plane piecewise homography + oblique video

**SOFTWARE_READY_FOR_OBLIQUE_VIDEO_VALIDATION**

Architecture decision remains **CONTINUE_CURRENT_ARCHITECTURE**.

| Phase | Status |
|-------|--------|
| Architectural empty-scene analysis | implemented |
| Target region prior | implemented |
| Multi-frame ChArUco + dual-H + seam verify | implemented |
| Active / excluded plane classification (ceiling) | implemented |
| Wall–ceiling boundary + usable/exclusion masks | implemented |
| Maximum inscribed video rectangle | implemented |
| Piecewise video remap + diagnostic + play-video | implemented |
| Core synthetic suite | **14/14 PASS** |
| Extended synthetic suite | **25/25 PASS** |
| Oblique video synthetic suite | **PASS** |
| Package tests | **78+ passed** |
| Adversarial self-review | 2 passes (ceiling-as-content, bbox/leakage) |
| Physical two-wall / oblique video validation | **not run** |

## Straightness note

“Straight” is defined in **camera / audience** coordinates. Other viewpoints may look distorted.

## Exact next command (when hardware ready)

```bash
procam-calibrate auto-two-plane \
  --run-dir procam-test-data/runs/two_plane_oblique_$(date +%Y%m%d_%H%M%S) \
  --mode physical \
  --setup-confirmed
```

Then:

```bash
procam-calibrate play-video \
  --calibration-run <RUN_DIR> \
  --diagnostic \
  --loop
```

Checklist: `docs/TWO_PLANE_PHYSICAL_RUNBOOK.md`  
Oblique setup: `docs/PHYSICAL_OBLIQUE_SETUP.md`  
Ceiling / max region / playback: `docs/CEILING_EXCLUSION.md`, `docs/MAXIMUM_VIDEO_REGION.md`, `docs/VIDEO_PLAYBACK.md`

Immutable synthetic: `procam-test-data/runs/two_plane_synthetic_20260717`
