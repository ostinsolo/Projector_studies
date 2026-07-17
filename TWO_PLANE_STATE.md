# TWO_PLANE_STATE

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

## Two-plane piecewise homography

**SOFTWARE_READY_FOR_PHYSICAL_VALIDATION**

Architecture decision remains **CONTINUE_CURRENT_ARCHITECTURE**.

| Phase | Status |
|-------|--------|
| Architectural empty-scene analysis | implemented |
| Target region prior | implemented |
| Multi-frame ChArUco + dual-H + seam verify | implemented |
| Piecewise pre-warp + refine | implemented |
| Core synthetic suite | **14/14 PASS** |
| Extended synthetic suite | **25/25 PASS** |
| Package tests | **71 passed** |
| Adversarial self-review | 2 passes (mirror, state isolation) |
| Physical two-wall validation | **not run** |

## Exact next command (when hardware ready)

```bash
procam-calibrate auto-two-plane \
  --run-dir procam-test-data/runs/two_plane_phys_$(date +%Y%m%d_%H%M%S) \
  --mode physical \
  --setup-confirmed
```

Checklist: `docs/TWO_PLANE_PHYSICAL_RUNBOOK.md`  
Setup request: `docs/TWO_PLANE_PHYSICAL_SETUP_REQUEST.md`

Immutable synthetic: `procam-test-data/runs/two_plane_synthetic_20260717`
