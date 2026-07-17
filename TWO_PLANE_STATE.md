# TWO_PLANE_STATE

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

## Two-plane + oblique video

**PHYSICAL_IN_PROGRESS** (not yet `VALIDATED_OBLIQUE_VIDEO`)

Architecture: **CONTINUE_CURRENT_ARCHITECTURE**

| Phase | Status |
|-------|--------|
| Software gates (ceiling / max rect / play-video) | **SOFTWARE_READY** — commit `38843a4`+ |
| Package tests | **80 passed** |
| Physical run | `procam-test-data/runs/two_plane_oblique_20260717_180931` |
| Two-plane model | selected (`prefer_two_plane` spatial fold) |
| Corrected median / p95 | **~1.83 / ~2.49** cam px |
| Seam | OK (median mismatch ~0.34) |
| Ceiling leakage (self-check) | **0** |
| Max video region | valid (~37% of usable mask) |
| `play-video --diagnostic` | **ok** (60 frames projected) |
| Absolute gate | **failed coverage** (sparse corrected ChArUco IDs) |
| Full VALIDATED_OBLIQUE_VIDEO | **not claimed** |

## Exact next command (coverage / finalize)

Keep devices fixed. Re-run on a **new** directory (or resume after fixing coverage lighting):

```bash
procam-calibrate auto-two-plane \
  --run-dir procam-test-data/runs/two_plane_oblique_$(date +%Y%m%d_%H%M%S) \
  --mode physical \
  --setup-confirmed
```

Playback of current calibration:

```bash
procam-calibrate play-video \
  --calibration-run procam-test-data/runs/two_plane_oblique_20260717_180931 \
  --diagnostic --loop
```

Checklist: `docs/TWO_PLANE_PHYSICAL_RUNBOOK.md` · `docs/PHYSICAL_OBLIQUE_SETUP.md`
