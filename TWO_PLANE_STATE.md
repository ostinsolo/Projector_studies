# TWO_PLANE_STATE

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

## Two-plane + oblique video

**PHYSICAL_IN_PROGRESS** — not `VALIDATED_OBLIQUE_VIDEO` / not `REGION_EXPANSION_VALIDATED`

Run: `procam-test-data/runs/two_plane_oblique_20260717_180931`

### Answered questions

1. **Why ~37% usable?** Irregular small usable footprint (12.6% of frame, 65.7% bbox fill) + axis-aligned 16:9. Not mainly margin. Free-aspect could reach ~52%.
2. **Lower-right anomaly?** Cell `r3c5` has **0** calib points; LR corner 82 px from nearest support. Residuals where supported stay &lt;2 px. Extrapolation / sparse right-wall coverage — not global H failure.
3. **Largest reliable 16:9 under local support gates?** Support-safe ≈ **92796 px (26.6% usable)** — *smaller* than the 37% mask-max rectangle.
4. **Coverage gate?** Validation-pattern detection sparsity (13 IDs), not proof of bad geometry.

### Evidence

- `region_forensics/FORENSIC_37PCT_REPORT.md`
- `region_optimisation/` (offline search + physical captures)
- `docs/REGION_EXPANSION_FORENSICS.md`

### To enlarge past support-safe

Need denser lower-right / wall-B correspondences (extra calib patterns or Gray-code) — **not** a full discard of current Hs unless devices moved.

```bash
procam-calibrate optimise-video-region \
  --calibration-run procam-test-data/runs/two_plane_oblique_20260717_180931 \
  --aspect 16:9 --physical --resume
```
