# Region expansion forensics (run `two_plane_oblique_20260717_180931`)

## Why ~37% of usable?

| Quantity | Value |
|----------|------:|
| Usable mask / camera frame | **12.6%** |
| Usable bbox fill (shape irregularity) | **65.7%** |
| Selected 16:9 / usable | **37.1%** |
| Approx free-aspect max / usable | **~52%** |
| Safety erode cost | **~0%** |

The usable mask is a small, irregular oblique footprint (ceiling cut + keystone). An axis-aligned **16:9** rectangle cannot fill most of that blob. Margin is not the main loss.

Artifacts: `…/region_forensics/FORENSIC_37PCT_REPORT.md`

## Lower-right anomaly

| Evidence | Result |
|----------|--------|
| Global calib median / p95 | **0.51 / 1.28** px |
| Cell `r3c5` (LR of selected rect) | **0 correspondences** |
| Nearest support to LR corner | **82 px** away |
| Cells with support | residuals typically **&lt; 2 px** |
| Corrected validation ChArUco IDs | **13** (coverage gate fail) |

**Diagnosis:** unsupported extrapolation at the lower-right of the *selected video rectangle*, plus sparse validation-pattern detections. Not a global dual-H failure (~70 px uncorrected → ~1.8 px corrected still stands where markers exist).

## Area vs support tradeoff

| Candidate | Area | % usable | LR supported? |
|-----------|-----:|---------:|:--------------|
| Baseline max-inscribed 16:9 | 129600 | 37.1% | **No** |
| Recomputed max inscribed | 135516 | 38.8% | **No** |
| Max **support-safe** 16:9 | 92796 | 26.6% | **Yes** |

Enlarging the rectangle without new right-wall / lower-right correspondences **increases** unsupported extrapolation. A smaller support-safe rectangle is the honest offline maximum under local gates.

## Coverage failure vs geometry failure

Absolute validation failed on **coverage**, not median error. The validation board under-detects on the piecewise-corrected frame. Fix with tiled / larger ChArUco validation (or Gray-code if tiles still fail) — not by discarding the homographies.

## Commands

```bash
procam-calibrate optimise-video-region \
  --calibration-run procam-test-data/runs/two_plane_oblique_20260717_180931 \
  --aspect 16:9

procam-calibrate optimise-video-region \
  --calibration-run procam-test-data/runs/two_plane_oblique_20260717_180931 \
  --aspect 16:9 --physical --resume
```

Evidence: `…/region_optimisation/`
