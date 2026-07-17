# Synthetic gate

pass=True

```json
{
  "pass": true,
  "homography_recovery": {
    "n_points": 264,
    "n_inliers": 264,
    "mean_reproj_px": 0.18928355329465946,
    "median_reproj_px": 0.17513875378274507,
    "p95_reproj_px": 0.37361290596821567,
    "max_reproj_px": 0.5830475603309349
  },
  "prewarp_roundtrip": {
    "mean_abs_intensity_diff_valid": 146.85952758789062,
    "corner_mean_px": 2.4009500300404224e-05,
    "corner_median_px": 2.064741064143176e-05,
    "corner_max_px": 5.379057260031205e-05
  },
  "thresholds": {
    "median_reproj_px": 1.0,
    "p95_reproj_px": 2.0,
    "corner_median_px": 1.5
  },
  "spaces": {
    "H_projector_to_camera": "ProjFB -> CamImg",
    "H_camera_to_projector": "CamImg -> ProjFB",
    "prewarp": "CamImg desired content -> ProjFB framebuffer"
  }
}
```
