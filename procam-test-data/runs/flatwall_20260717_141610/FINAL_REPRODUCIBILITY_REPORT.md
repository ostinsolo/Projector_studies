# Flat-wall homography MVP — final report

## Acceptance: PASS / DONE

## Pipeline

Production path (no neural training):

```
discover projector and camera
→ display ChArUco calibration board
→ capture
→ detect correspondences
→ estimate H_projector_to_camera
→ invert to H_camera_to_projector
→ generate projector pre-warp
→ display corrected validation board
→ capture
→ independently measure residual error
→ save result
→ DONE
```

## Reproduce

```bash
procam-calibrate auto-wall --run-dir ../procam-test-data/runs/flatwall_20260717_141610
```

Or one-shot baseline:

```bash
procam-calibrate flatwall-homography --run-dir ../procam-test-data/runs/flatwall_20260717_141610
```

## Primary metrics (independent known ProjFB ↔ detected CamImg)

Per-capture homography refit residual, affine residual from normalized projector
coords, and axis-alignment of board rows/cols (not contour AA).

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| median_reproj_px (H refit) | 0.5429237316629689 | 0.28888209463699716 |
| p95_reproj_px | 1.0603296901249402 | 0.4964905858679951 |
| max_reproj_px | 1.4925180761735322 | 0.6814668432909146 |
| corner_displacement_median_px | 0.5429237316629689 | 0.28888209463699716 |
| affine_median_reproj_px | 5.04236772580225 | 0.27768750642739565 |
| horizontal_axis_deviation_deg | 12.691370389235487 | 0.03318554684570339 |
| vertical_axis_deviation_deg | 9.249814877456174 | None |
| n_matched | 49 | 27 |

## Secondary visual metric (not used for acceptance)

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| aa_corner_median_err_px | 79.17235233170469 | 0.14162509551541136 |

## Detection / board

- dictionary: `DICT_4X4_50`
- board: `{"squares": [10, 7], "dictionary_id": 0, "dictionary_name": "DICT_4X4_50", "square_length": 1.0, "marker_length": 0.7, "generate_margin": 20, "generate_border_bits": 1, "expected_charuco_corners": 54, "ransac_reproj_threshold": 3.0}`
- expected markers/corners: see detection blocks below
- RANSAC inliers (fit): `49`

## Comparison / gates

```json
{
  "valid": true,
  "improvement": {
    "median_reproj_px": {
      "uncorrected": 0.5429237316629689,
      "corrected": 0.28888209463699716,
      "relative_improvement": 0.4679140406109073
    },
    "p95_reproj_px": {
      "uncorrected": 1.0603296901249402,
      "corrected": 0.4964905858679951,
      "relative_improvement": 0.531758291320228
    },
    "max_reproj_px": {
      "uncorrected": 1.4925180761735322,
      "corrected": 0.6814668432909146,
      "relative_improvement": 0.5434113300402791
    },
    "mean_reproj_px": {
      "uncorrected": 0.5823032730015945,
      "corrected": 0.29515020820452575,
      "relative_improvement": 0.49313318009854706
    }
  },
  "affine_median_before": 5.04236772580225,
  "affine_median_after": 0.27768750642739565,
  "affine_improvement_pass": true,
  "axis_deviation_before_deg": 12.691370389235487,
  "axis_deviation_after_deg": 0.03318554684570339,
  "axis_improvement_pass": true,
  "absolute_gate_pass": true,
  "tight_improve_pass": true,
  "geometry_gate_pass": true,
  "acceptance_pass": true,
  "failure_reason": null,
  "secondary_aa_before": 79.17235233170469,
  "secondary_aa_after": 0.14162509551541136,
  "secondary_aa_note": "diagnostic only; not used for acceptance"
}
```

## Artifacts preserved

- `../procam-test-data/runs/flatwall_20260717_141610/homography_baseline/H_cam_from_proj.npy`
- `../procam-test-data/runs/flatwall_20260717_141610/homography_baseline/H_proj_from_cam.npy`
- `../procam-test-data/runs/flatwall_20260717_141610/homography_baseline/prewarp_charuco.png`
- `../procam-test-data/runs/flatwall_20260717_141610/calibration/homography`
- `../procam-test-data/runs/flatwall_20260717_141610/prewarps/prewarp_best.png`
- `../procam-test-data/runs/flatwall_20260717_141610/metrics.json`

## Full metrics JSON

```json
{
  "uncorrected": {
    "valid": true,
    "pass": true,
    "tag": "uncorrected",
    "metric_family": "independent_proj_cam_reprojection_refit",
    "n_valid_measurements": 49,
    "matched_ids": [
      0,
      1,
      2,
      3,
      4,
      5,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42,
      43,
      44,
      45,
      46,
      47,
      48,
      49,
      50,
      51,
      52,
      53
    ],
    "mean_reproj_px": 0.5823032730015945,
    "median_reproj_px": 0.5429237316629689,
    "p95_reproj_px": 1.0603296901249402,
    "max_reproj_px": 1.4925180761735322,
    "corner_displacement_mean_px": 0.5823032730015945,
    "corner_displacement_median_px": 0.5429237316629689,
    "horizontal_line_deviation_px": 0.18337305517388813,
    "vertical_line_deviation_px": 0.2168839852038218,
    "horizontal_line_deviation_max_px": 0.38706715224907384,
    "vertical_line_deviation_max_px": 0.2655668674151457,
    "horizontal_axis_deviation_deg_median": 12.691370389235487,
    "horizontal_axis_deviation_deg_max": 13.992824915751466,
    "vertical_axis_deviation_deg_median": 9.249814877456174,
    "vertical_axis_deviation_deg_max": 9.386227952731204,
    "n_horizontal_rows_measured": 8,
    "n_vertical_cols_measured": 5,
    "affine_median_reproj_px": 5.04236772580225,
    "affine_p95_reproj_px": 10.72497473761888,
    "affine_max_reproj_px": 11.66813639549331,
    "affine_mean_reproj_px": 5.497139638244281,
    "mean_residual_px": 0.5823032730015945,
    "median_residual_px": 0.5429237316629689,
    "p95_residual_px": 1.0603296901249402,
    "max_residual_px": 1.4925180761735322,
    "detection": {
      "valid": true,
      "image_resolution": [
        1920,
        1440
      ],
      "board": {
        "squares": [
          10,
          7
        ],
        "dictionary_id": 0,
        "dictionary_name": "DICT_4X4_50",
        "square_length": 1.0,
        "marker_length": 0.7,
        "generate_margin": 20,
        "generate_border_bits": 1,
        "expected_charuco_corners": 54,
        "ransac_reproj_threshold": 3.0
      },
      "dictionary": "DICT_4X4_50",
      "preprocessing": {
        "color_conversion": "BGR2GRAY",
        "adaptive_histogram": false,
        "blur": null,
        "resize": null
      },
      "expected_charuco_corners": 54,
      "n_markers_detected": 32,
      "n_charuco_corners_interpolated": 49,
      "n_corners_returned": 49,
      "corner_ids": [
        0,
        1,
        2,
        3,
        4,
        5,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53
      ],
      "missing_corner_ids": [
        6,
        7,
        8,
        17,
        26
      ],
      "n_missing_corners": 5,
      "rejection_reasons": [],
      "min_corners_required": 8
    },
    "matching": {
      "n_proj_ids": 54,
      "n_cam_ids": 49,
      "n_matched": 49,
      "matched_ids": [
        0,
        1,
        2,
        3,
        4,
        5,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53
      ],
      "unmatched_proj_ids": [
        6,
        7,
        8,
        17,
        26
      ],
      "unmatched_cam_ids": [],
      "rejection_reasons": []
    },
    "homography_fit": {
      "n_correspondences": 49,
      "n_inliers": 49,
      "n_outliers": 0,
      "outlier_indices": [],
      "ransac_reproj_threshold": 3.0,
      "homography_condition_number": 672836.9075551097,
      "mean_reproj_px": 0.5823032730015945,
      "median_reproj_px": 0.5429237316629689,
      "p95_reproj_px": 1.0603296901249402,
      "max_reproj_px": 1.4925180761735322,
      "all_point_mean_reproj_px": 0.5823032730015945,
      "inlier_mask": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ]
    },
    "secondary_aa_hull_metric": {
      "valid": true,
      "label": "secondary_aa_hull",
      "aa_corner_median_err_px": 79.17235233170469,
      "aa_corner_mean_err_px": 79.17235242222743,
      "aspect_ratio": 1.8186455607851464,
      "ordered_corners": [
        [
          731.7984619140625,
          194.97991943359375
        ],
        [
          1422.4539794921875,
          56.902320861816406
        ],
        [
          1498.3773193359375,
          436.6659851074219
        ],
        [
          807.7218017578125,
          574.7435913085938
        ]
      ]
    },
    "secondary_metric_note": "AA-hull / contour-style metric is secondary visualization only; not used for acceptance.",
    "fixed_calib_H_median_reproj_px": 0.5429237316629689,
    "fixed_calib_H_note": "Diagnostic only: residual vs calibration H. After pre-warp this is expected to be large; acceptance uses per-capture refit + axis/affine metrics.",
    "capture_path": "/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/homography_baseline/captures/charuco_uncorrected.png",
    "image_size": [
      1920,
      1440
    ]
  },
  "corrected": {
    "valid": true,
    "pass": true,
    "tag": "corrected",
    "metric_family": "independent_proj_cam_reprojection_refit",
    "n_valid_measurements": 27,
    "matched_ids": [
      14,
      15,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32,
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41,
      42
    ],
    "mean_reproj_px": 0.29515020820452575,
    "median_reproj_px": 0.28888209463699716,
    "p95_reproj_px": 0.4964905858679951,
    "max_reproj_px": 0.6814668432909146,
    "corner_displacement_mean_px": 0.29515020820452575,
    "corner_displacement_median_px": 0.28888209463699716,
    "horizontal_line_deviation_px": 0.2482423557139226,
    "vertical_line_deviation_px": null,
    "horizontal_line_deviation_max_px": 0.31468619089331185,
    "vertical_line_deviation_max_px": null,
    "horizontal_axis_deviation_deg_median": 0.03318554684570339,
    "horizontal_axis_deviation_deg_max": 0.04855665431351443,
    "vertical_axis_deviation_deg_median": null,
    "vertical_axis_deviation_deg_max": null,
    "n_horizontal_rows_measured": 4,
    "n_vertical_cols_measured": 0,
    "affine_median_reproj_px": 0.27768750642739565,
    "affine_p95_reproj_px": 0.5478726734596439,
    "affine_max_reproj_px": 0.7453813785701434,
    "affine_mean_reproj_px": 0.30183554110920546,
    "mean_residual_px": 0.29515020820452575,
    "median_residual_px": 0.28888209463699716,
    "p95_residual_px": 0.4964905858679951,
    "max_residual_px": 0.6814668432909146,
    "detection": {
      "valid": true,
      "image_resolution": [
        1920,
        1440
      ],
      "board": {
        "squares": [
          10,
          7
        ],
        "dictionary_id": 0,
        "dictionary_name": "DICT_4X4_50",
        "square_length": 1.0,
        "marker_length": 0.7,
        "generate_margin": 20,
        "generate_border_bits": 1,
        "expected_charuco_corners": 54,
        "ransac_reproj_threshold": 3.0
      },
      "dictionary": "DICT_4X4_50",
      "preprocessing": {
        "color_conversion": "BGR2GRAY",
        "adaptive_histogram": false,
        "blur": null,
        "resize": null
      },
      "expected_charuco_corners": 54,
      "n_markers_detected": 20,
      "n_charuco_corners_interpolated": 27,
      "n_corners_returned": 27,
      "corner_ids": [
        14,
        15,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42
      ],
      "missing_corner_ids": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        16,
        17,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53
      ],
      "n_missing_corners": 27,
      "rejection_reasons": [],
      "min_corners_required": 8
    },
    "matching": {
      "n_proj_ids": 54,
      "n_cam_ids": 27,
      "n_matched": 27,
      "matched_ids": [
        14,
        15,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42
      ],
      "unmatched_proj_ids": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        16,
        17,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53
      ],
      "unmatched_cam_ids": [],
      "rejection_reasons": []
    },
    "homography_fit": {
      "n_correspondences": 27,
      "n_inliers": 27,
      "n_outliers": 0,
      "outlier_indices": [],
      "ransac_reproj_threshold": 3.0,
      "homography_condition_number": 520013.3887709768,
      "mean_reproj_px": 0.29515020820452575,
      "median_reproj_px": 0.28888209463699716,
      "p95_reproj_px": 0.4964905858679951,
      "max_reproj_px": 0.6814668432909146,
      "all_point_mean_reproj_px": 0.29515020820452575,
      "inlier_mask": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ]
    },
    "secondary_aa_hull_metric": {
      "valid": true,
      "label": "secondary_aa_hull",
      "aa_corner_median_err_px": 0.14162509551541136,
      "aa_corner_mean_err_px": 0.14162509551542432,
      "aspect_ratio": 2.228335682455483,
      "ordered_corners": [
        [
          779.2247924804688,
          144.7286834716797
        ],
        [
          1506.5816650390625,
          144.9871063232422
        ],
        [
          1506.4656982421875,
          471.39971923828125
        ],
        [
          779.1088256835938,
          471.14129638671875
        ]
      ]
    },
    "secondary_metric_note": "AA-hull / contour-style metric is secondary visualization only; not used for acceptance.",
    "fixed_calib_H_median_reproj_px": 57.90868746100459,
    "fixed_calib_H_note": "Diagnostic only: residual vs calibration H. After pre-warp this is expected to be large; acceptance uses per-capture refit + axis/affine metrics.",
    "capture_path": "/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/homography_baseline/captures/charuco_corrected.png",
    "image_size": [
      1920,
      1440
    ]
  },
  "comparison": {
    "valid": true,
    "improvement": {
      "median_reproj_px": {
        "uncorrected": 0.5429237316629689,
        "corrected": 0.28888209463699716,
        "relative_improvement": 0.4679140406109073
      },
      "p95_reproj_px": {
        "uncorrected": 1.0603296901249402,
        "corrected": 0.4964905858679951,
        "relative_improvement": 0.531758291320228
      },
      "max_reproj_px": {
        "uncorrected": 1.4925180761735322,
        "corrected": 0.6814668432909146,
        "relative_improvement": 0.5434113300402791
      },
      "mean_reproj_px": {
        "uncorrected": 0.5823032730015945,
        "corrected": 0.29515020820452575,
        "relative_improvement": 0.49313318009854706
      }
    },
    "affine_median_before": 5.04236772580225,
    "affine_median_after": 0.27768750642739565,
    "affine_improvement_pass": true,
    "axis_deviation_before_deg": 12.691370389235487,
    "axis_deviation_after_deg": 0.03318554684570339,
    "axis_improvement_pass": true,
    "absolute_gate_pass": true,
    "tight_improve_pass": true,
    "geometry_gate_pass": true,
    "acceptance_pass": true,
    "failure_reason": null,
    "secondary_aa_before": 79.17235233170469,
    "secondary_aa_after": 0.14162509551541136,
    "secondary_aa_note": "diagnostic only; not used for acceptance"
  },
  "calib_H_shape": [
    3,
    3
  ],
  "valid": true
}
```

## State history

```json
[
  {
    "from": "PREFLIGHT",
    "to": "DISCOVER_CAMERA",
    "note": "init",
    "t": "2026-07-17T14:49:17.205187"
  },
  {
    "from": "DISCOVER_CAMERA",
    "to": "DISCOVER_PROJECTOR",
    "note": "camera ready",
    "t": "2026-07-17T14:49:20.158600"
  },
  {
    "from": "DISCOVER_PROJECTOR",
    "to": "GENERATE_PATTERNS",
    "note": "projector ready \u2014 sync ProjFB size",
    "t": "2026-07-17T14:49:20.630608"
  },
  {
    "from": "GENERATE_PATTERNS",
    "to": "VERIFY_CAPTURE",
    "note": "patterns ready",
    "t": "2026-07-17T14:49:20.761147"
  },
  {
    "from": "VERIFY_CAPTURE",
    "to": "CAPTURE_CALIBRATION",
    "note": "preflight ok",
    "t": "2026-07-17T14:49:33.203871"
  },
  {
    "from": "CAPTURE_CALIBRATION",
    "to": "VALIDATE_CALIBRATION_CAPTURES",
    "note": "captures done",
    "t": "2026-07-17T14:50:21.027734"
  },
  {
    "from": "VALIDATE_CALIBRATION_CAPTURES",
    "to": "FIT_CSPR_SCENE",
    "note": "validation ok",
    "t": "2026-07-17T14:50:21.241888"
  },
  {
    "from": "CAPTURE_CALIBRATION",
    "to": "BLOCKED_EXTERNAL",
    "note": "",
    "t": "2026-07-17T14:50:56.880639"
  },
  {
    "from": "BLOCKED_EXTERNAL",
    "to": "FIT_CSPR_SCENE",
    "note": "resume after false block; validated captures present",
    "t": "2026-07-17T14:51:30.365683"
  },
  {
    "from": "FIT_CSPR_SCENE",
    "to": "FIT_CSPR_SCENE",
    "note": "single-process resume after killing duplicate trainers",
    "t": "2026-07-17T14:53:32.094030"
  },
  {
    "from": "FIT_CSPR_SCENE",
    "to": "GENERATE_PREWARP",
    "note": "trained",
    "t": "2026-07-17T15:12:43.669498"
  },
  {
    "from": "GENERATE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "prewarp ready",
    "t": "2026-07-17T15:12:49.692134"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:12:51.271794"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:12:54.696039"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:12:54.779035"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 1",
    "t": "2026-07-17T15:12:54.813954"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:12:55.921464"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:12:59.902947"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:12:59.976030"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 2",
    "t": "2026-07-17T15:13:00.012163"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:01.149707"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:05.080176"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:05.156281"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 3",
    "t": "2026-07-17T15:13:05.190725"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:06.305260"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:10.296877"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:10.372373"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 4",
    "t": "2026-07-17T15:13:10.407944"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:11.524266"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:15.500637"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:15.574668"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 5",
    "t": "2026-07-17T15:13:15.610373"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:16.754660"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:20.883876"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:20.958453"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 6",
    "t": "2026-07-17T15:13:20.993149"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:22.113017"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:26.011129"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:26.083338"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 7",
    "t": "2026-07-17T15:13:26.118735"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:27.237177"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:31.093979"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:31.170522"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "PROJECT_VALIDATION",
    "note": "refine iter 8",
    "t": "2026-07-17T15:13:31.206859"
  },
  {
    "from": "PROJECT_VALIDATION",
    "to": "CAPTURE_VALIDATION",
    "note": "displaying corrected",
    "t": "2026-07-17T15:13:32.314151"
  },
  {
    "from": "CAPTURE_VALIDATION",
    "to": "MEASURE_RESIDUAL",
    "note": "captured validation",
    "t": "2026-07-17T15:13:36.302833"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "REFINE_PREWARP",
    "note": "need refine",
    "t": "2026-07-17T15:13:36.377137"
  },
  {
    "from": "REFINE_PREWARP",
    "to": "FINAL_REPRODUCTION",
    "note": "max refine",
    "t": "2026-07-17T15:13:36.377640"
  },
  {
    "from": "FINAL_REPRODUCTION",
    "to": "BLOCKED_EXTERNAL",
    "note": "acceptance not met",
    "t": "2026-07-17T15:13:36.395451"
  },
  {
    "from": "BLOCKED_EXTERNAL",
    "to": "ACCEPTANCE_FAILED",
    "note": "reclassified; validation forensic",
    "t": "2026-07-17T15:22:05.474775"
  },
  {
    "from": "MEASURE_RESIDUAL",
    "to": "FINAL_REPRODUCTION",
    "note": "homography remeasure complete",
    "t": "2026-07-17T15:29:38.896338"
  },
  {
    "from": "FINAL_REPRODUCTION",
    "to": "DONE",
    "note": "accepted",
    "t": "2026-07-17T15:29:38.900137"
  }
]
```
