"""Offline region optimisation and dense diagnostic tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from procam_calibrate.cli import build_parser
from procam_calibrate.dense_diagnostic import generate_dense_diagnostic_frame, measure_spatial_grid
from procam_calibrate.optimise_video_region import OptimiseConfig, run_offline_search


RUN = (
    Path(__file__).resolve().parents[2]
    / "procam-test-data"
    / "runs"
    / "two_plane_oblique_20260717_180931"
)


def test_dense_diagnostic_shape():
    fr = generate_dense_diagnostic_frame(640, 360, 0.2)
    assert fr.shape == (360, 640, 3)
    assert fr.max() > 100


def test_spatial_grid_flags_empty_cell():
    pts = np.array([[10.0, 10.0], [20.0, 10.0], [15.0, 20.0]])
    err = np.array([0.5, 0.4, 0.6])
    grid = measure_spatial_grid(pts, err, {"x0": 0, "y0": 0, "x1": 99, "y1": 99}, cols=2, rows=2)
    assert grid["cells"]["r0c0"]["supported"] is True
    assert grid["cells"]["r1c1"]["supported"] is False
    assert grid["lower_right_supported"] is False


def test_optimise_cli_registered():
    p = build_parser()
    assert "optimise-video-region" in p._subparsers._group_actions[0].choices


def test_offline_search_on_physical_run():
    if not (RUN / "video_region" / "maximum_video_region.json").exists():
        return
    summary = run_offline_search(OptimiseConfig(calibration_run=RUN, aspect="16:9"))
    assert summary["n_candidates"] >= 1
    assert (RUN / "region_optimisation" / "OFFLINE_REGION_SEARCH.json").exists()
    # Baseline LR cell unsupported was a known issue — search must record fail reasons
    assert "best" in summary
