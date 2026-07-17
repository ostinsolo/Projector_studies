"""Projected-region preflight (partial camera FOV / oblique coverage)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from procam_calibrate.auto_two_plane import evaluate_projection_preflight


def test_global_mean_would_fail_but_projected_passes():
    h, w = 480, 640
    black = np.zeros((h, w), dtype=np.uint8)
    white = np.zeros((h, w), dtype=np.uint8)
    # Small bright projected patch (both left and right of centre)
    white[80:220, 180:420] = 120
    # Global mean contrast is tiny
    assert float(white.mean() - black.mean()) < 40
    out = evaluate_projection_preflight(black, white)
    assert out["pass"] is True
    assert out["checks"]["projected_contrast"] >= 35
    assert out["checks"]["both_sides_lit"] is True


def test_physical_oblique_fixture_preflight():
    root = Path(__file__).resolve().parents[2] / "procam-test-data" / "runs"
    run = root / "two_plane_oblique_20260717_180218" / "preflight"
    if not (run / "white_cam.png").exists():
        return  # fixture optional outside this workspace
    gb = cv2.cvtColor(cv2.imread(str(run / "black_cam.png")), cv2.COLOR_BGR2GRAY)
    gw = cv2.cvtColor(cv2.imread(str(run / "white_cam.png")), cv2.COLOR_BGR2GRAY)
    out = evaluate_projection_preflight(gb, gw)
    assert out["pass"] is True, out["checks"]
