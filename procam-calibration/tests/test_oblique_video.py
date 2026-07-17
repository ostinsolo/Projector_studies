"""Ceiling exclusion, max video region, piecewise playback (software gates)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from procam_calibrate.excluded_geometry import classify_active_and_excluded_planes
from procam_calibrate.synthetic_oblique_video import (
    generate_two_wall_with_ceiling,
    run_oblique_video_suite,
)
from procam_calibrate.video_playback import (
    generate_diagnostic_frame,
    measure_ceiling_leakage,
    write_diagnostic_video,
)
from procam_calibrate.video_region import maximum_inscribed_rectangle, parse_aspect
from procam_calibrate.cli import build_parser


def test_ceiling_not_artistic_third_plane():
    sc = generate_two_wall_with_ceiling(0.0, rng=np.random.default_rng(0))
    cls = classify_active_and_excluded_planes(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    assert "third artistic" in cls.get("note", "").lower() or "excluded" in cls.get("model", "")
    assert cls.get("valid_active") is True
    assert cls.get("ceiling_detected") is True
    assert cls.get("H_cam_from_proj_plane_A") is not None
    assert cls.get("H_cam_from_proj_plane_B") is not None
    # Ceiling labels present; walls not contaminated as ceiling-only model
    labs = np.array(cls["labels"])
    assert (labs == 2).sum() >= 8
    assert (labs == 0).sum() >= 12 and (labs == 1).sum() >= 12


def test_oblique_video_suite(tmp_path: Path):
    summary = run_oblique_video_suite(tmp_path / "obl")
    failed = [c for c in summary["cases"] if not c["pass"]]
    assert summary["all_pass"], failed


def test_max_rect_respects_aspect_and_mask():
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[50:350, 40:560] = 255
    # cut ceiling band
    mask[:80, :] = 0
    r = maximum_inscribed_rectangle(mask, aspect=16 / 9, step=4)
    assert r["valid"]
    poly = np.array(r["camera_polygon"])
    assert poly[:, 1].min() >= 80 - 1
    assert abs(r["aspect"] - 16 / 9) < 1e-6
    assert r["claim"] == "maximum_found_under_current_constraints"


def test_ceiling_leakage_gate():
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    frame[40:100, 20:170] = 255
    ceil = np.zeros((108, 192), dtype=np.uint8)
    ceil[:30, :] = 255
    ok = measure_ceiling_leakage(frame, ceil)
    assert ok["pass"]
    # Intrusion
    frame2 = frame.copy()
    frame2[5:25, 50:100] = 200
    bad = measure_ceiling_leakage(frame2, ceil)
    assert bad["ceiling_leak_fraction"] > 0.001


def test_diagnostic_video_writer(tmp_path: Path):
    path = write_diagnostic_video(tmp_path / "diag.mp4", w=320, h=180, n_frames=5, fps=10)
    assert path.exists() and path.stat().st_size > 0
    fr = generate_diagnostic_frame(320, 180, 0.5)
    assert fr.shape == (180, 320, 3)


def test_play_video_cli_help():
    p = build_parser()
    text = p._subparsers._group_actions[0].choices["play-video"].format_help()
    assert "diagnostic" in text and "calibration-run" in text
    assert "contain" in text


def test_parse_aspect():
    assert abs(parse_aspect("16:9") - 16 / 9) < 1e-9
    assert abs(parse_aspect(1.0) - 1.0) < 1e-9
