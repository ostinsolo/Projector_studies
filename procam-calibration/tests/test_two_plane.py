"""Two-plane piecewise homography software gates (no physical devices)."""

from __future__ import annotations

import numpy as np

from procam_calibrate.charuco import apply_H
from procam_calibrate.synthetic_two_plane import (
    generate_one_plane_control,
    generate_two_plane_correspondences,
    run_synthetic_suite,
)
from procam_calibrate.two_plane import (
    build_plane_masks,
    compose_piecewise_prewarp,
    define_desired_per_plane,
    estimate_straight_seam,
    fit_single_homography,
    fit_two_homographies,
    run_two_plane_fit_from_points,
)


def test_one_plane_control_rejects_two_plane():
    sc = generate_one_plane_control(noise_px=0.0)
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    assert fit["model"] == "one_plane"
    assert fit["two_plane_hypothesis"] is False


def test_two_plane_assignment_and_single_h_fails():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(0))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    assert fit["model"] == "two_plane" and fit["valid"]
    pred = np.array(fit["labels"])
    true = sc["true_labels"]
    acc = max(float((pred == true).mean()), float((pred == 1 - true).mean()))
    assert acc >= 0.95
    assert fit["comparison"]["one_H_accepts_flat_wall"] is False
    _, err1, _ = fit_single_homography(sc["pts_proj"], sc["pts_cam"])
    assert float(np.percentile(err1, 95)) > 5.0


def test_masks_exclusive_no_holes():
    sc = generate_two_plane_correspondences(60.0, True, 0.0, rng=np.random.default_rng(1))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    ma, mb, viz = build_plane_masks(seam, 1920, 1080)
    assert int(np.sum((ma > 0) & (mb > 0))) == 0
    assert int(np.sum((ma == 0) & (mb == 0))) == 0
    assert viz.shape == (1080, 1920, 3)


def test_swapped_plane_matrices_detected():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(2))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    labels = np.array(fit["labels"])
    # Correct assignment error tiny; swapped should be huge on each side
    err_ok = np.linalg.norm(apply_H(Ha, sc["pts_proj"][labels == 0]) - sc["pts_cam"][labels == 0], axis=1)
    err_swap = np.linalg.norm(apply_H(Hb, sc["pts_proj"][labels == 0]) - sc["pts_cam"][labels == 0], axis=1)
    assert float(np.median(err_ok)) < 1.0
    assert float(np.median(err_swap)) > 10.0


def test_reversed_seam_side_flips_masks():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(3))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    ma, mb, _ = build_plane_masks(seam, 1920, 1080)
    seam_rev = dict(seam)
    a, b, c = seam["line_abc"]
    seam_rev["line_abc"] = [-a, -b, -c]
    ma2, mb2, _ = build_plane_masks(seam_rev, 1920, 1080)
    assert np.array_equal(ma > 0, mb2 > 0)
    assert np.array_equal(mb > 0, ma2 > 0)


def test_piecewise_compose_meta_exclusive():
    sc = generate_two_plane_correspondences(90.0, False, 0.0, rng=np.random.default_rng(4))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    ma, mb, _ = build_plane_masks(seam, 1920, 1080)
    Hda = define_desired_per_plane(Ha, sc["pts_proj"][labels == 0], 1920, 1080)
    Hdb = define_desired_per_plane(Hb, sc["pts_proj"][labels == 1], 1920, 1080)
    content = np.zeros((1080, 1920, 3), np.uint8)
    content[:] = (10, 20, 30)
    out, _, _, meta = compose_piecewise_prewarp(content, Ha, Hb, ma, mb, Hda, Hdb, 1920, 1080)
    assert out.shape == (1080, 1920, 3)
    assert meta["exclusive"] is True
    assert meta["overlap_pixels"] == 0
    assert meta["hole_pixels"] == 0


def test_wrong_composition_order_breaks_desired_chain():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(5))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    Ha = fit["H_cam_from_proj_plane_A"]
    labels = np.array(fit["labels"])
    Hda = define_desired_per_plane(Ha, sc["pts_proj"][labels == 0], 1920, 1080)
    from procam_calibrate.homography_baseline import forward_H_proj_from_source

    H_fwd = forward_H_proj_from_source(Ha, Hda)
    pts = sc["pts_proj"][labels == 0][:8]
    ok = apply_H(Ha, apply_H(H_fwd, pts))
    want = apply_H(Hda, pts)
    assert float(np.median(np.linalg.norm(ok - want, axis=1))) < 1e-4
    # Sampling matrix mistaken for forward point matrix
    H_wrong = np.linalg.inv(H_fwd)
    bad = apply_H(Ha, apply_H(H_wrong, pts))
    assert float(np.median(np.linalg.norm(bad - want, axis=1))) > 5.0
    assert float(np.median(np.linalg.norm(ok - want, axis=1))) < float(
        np.median(np.linalg.norm(bad - want, axis=1))
    )


def test_single_h_on_two_plane_data_fails_gate():
    sc = generate_two_plane_correspondences(60.0, True, 0.0, rng=np.random.default_rng(6))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    assert fit["comparison"]["one_H_accepts_flat_wall"] is False


def test_full_synthetic_suite(tmp_path):
    summary = run_synthetic_suite(tmp_path / "suite")
    assert summary["all_pass"] is True
    assert summary["n_pass"] == summary["n_cases"]


def test_run_writes_required_artifacts(tmp_path):
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(7))
    out = tmp_path / "tp"
    run_two_plane_fit_from_points(sc["pts_proj"], sc["pts_cam"], sc["ids"], out)
    for name in (
        "H_cam_from_proj_plane_A.npy",
        "H_cam_from_proj_plane_B.npy",
        "plane_assignment_by_id.json",
        "single_vs_two_model_comparison.json",
        "plane_masks_projfb.png",
        "seam_model.json",
        "H_proj_from_source_plane_A.npy",
        "H_proj_from_source_plane_B.npy",
        "prewarp_piecewise.png",
    ):
        assert (out / name).exists(), name
