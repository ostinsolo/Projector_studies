"""Architectural scene analysis and coordinate-convention gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from procam_calibrate.architecture_analysis import (
    analyse_architecture,
    propose_target_region,
    render_synthetic_two_wall_scene,
    verify_seam_with_architecture_prior,
    TargetRegionConfig,
)
from procam_calibrate.coordinate_conventions import (
    detect_mirrored_or_inverted,
    verify_forward_composition,
    wrong_direction_error,
)
from procam_calibrate.homography_baseline import forward_H_proj_from_source
from procam_calibrate.synthetic_two_plane import (
    generate_two_plane_correspondences,
    run_extended_synthetic_suite,
    run_synthetic_suite,
)
from procam_calibrate.two_plane import (
    define_desired_per_plane,
    estimate_straight_seam,
    fit_two_homographies,
)


def test_arch_detects_central_fold(tmp_path: Path):
    img, gt = render_synthetic_two_wall_scene(seam_x_frac=0.5, rng=np.random.default_rng(0))
    report = analyse_architecture(img, tmp_path / "arch")
    assert (tmp_path / "arch" / "architectural_overlay.png").exists()
    active = [c for c in report["seam_candidates"] if c["status"] == "candidate"]
    assert active
    assert abs(active[0]["mid_x"] - gt["seam_x"]) < 0.1 * img.shape[1]


def test_arch_shadow_not_sole_authority(tmp_path: Path):
    img, gt = render_synthetic_two_wall_scene(
        seam_x_frac=0.55, add_shadow_line=True, rng=np.random.default_rng(1)
    )
    report = analyse_architecture(img, tmp_path / "arch")
    assert report["authority"] == "prior_only"
    # Correspondence seam wins even if prior disagrees
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(2))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    labels = np.array(fit["labels"])
    seam = estimate_straight_seam(
        sc["pts_proj"],
        labels,
        1920,
        1080,
        fit["H_cam_from_proj_plane_A"],
        fit["H_cam_from_proj_plane_B"],
    )
    ver = verify_seam_with_architecture_prior(
        seam, report["seam_candidates"], 1920, cam_w=img.shape[1]
    )
    assert ver.get("authority") == "correspondence_seam_final" or ver.get("used_prior") is False


def test_propose_target_region_margins():
    img, _ = render_synthetic_two_wall_scene()
    report = analyse_architecture(img)
    region = propose_target_region(img.shape[:2], report["seam_candidates"], TargetRegionConfig())
    poly = np.array(region["camera_polygon"])
    assert poly[:, 0].min() >= 0 and poly[:, 0].max() < img.shape[1]
    assert region["coordinate_space"].startswith("camera")


def test_forward_composition_and_wrong_direction():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(3))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    Ha = fit["H_cam_from_proj_plane_A"]
    labels = np.array(fit["labels"])
    H_des = define_desired_per_plane(Ha, sc["pts_proj"][labels == 0], 1920, 1080)
    pts = np.array([[50, 50], [900, 50], [900, 500], [50, 500]], dtype=np.float64)
    assert verify_forward_composition(Ha, H_des, pts)["ok"]
    H_fwd = forward_H_proj_from_source(Ha, H_des)
    assert detect_mirrored_or_inverted(H_fwd, 1920, 1080)["ok"]
    wd = wrong_direction_error(Ha, sc["pts_proj"][labels == 0][:12], sc["pts_cam"][labels == 0][:12])
    assert wd["wrong_is_worse"]


def test_core_and_extended_synthetic_suites(tmp_path: Path):
    core = run_synthetic_suite(tmp_path / "core")
    assert core["all_pass"] and core["n_cases"] == 14
    ext = run_extended_synthetic_suite(tmp_path / "ext")
    assert ext["core_14_all_pass"]
    assert ext["all_pass"], [c for c in ext["cases"] if not c["pass"]]


def test_mirrored_homography_flagged():
    H_flip = np.array([[-1, 0, 1920], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    mir = detect_mirrored_or_inverted(H_flip, 1920, 1080)
    assert mir["mirrored_x"] is True
    assert mir["ok"] is False


def test_architecture_sm_does_not_touch_root_project_state(tmp_path: Path):
    from procam_calibrate.auto_two_plane import AutoTwoPlane, AutoTwoPlaneConfig

    repo = Path(__file__).resolve().parents[2]
    repo_ps = repo / "PROJECT_STATE.md"
    docs_setup = repo / "docs" / "TWO_PLANE_PHYSICAL_SETUP_REQUEST.md"
    before_ps = repo_ps.read_text() if repo_ps.exists() else None
    before_docs = docs_setup.read_text() if docs_setup.exists() else None
    cfg = AutoTwoPlaneConfig(
        run_dir=tmp_path / "iso",
        setup_confirmed=False,
        offline=False,
        two_plane_state_path=tmp_path / "local.md",
        project_state_path=None,
    )
    AutoTwoPlane(cfg).run()
    after_ps = repo_ps.read_text() if repo_ps.exists() else None
    after_docs = docs_setup.read_text() if docs_setup.exists() else None
    assert before_ps == after_ps
    assert before_docs == after_docs


def test_state_machine_offline_architecture_stages(tmp_path: Path):
    from procam_calibrate.auto_two_plane import AutoTwoPlane, AutoTwoPlaneConfig
    import json

    run = tmp_path / "sm"
    cfg = AutoTwoPlaneConfig(
        run_dir=run,
        mode="physical",
        setup_confirmed=True,
        offline=True,
        two_plane_state_path=tmp_path / "tp.md",
    )
    # Seed obs so later stages can proceed if we advance past architecture
    from procam_calibrate.synthetic_two_plane import generate_two_plane_correspondences

    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(9))
    keys = [f"{i}:{i % 40}" for i in range(len(sc["ids"]))]
    obs = {
        keys[i]: {
            "pattern_index": int(keys[i].split(":")[0]),
            "corner_id": int(keys[i].split(":")[1]),
            "proj": sc["pts_proj"][i].tolist(),
            "cam": sc["pts_cam"][i].tolist(),
        }
        for i in range(len(keys))
    }
    (run / "correspondences").mkdir(parents=True)
    (run / "correspondences" / "aggregated_obs.json").write_text(json.dumps(obs))

    m = AutoTwoPlane(cfg)
    # Run only through propose target
    m.step_preflight()
    assert m.state["state"] == "DISCOVER_PROJECTOR"
    m.step_discover_projector()
    m.step_discover_camera()
    m.step_capture_empty_scene()
    m.step_analyse_architecture()
    m.step_propose_target_region()
    assert m.state["state"] == "VERIFY_TWO_PLANE_SETUP"
    assert (run / "architecture" / "architectural_overlay.png").exists()
    assert (run / "architecture" / "proposed_target_region.json").exists()
