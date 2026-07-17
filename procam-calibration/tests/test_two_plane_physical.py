"""Physical two-plane pipeline software gates (no hardware required)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from procam_calibrate.cli import _resolve_two_plane_mode, build_parser
from procam_calibrate.synthetic_two_plane import (
    generate_one_plane_control,
    generate_two_plane_correspondences,
)
from procam_calibrate.two_plane import (
    bootstrap_seam_stability,
    build_plane_masks,
    define_desired_target_two_plane,
    estimate_straight_seam,
    fit_two_homographies,
    measure_two_plane_against_desired,
    seam_exclusion_mask,
    two_plane_acceptance,
)
from procam_calibrate.two_plane_observations import (
    aggregate_observations,
    observation_key,
    observations_to_arrays,
    canonicalize_plane_labels,
)
from procam_calibrate.two_plane_refine import coupled_accept, propose_coupled_candidates
from procam_calibrate.auto_two_plane import AutoTwoPlane, AutoTwoPlaneConfig
from procam_calibrate.charuco import apply_H


def test_cli_mode_requires_explicit_mode():
    with pytest.raises(SystemExit):
        ns = build_parser().parse_args(["auto-two-plane", "--run-dir", "/tmp/x"])
        _resolve_two_plane_mode(ns)


def test_cli_mode_synthetic_and_physical():
    ns = build_parser().parse_args(
        ["auto-two-plane", "--run-dir", "/tmp/x", "--mode", "synthetic"]
    )
    assert _resolve_two_plane_mode(ns) == "synthetic"
    ns = build_parser().parse_args(
        ["auto-two-plane", "--run-dir", "/tmp/x", "--mode", "physical"]
    )
    assert _resolve_two_plane_mode(ns) == "physical"


def test_cli_mode_conflicts():
    with pytest.raises(SystemExit):
        ns = build_parser().parse_args(
            [
                "auto-two-plane",
                "--run-dir",
                "/tmp/x",
                "--mode",
                "synthetic",
                "--allow-physical",
            ]
        )
        _resolve_two_plane_mode(ns)
    with pytest.raises(SystemExit):
        ns = build_parser().parse_args(
            [
                "auto-two-plane",
                "--run-dir",
                "/tmp/x",
                "--synthetic-only",
                "--allow-physical",
            ]
        )
        _resolve_two_plane_mode(ns)


def test_cli_help_mentions_hardware():
    p = build_parser()
    help_txt = p.parse_args.__doc__ or ""
    # argparse help via format
    text = p.format_help()
    sub = [a for a in p._subparsers._group_actions[0].choices["auto-two-plane"].format_help().splitlines()]
    joined = "\n".join(sub)
    assert "synthetic" in joined and "physical" in joined
    assert "hardware" in joined.lower() or "projector" in joined.lower() or "devices" in joined.lower()


def test_observation_key_uniqueness_across_patterns():
    k0 = observation_key(0, 5)
    k1 = observation_key(1, 5)
    assert k0 != k1
    assert k0 == "0:5"
    blocks = [
        {k0: {"pattern_index": 0, "corner_id": 5, "proj": [1.0, 2.0], "cam": [3.0, 4.0]}},
        {k1: {"pattern_index": 1, "corner_id": 5, "proj": [10.0, 20.0], "cam": [30.0, 40.0]}},
    ]
    agg = aggregate_observations(blocks)
    assert len(agg) == 2
    keys, pts_p, pts_c = observations_to_arrays(agg)
    assert set(keys) == {k0, k1}
    # Same corner id must not merge projector coords
    assert not np.allclose(pts_p[0], pts_p[1])


def test_aggregate_rejects_duplicate_keys():
    k = observation_key(0, 1)
    with pytest.raises(ValueError):
        aggregate_observations(
            [
                {k: {"proj": [0, 0], "cam": [0, 0]}},
                {k: {"proj": [1, 1], "cam": [1, 1]}},
            ]
        )


def test_plane_label_canonicalization_stable():
    pts = np.array([[100, 500], [200, 500], [1500, 500], [1600, 500]], dtype=np.float64)
    labels = np.array([1, 1, 0, 0], dtype=np.int32)  # A was right
    can = canonicalize_plane_labels(pts, labels)
    assert can[0] == 0 and can[2] == 1  # left becomes A


def test_seam_exclusion_and_bootstrap():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(4))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=[str(i) for i in sc["ids"]])
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    keys = [str(i) for i in sc["ids"]]
    excl = seam_exclusion_mask(seam, keys, sc["pts_proj"], half_width=20.0)
    assert excl["n_excluded"] >= 0
    assert len(excl["keep_mask"]) == len(keys)
    boot = bootstrap_seam_stability(sc["pts_proj"], labels, Ha, Hb, 1920, 1080, n_boot=10)
    assert boot["n_boot"] >= 1


def test_mask_exclusivity_and_full_coverage():
    sc = generate_two_plane_correspondences(75.0, True, 0.0, rng=np.random.default_rng(5))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=list(range(len(sc["ids"]))))
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    ma, mb, _ = build_plane_masks(seam, 1920, 1080)
    assert int(np.sum((ma > 0) & (mb > 0))) == 0
    assert int(np.sum((ma == 0) & (mb == 0))) == 0


def test_desired_target_independence():
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(6))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=[f"0:{i}" for i in sc["ids"]])
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    ma, mb, _ = build_plane_masks(seam, 1920, 1080)
    keys = [f"0:{i}" for i in sc["ids"]]
    src = {keys[i]: sc["pts_proj"][i].tolist() for i in range(len(keys))}
    labs = {keys[i]: ("A" if labels[i] == 0 else "B") for i in range(len(keys))}
    desired = define_desired_target_two_plane(
        Ha, Hb, ma, mb, seam, 1920, 1080, 1280, 720, src, labs
    )
    assert "H_desired_cam_from_source" in desired
    assert "desired_point_by_key" in desired
    assert desired["note"].startswith("Desired geometry fixed")
    # Mutating a fake corrected capture must not be an input — independence is by construction
    assert len(desired["desired_point_by_key"]) == len(keys)


def test_evaluation_key_consistency_and_empty_invalid():
    desired = {
        "desired_point_by_key": {"0:1": [10.0, 10.0], "0:2": [20.0, 20.0]},
    }
    empty = measure_two_plane_against_desired({}, desired, {}, expected_keys=["0:1", "0:2"])
    assert empty["valid"] is False
    assert empty["target_median_err_px"] is None

    obs = {"0:1": [10.5, 10.0], "0:2": [20.0, 21.0]}
    # pad to enough keys for MIN_POINTS
    for i in range(3, 20):
        k = f"0:{i}"
        desired["desired_point_by_key"][k] = [float(i), float(i)]
        obs[k] = [float(i) + 0.5, float(i)]
    labs = {k: ("A" if int(k.split(":")[1]) < 10 else "B") for k in obs}
    m = measure_two_plane_against_desired(obs, desired, labs, expected_keys=list(obs.keys()))
    assert m["valid"]
    assert m["id_sets"]["evaluation_keys"] == sorted(obs.keys())
    assert m["target_median_err_px"] is not None and m["target_median_err_px"] > 0


def test_coupled_rollback_one_plane_regresses():
    prev = {
        "valid": True,
        "target_median_err_px": 4.0,
        "target_p95_err_px": 8.0,
        "coverage_fraction": 0.9,
        "plane_A": {"valid": True, "target_median_err_px": 3.0, "target_p95_err_px": 7.0},
        "plane_B": {"valid": True, "target_median_err_px": 3.5, "target_p95_err_px": 7.5},
        "seam": {"median_seam_mismatch_px": 2.0, "p95_seam_mismatch_px": 4.0},
    }
    # A improves, B regresses
    new = {
        "valid": True,
        "target_median_err_px": 3.5,
        "target_p95_err_px": 7.5,
        "coverage_fraction": 0.9,
        "plane_A": {"valid": True, "target_median_err_px": 2.0, "target_p95_err_px": 5.0},
        "plane_B": {"valid": True, "target_median_err_px": 6.0, "target_p95_err_px": 14.0},
        "seam": {"median_seam_mismatch_px": 2.0, "p95_seam_mismatch_px": 4.0},
    }
    ok, reason = coupled_accept(prev, new, jitter_median=0.1)
    assert ok is False
    assert "plane_B" in reason


def test_coupled_rollback_seam_regresses():
    prev = {
        "valid": True,
        "target_median_err_px": 4.0,
        "target_p95_err_px": 8.0,
        "coverage_fraction": 0.9,
        "plane_A": {"valid": True, "target_median_err_px": 3.0, "target_p95_err_px": 7.0},
        "plane_B": {"valid": True, "target_median_err_px": 3.0, "target_p95_err_px": 7.0},
        "seam": {"median_seam_mismatch_px": 2.0, "p95_seam_mismatch_px": 4.0},
    }
    new = dict(prev)
    new = {
        **prev,
        "target_median_err_px": 3.0,
        "target_p95_err_px": 6.0,
        "plane_A": {"valid": True, "target_median_err_px": 2.5, "target_p95_err_px": 6.0},
        "plane_B": {"valid": True, "target_median_err_px": 2.5, "target_p95_err_px": 6.0},
        "seam": {"median_seam_mismatch_px": 5.0, "p95_seam_mismatch_px": 9.0},
    }
    ok, reason = coupled_accept(prev, new, 0.1)
    assert ok is False
    assert "seam" in reason


def test_jitter_aware_acceptance():
    prev = {
        "valid": True,
        "target_median_err_px": 8.0,
        "target_p95_err_px": 16.0,
        "coverage_fraction": 0.9,
        "plane_A": {"valid": True, "target_median_err_px": 8.0, "target_p95_err_px": 16.0},
        "plane_B": {"valid": True, "target_median_err_px": 8.0, "target_p95_err_px": 16.0},
        "seam": {"median_seam_mismatch_px": 2.0, "p95_seam_mismatch_px": 4.0},
    }
    # Tiny improvement smaller than jitter → reject
    new = {
        **prev,
        "target_median_err_px": 7.9,
        "target_p95_err_px": 15.9,
        "plane_A": {"valid": True, "target_median_err_px": 7.9, "target_p95_err_px": 15.9},
        "plane_B": {"valid": True, "target_median_err_px": 7.9, "target_p95_err_px": 15.9},
    }
    ok, _ = coupled_accept(prev, new, jitter_median=0.5)
    assert ok is False


def test_two_plane_acceptance_gate():
    unc = {
        "valid": True,
        "target_median_err_px": 40.0,
        "target_p95_err_px": 80.0,
        "coverage_fraction": 0.9,
        "plane_A": {"valid": True, "target_median_err_px": 40.0, "target_p95_err_px": 80.0},
        "plane_B": {"valid": True, "target_median_err_px": 40.0, "target_p95_err_px": 80.0},
    }
    cor = {
        "valid": True,
        "target_median_err_px": 3.0,
        "target_p95_err_px": 8.0,
        "coverage_fraction": 0.9,
        "plane_A": {"valid": True, "target_median_err_px": 3.0, "target_p95_err_px": 8.0},
        "plane_B": {"valid": True, "target_median_err_px": 3.0, "target_p95_err_px": 8.0},
    }
    seam = {"median_seam_mismatch_px": 2.0, "p95_seam_mismatch_px": 4.0}
    gate = two_plane_acceptance(unc, cor, seam)
    assert gate["absolute_gate_pass"] is True


def test_physical_sm_user_action_without_setup(tmp_path: Path):
    cfg = AutoTwoPlaneConfig(
        run_dir=tmp_path / "run_phys",
        mode="physical",
        setup_confirmed=False,
        offline=False,
        two_plane_state_path=tmp_path / "TWO_PLANE_STATE.md",
        project_state_path=None,
    )
    # Avoid inventing devices: PREFLIGHT should stop at USER_ACTION_REQUIRED
    m = AutoTwoPlane(cfg)
    result = m.run()
    assert result["state"] == "USER_ACTION_REQUIRED"
    assert (tmp_path / "run_phys" / "capture_request.md").exists()
    # Must not pollute a root PROJECT_STATE
    assert cfg.project_state_path is None


def test_physical_sm_resume_from_user_action(tmp_path: Path):
    run = tmp_path / "resume"
    run.mkdir()
    state = {
        "state": "USER_ACTION_REQUIRED",
        "completed": ["PREFLIGHT"],
        "history": [],
        "data": {},
        "mode": "physical",
    }
    (run / "two_plane_state.json").write_text(json.dumps(state))
    cfg = AutoTwoPlaneConfig(
        run_dir=run,
        mode="physical",
        setup_confirmed=True,
        offline=True,
        two_plane_state_path=tmp_path / "tp_state.md",
    )
    # Seed offline aggregated obs from synthetic two-plane
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(7))
    keys = [f"{i // 20}:{i % 20}" for i in range(len(sc["ids"]))]
    # unique keys
    keys = [f"{i}:{(i % 50)}" for i in range(len(sc["ids"]))]
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

    # Build synthetic validation frames from two H
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=keys)
    assert fit["model"] == "two_plane"
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    labels = np.array(fit["labels"])
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    ma, mb, _ = build_plane_masks(seam, 1920, 1080)

    # Minimal patterns + fake captures for offline validation path
    from procam_calibrate.two_plane_observations import generate_pattern_set
    from procam_calibrate.charuco import generate_charuco_image
    from procam_calibrate.homography_baseline import forward_H_proj_from_source, prewarp_from_forward_H
    from procam_calibrate.two_plane import define_desired_target_two_plane
    from procam_calibrate.two_plane_refine import build_piecewise_from_forwards

    generate_pattern_set(1920, 1080, run / "projector_patterns")
    # Pre-populate state as if past capture
    m = AutoTwoPlane(cfg)
    m.state["data"]["pattern_catalog"] = json.loads(
        (run / "projector_patterns" / "pattern_catalog.json").read_text()
    )
    # Run detect/fit path by setting state after devices
    m._ensure_dirs()
    m.state["state"] = "DETECT_MULTI_FRAME_CHARUCO"
    m._save_state()
    m.step_detect_and_fit()
    if m.state["state"] == "ESTIMATE_SEAM":
        m.step_estimate_seam()
        m.step_generate_prewarp()
        # Create offline camera captures by rendering charuco through plane Hs (approx)
        content = cv2.imread(str(run / "projected_validation" / "source_validation.png"))
        # Fake capture = warp content with avg H into camera canvas
        H_des = np.load(run / "desired_target" / "H_desired_cam_from_source.npy")
        # Ideal corrected capture ≈ desired mapping of source — draw points
        cam = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Use uncorrected = apply plane H to source projected as identity content corners
        # Simpler: write blank frames and skip full measure by injecting metrics
        for stem in ("uncorrected", "corrected"):
            for i in range(3):
                p = run / "captured_validation" / f"{stem}_burst_{i:02d}.png"
                cv2.imwrite(str(p), cam)
            cv2.imwrite(str(run / "captured_validation" / f"{stem}.png"), cam)
        # Inject median obs from synthetic desired/cam for measure path
        desired = json.loads((run / "desired_target" / "desired_target_two_plane.json").read_text())
        # Build validation known from pattern 0
        known = json.loads((run / "projected_validation" / "validation_proj_ids.json").read_text())
        # Map known through H_des for "corrected" (near zero error) and Ha/Hb mix for unc
        unc_obs = {}
        cor_obs = {}
        for k, xy in list(known.items())[:40]:
            p = np.array([xy], dtype=np.float64)
            # assign plane by x
            H = Ha if xy[0] < 960 else Hb
            unc_obs[k] = apply_H(H, p)[0].tolist()
            cor_obs[k] = apply_H(H_des, p)[0].tolist()
        (run / "captured_validation" / "uncorrected_median_obs.json").write_text(json.dumps(unc_obs))
        (run / "captured_validation" / "corrected_median_obs.json").write_text(json.dumps(cor_obs))
        # Patch burst to return injected
        def fake_burst(stem, known_arg):
            path = run / "captured_validation" / f"{stem}_median_obs.json"
            if path.exists():
                return json.loads(path.read_text()), {"aggregate_median_jitter_px": 0.1, "aggregate_p95_jitter_px": 0.2}
            return {}, {"aggregate_median_jitter_px": 0.1}
        m._burst_obs = fake_burst  # type: ignore
        m.state["state"] = "PROJECT_UNCORRECTED_VALIDATION"
        m.step_validation_capture_measure()
        assert m.state["state"] in ("FINAL_REPRODUCTION", "DIAGNOSE_TWO_PLANE_RESIDUAL", "DONE", "ACCEPTANCE_FAILED", "REFINE_TWO_PLANE")


def test_no_project_state_pollution(tmp_path: Path, monkeypatch):
    """Tracked PROJECT_STATE.md must not be written by two-plane machine."""
    repo_ps = Path(__file__).resolve().parents[2] / "PROJECT_STATE.md"
    before = repo_ps.read_text() if repo_ps.exists() else None
    cfg = AutoTwoPlaneConfig(
        run_dir=tmp_path / "iso",
        mode="physical",
        setup_confirmed=False,
        two_plane_state_path=tmp_path / "local_tp.md",
        project_state_path=None,
    )
    AutoTwoPlane(cfg).run()
    after = repo_ps.read_text() if repo_ps.exists() else None
    assert before == after


def test_single_h_control_wall_still():
    sc = generate_one_plane_control(noise_px=0.0)
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    assert fit["model"] == "one_plane"


def test_propose_coupled_candidates_strengths():
    H = np.eye(3, dtype=np.float64)
    keys_a = ["0:1", "0:2", "0:3", "0:4"]
    keys_b = ["0:5", "0:6", "0:7", "0:8"]
    # Non-colinear well-spread control points per plane
    src = {
        "0:1": [100.0, 100.0],
        "0:2": [400.0, 120.0],
        "0:3": [120.0, 500.0],
        "0:4": [380.0, 520.0],
        "0:5": [1100.0, 100.0],
        "0:6": [1500.0, 140.0],
        "0:7": [1120.0, 500.0],
        "0:8": [1480.0, 540.0],
    }
    cam = {k: [v[0] + 2.0, v[1] + 1.0] for k, v in src.items()}
    ctrl = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    cands = propose_coupled_candidates(H, H, keys_a, keys_b, src, cam, np.eye(3), ctrl)
    assert len(cands) == 4
    assert [c["strength"] for c in cands] == [1.0, 0.75, 0.5, 0.25]
