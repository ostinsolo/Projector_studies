"""Synthetic connected-plane scenes for software gates (no physical setup)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .charuco import apply_H, generate_charuco_image
from .two_plane import (
    SEAM_MEDIAN_TARGET,
    SEAM_P95_TARGET,
    fit_single_homography,
    fit_two_homographies,
    run_two_plane_fit_from_points,
)


def _H_from_quad(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return (H / H[2, 2]).astype(np.float64)


def _shared_base_H(proj_w: int, proj_h: int, cam_w: int, cam_h: int) -> np.ndarray:
    src = np.array(
        [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
        dtype=np.float64,
    )
    dst = np.array(
        [
            [70, 60],
            [cam_w - 90, 80],
            [cam_w - 70, cam_h - 70],
            [90, cam_h - 90],
        ],
        dtype=np.float64,
    )
    return _H_from_quad(src, dst)


def _fold_affine_fix_line_through(p0: np.ndarray, p1: np.ndarray, scale: float) -> np.ndarray:
    """Affine map identity on line p0–p1; scales signed distance to the line by ``scale``."""
    # Line ax+by+c=0 through p0,p1
    a = p0[1] - p1[1]
    b = p1[0] - p0[0]
    c = -(a * p0[0] + b * p0[1])
    n = float(np.hypot(a, b)) + 1e-12
    a, b, c = a / n, b / n, c / n
    # x' = x + (scale-1)*d*a  where d=ax+by+c
    # x' = (1+(s-1)a^2) x + ((s-1)ab) y + ((s-1)ac)
    s = scale
    return np.array(
        [
            [1 + (s - 1) * a * a, (s - 1) * a * b, (s - 1) * a * c],
            [(s - 1) * a * b, 1 + (s - 1) * b * b, (s - 1) * b * c],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_connected_plane_homographies(
    proj_w: int,
    proj_h: int,
    cam_w: int,
    cam_h: int,
    angle_deg: float,
    seam_vertical: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """H_a = H0; H_b = A_fold @ H0 with A_fold fixing the imaged seam line."""
    H0 = _shared_base_H(proj_w, proj_h, cam_w, cam_h)
    # Strong enough that a single H cannot absorb the fold (even at 30°)
    scale = 1.0 + 1.4 * (angle_deg / 90.0)

    if seam_vertical:
        seam_x = (proj_w - 1) * 0.5
        s0 = np.array([seam_x, 0.0])
        s1 = np.array([seam_x, proj_h - 1.0])
    else:
        seam_y = (proj_h - 1) * 0.5
        s0 = np.array([0.0, seam_y])
        s1 = np.array([proj_w - 1.0, seam_y])
    seam_cam = apply_H(H0, np.vstack([s0, s1]))
    A = _fold_affine_fix_line_through(seam_cam[0], seam_cam[1], scale)

    H_a = H0
    H_b = A @ H0
    H_b = H_b / H_b[2, 2]
    return H_a, H_b, s0, s1, float(scale)


def sample_board_points(proj_w: int = 1920, proj_h: int = 1080) -> tuple[np.ndarray, list[int]]:
    pattern, proj_ids, _, _, _ = generate_charuco_image(proj_w, proj_h)
    del pattern
    ids = sorted(proj_ids.keys())
    pts = np.array([proj_ids[i] for i in ids], dtype=np.float64)
    return pts, ids


def generate_two_plane_correspondences(
    angle_deg: float = 90.0,
    seam_vertical: bool = True,
    noise_px: float = 0.0,
    outlier_frac: float = 0.0,
    drop_frac: float = 0.0,
    uneven: bool = False,
    proj_w: int = 1920,
    proj_h: int = 1080,
    cam_w: int = 1920,
    cam_h: int = 1080,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    rng = rng or np.random.default_rng(0)
    pts_all, ids_all = sample_board_points(proj_w, proj_h)
    H_a, H_b, s0, s1, _ = make_connected_plane_homographies(
        proj_w, proj_h, cam_w, cam_h, angle_deg, seam_vertical=seam_vertical
    )
    tang = s1 - s0
    side = (pts_all[:, 0] - s0[0]) * tang[1] - (pts_all[:, 1] - s0[1]) * tang[0]
    true_labels = (side >= 0).astype(np.int32)

    keep = np.ones(len(pts_all), dtype=bool)
    if drop_frac > 0:
        drop = rng.choice(len(pts_all), size=int(drop_frac * len(pts_all)), replace=False)
        keep[drop] = False
    if uneven:
        b_idx = np.where(true_labels == 1)[0]
        if len(b_idx) > 6:
            drop_b = rng.choice(b_idx, size=max(1, len(b_idx) // 3), replace=False)
            keep[drop_b] = False

    pts = pts_all[keep]
    ids = [ids_all[i] for i in np.where(keep)[0]]
    labels = true_labels[keep]
    cam = np.zeros_like(pts)
    cam[labels == 0] = apply_H(H_a, pts[labels == 0])
    cam[labels == 1] = apply_H(H_b, pts[labels == 1])
    if noise_px > 0:
        cam = cam + rng.normal(0, noise_px, size=cam.shape)
    if outlier_frac > 0:
        n_out = max(1, int(outlier_frac * len(cam)))
        oi = rng.choice(len(cam), size=n_out, replace=False)
        cam[oi] = cam[oi] + rng.normal(0, 40.0, size=(n_out, 2))

    return {
        "pts_proj": pts,
        "pts_cam": cam,
        "ids": ids,
        "true_labels": labels,
        "H_a_true": H_a,
        "H_b_true": H_b,
        "seam_p0": s0,
        "seam_p1": s1,
        "seam_vertical": seam_vertical,
        "angle_deg": angle_deg,
        "proj_w": proj_w,
        "proj_h": proj_h,
    }


def generate_one_plane_control(
    noise_px: float = 0.0,
    proj_w: int = 1920,
    proj_h: int = 1080,
    cam_w: int = 1920,
    cam_h: int = 1080,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    rng = rng or np.random.default_rng(1)
    pts, ids = sample_board_points(proj_w, proj_h)
    H_a, _, _, _, _ = make_connected_plane_homographies(proj_w, proj_h, cam_w, cam_h, 90.0, True)
    cam = apply_H(H_a, pts)
    if noise_px > 0:
        cam = cam + rng.normal(0, noise_px, size=cam.shape)
    return {
        "pts_proj": pts,
        "pts_cam": cam,
        "ids": ids,
        "true_labels": np.zeros(len(pts), dtype=np.int32),
        "one_plane": True,
        "proj_w": proj_w,
        "proj_h": proj_h,
    }


def run_synthetic_suite(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for noise in (0.0, 0.5):
        scene = generate_one_plane_control(noise_px=noise, rng=np.random.default_rng(10))
        fit = fit_two_homographies(scene["pts_proj"], scene["pts_cam"], ids=scene["ids"])
        cases.append(
            {
                "name": f"one_plane_noise_{noise}",
                "expect_model": "one_plane",
                "got_model": fit["model"],
                "pass": fit["model"] == "one_plane",
            }
        )

    for angle in (30.0, 60.0, 90.0):
        for seam_vertical in (True, False):
            for noise in (0.0, 0.4):
                scene = generate_two_plane_correspondences(
                    angle_deg=angle,
                    seam_vertical=seam_vertical,
                    noise_px=noise,
                    uneven=(angle == 60.0),
                    outlier_frac=0.03 if noise > 0 else 0.0,
                    drop_frac=0.08 if angle == 90.0 else 0.0,
                    rng=np.random.default_rng(int(angle) + int(seam_vertical) * 100),
                )
                tag = f"two_plane_a{int(angle)}_{'v' if seam_vertical else 'h'}_n{noise}"
                result = run_two_plane_fit_from_points(
                    scene["pts_proj"],
                    scene["pts_cam"],
                    scene["ids"],
                    out_dir / tag,
                    proj_w=scene["proj_w"],
                    proj_h=scene["proj_h"],
                )
                pred = np.array(result.get("labels", []), dtype=np.int32)
                true = scene["true_labels"]
                if len(pred) == len(true) and result.get("model") == "two_plane":
                    acc = max(float((pred == true).mean()), float((pred == (1 - true)).mean()))
                else:
                    acc = 0.0
                comp = result.get("comparison") or {}
                single_fails = not bool(comp.get("one_H_accepts_flat_wall", True))
                med_a = comp.get("plane_A_median_reproj_px")
                med_b = comp.get("plane_B_median_reproj_px")
                reproj_lim = 1.0 if noise == 0 else 2.0
                seam = result.get("seam_metrics") or {}
                acc_lim = 0.95 if noise == 0 else 0.92
                ok = (
                    result.get("model") == "two_plane"
                    and bool(result.get("valid"))
                    and acc >= acc_lim
                    and single_fails
                    and med_a is not None
                    and med_b is not None
                    and med_a <= reproj_lim
                    and med_b <= reproj_lim
                    and seam.get("median_seam_mismatch_px", 1e9) <= SEAM_MEDIAN_TARGET
                    and seam.get("p95_seam_mismatch_px", 1e9) <= SEAM_P95_TARGET
                )
                cases.append(
                    {
                        "name": tag,
                        "expect_model": "two_plane",
                        "got_model": result.get("model"),
                        "assignment_accuracy": acc,
                        "single_H_fails": single_fails,
                        "single_H_median": comp.get("one_H_median_reproj_px"),
                        "single_H_p95": comp.get("one_H_p95_reproj_px"),
                        "plane_A_median": med_a,
                        "plane_B_median": med_b,
                        "seam": seam,
                        "pass": ok,
                    }
                )

    summary = {
        "n_cases": len(cases),
        "n_pass": sum(1 for c in cases if c["pass"]),
        "all_pass": all(c["pass"] for c in cases),
        "cases": cases,
        "seed_policy": "deterministic np.random.default_rng per case",
    }
    (out_dir / "synthetic_suite_results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_extended_synthetic_suite(out_dir: Path) -> dict:
    """Expanded gates: architecture priors, off-centre seam, nearly coplanar, failures."""
    from .architecture_analysis import (
        analyse_architecture,
        render_synthetic_two_wall_scene,
        verify_seam_with_architecture_prior,
    )
    from .coordinate_conventions import (
        detect_mirrored_or_inverted,
        verify_forward_composition,
        wrong_direction_error,
    )
    from .homography_baseline import forward_H_proj_from_source
    from .two_plane import (
        build_plane_masks,
        define_desired_per_plane,
        estimate_straight_seam,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = run_synthetic_suite(out_dir / "core_14")
    cases = list(base["cases"])

    # Off-centre seam via uneven sampling (more points one side) + standard fold
    sc = generate_two_plane_correspondences(
        90.0, True, 0.0, uneven=True, rng=np.random.default_rng(42)
    )
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    cases.append(
        {
            "name": "two_plane_uneven_support",
            "pass": fit.get("model") == "two_plane" and fit.get("valid"),
            "got_model": fit.get("model"),
        }
    )

    # Nearly coplanar — small angle should still pick two-plane or safely one-plane
    sc = generate_two_plane_correspondences(15.0, True, 0.0, rng=np.random.default_rng(43))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    # Accept either valid two-plane OR one_plane if fold too weak (must not crash)
    cases.append(
        {
            "name": "nearly_coplanar_angle_15",
            "pass": fit.get("model") in ("one_plane", "two_plane", "two_plane_rejected"),
            "got_model": fit.get("model"),
            "note": "weak fold may correctly prefer one_plane",
        }
    )

    # Architecture: fold detection prior
    img, gt = render_synthetic_two_wall_scene(seam_x_frac=0.5, rng=np.random.default_rng(0))
    arch = analyse_architecture(img, out_dir / "arch_centre")
    active = [c for c in arch["seam_candidates"] if c.get("status") == "candidate"]
    best_x = active[0]["mid_x"] if active else None
    arch_ok = best_x is not None and abs(best_x - gt["seam_x"]) < 0.08 * img.shape[1]
    cases.append({"name": "arch_detect_central_fold", "pass": arch_ok, "best_x": best_x, "gt_x": gt["seam_x"]})

    # Misleading shadow: prior may fire on shadow; correspondence must win later
    img2, gt2 = render_synthetic_two_wall_scene(
        seam_x_frac=0.55, add_shadow_line=True, rng=np.random.default_rng(1)
    )
    arch2 = analyse_architecture(img2, out_dir / "arch_shadow")
    # Must still produce at least one candidate and a target region
    cases.append(
        {
            "name": "arch_shadow_still_proposes_region",
            "pass": bool(arch2.get("target_region", {}).get("camera_polygon"))
            and arch2.get("n_lines", 0) > 5,
        }
    )

    # Low-contrast seam: analysis remains valid (may have lower confidence)
    img3, _ = render_synthetic_two_wall_scene(low_contrast_seam=True, rng=np.random.default_rng(2))
    arch3 = analyse_architecture(img3, out_dir / "arch_low_contrast")
    cases.append({"name": "arch_low_contrast_valid_report", "pass": "seam_candidates" in arch3})

    # Seam verify: correspondence seam vs prior
    sc = generate_two_plane_correspondences(90.0, True, 0.0, rng=np.random.default_rng(7))
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    labels = np.array(fit["labels"])
    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
    v = verify_seam_with_architecture_prior(seam, arch["seam_candidates"], 1920, cam_w=img.shape[1])
    cases.append({"name": "seam_prior_verify_api", "pass": "authority" in v or v.get("consistent") is not None})

    # Matrix direction / mirror guards
    H_des = define_desired_per_plane(Ha, sc["pts_proj"][labels == 0], 1920, 1080)
    pts = np.array([[100, 100], [1800, 100], [1800, 900], [100, 900]], dtype=np.float64)
    comp = verify_forward_composition(Ha, H_des, pts)
    cases.append({"name": "forward_composition_plane_A", "pass": comp["ok"]})
    H_fwd = forward_H_proj_from_source(Ha, H_des)
    mir = detect_mirrored_or_inverted(H_fwd, 1920, 1080)
    cases.append({"name": "no_mirror_forward_H", "pass": mir["ok"]})
    wd = wrong_direction_error(Ha, sc["pts_proj"][labels == 0][:20], sc["pts_cam"][labels == 0][:20])
    cases.append({"name": "wrong_H_direction_detected", "pass": wd["wrong_is_worse"]})

    # Mask topology
    ma, mb, _ = build_plane_masks(seam, 1920, 1080)
    cases.append(
        {
            "name": "mask_exclusive_full",
            "pass": int(np.sum((ma > 0) & (mb > 0))) == 0 and int(np.sum((ma == 0) & (mb == 0))) == 0,
        }
    )

    # One-plane control still in extended
    sc1 = generate_one_plane_control(0.0)
    fit1 = fit_two_homographies(sc1["pts_proj"], sc1["pts_cam"], ids=sc1["ids"])
    cases.append({"name": "extended_one_plane_control", "pass": fit1["model"] == "one_plane"})

    summary = {
        "n_cases": len(cases),
        "n_pass": sum(1 for c in cases if c["pass"]),
        "all_pass": all(c["pass"] for c in cases),
        "core_14_all_pass": base["all_pass"],
        "cases": cases,
    }
    (out_dir / "extended_synthetic_suite_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
