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
    }
    (out_dir / "synthetic_suite_results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
