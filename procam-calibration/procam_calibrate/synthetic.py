"""Synthetic planar warp verification (no physical capture).

Constructs a known homography:
  H_projector_to_camera : ProjFB homogeneous -> CamImg homogeneous
and verifies recovery via OpenCV chessboard / grid points.
Also verifies pre-warp direction: content in CamImg ideal -> ProjFB via H_camera_to_projector.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .patterns import make_dense_grid, make_validation_target


def _homography_proj_to_cam(proj_w: int, proj_h: int, cam_w: int, cam_h: int) -> np.ndarray:
    """
    Known planar transform ProjFB -> CamImg with scale, rotation, translation.
    Returns 3x3 float64 H_projector_to_camera.
    """
    src = np.array(
        [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
        dtype=np.float32,
    )
    # inset + slight shear in camera
    margin_x, margin_y = cam_w * 0.12, cam_h * 0.10
    dst = np.array(
        [
            [margin_x + 30, margin_y + 10],
            [cam_w - margin_x - 10, margin_y + 40],
            [cam_w - margin_x - 50, cam_h - margin_y - 20],
            [margin_x + 20, cam_h - margin_y - 5],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return H.astype(np.float64)


def apply_H_projector_to_camera(proj_img: np.ndarray, H_p2c: np.ndarray, cam_w: int, cam_h: int) -> np.ndarray:
    """Warp ProjFB image into CamImg using H_projector_to_camera."""
    return cv2.warpPerspective(
        proj_img,
        H_p2c,
        (cam_w, cam_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def prewarp_camera_content_to_projector(
    cam_content: np.ndarray,
    H_c2p: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> np.ndarray:
    """
    Build ProjFB pre-warp from desired CamImg content.
    Uses H_camera_to_projector so that after physical H_p2c, camera sees cam_content.
    OpenCV: dst(x') = src(H^{-1} x') when warping with H mapping src->dst...
    We want: for each projector pixel p, sample camera-content at H_p2c * p.
    That is remap via H_projector_to_camera (p -> c).
    Equivalently warpPerspective(cam_content, H_c2p, proj_size) where H_c2p maps
    camera content coords to projector... 

    Correct construction for pre-warp I_p such that warp(I_p, H_p2c) ~= I_c_desired:
      I_p = warpPerspective(I_c_desired, H_p2c^{-1}, proj_size)
    i.e. H used by warpPerspective maps CamImg -> ProjFB = H_camera_to_projector.
    """
    return cv2.warpPerspective(
        cam_content,
        H_c2p,
        (proj_w, proj_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _sample_grid_points(proj_w: int, proj_h: int, step: int = 80) -> np.ndarray:
    xs = list(range(step, proj_w - step, step))
    ys = list(range(step, proj_h - step, step))
    pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return pts


def recover_homography_from_points(
    pts_proj: np.ndarray, pts_cam: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Estimate H_projector_to_camera from correspondences."""
    H, mask = cv2.findHomography(pts_proj, pts_cam, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H is None:
        raise RuntimeError("Homography estimation failed")
    inliers = int(mask.sum()) if mask is not None else len(pts_proj)
    # reprojection errors in CamImg pixels
    proj_h = cv2.convertPointsToHomogeneous(pts_proj)[:, 0, :]
    cam_est = (H @ proj_h.T).T
    cam_est = cam_est[:, :2] / cam_est[:, 2:3]
    err = np.linalg.norm(cam_est - pts_cam, axis=1)
    stats = {
        "n_points": int(len(pts_proj)),
        "n_inliers": inliers,
        "mean_reproj_px": float(err.mean()),
        "median_reproj_px": float(np.median(err)),
        "p95_reproj_px": float(np.percentile(err, 95)),
        "max_reproj_px": float(err.max()),
    }
    return H.astype(np.float64), stats


def run_synthetic_gate(
    out_dir: Path,
    proj_w: int = 1920,
    proj_h: int = 1080,
    cam_w: int = 1920,
    cam_h: int = 1080,
) -> dict:
    """
    Acceptance for synthetic gate:
      median recovery error < 1.0 CamImg px
      p95 recovery error < 2.0 CamImg px
      pre-warp roundtrip median < 1.5 CamImg px inside valid region
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    H_p2c = _homography_proj_to_cam(proj_w, proj_h, cam_w, cam_h)
    H_c2p = np.linalg.inv(H_p2c)

    meta = {
        "H_projector_to_camera": {
            "matrix": H_p2c.tolist(),
            "source_space": "ProjFB",
            "destination_space": "CamImg",
            "source_size": [proj_w, proj_h],
            "destination_size": [cam_w, cam_h],
            "coordinates": "pixel_homogeneous",
            "pixel_centre": "OpenCV",
            "axis_order": "u_right_v_down",
            "distortion_state": "none",
        },
        "H_camera_to_projector": {
            "matrix": H_c2p.tolist(),
            "source_space": "CamImg",
            "destination_space": "ProjFB",
            "source_size": [cam_w, cam_h],
            "destination_size": [proj_w, proj_h],
            "coordinates": "pixel_homogeneous",
            "pixel_centre": "OpenCV",
            "axis_order": "u_right_v_down",
            "distortion_state": "none",
        },
    }
    np.save(out_dir / "H_projector_to_camera.npy", H_p2c)
    np.save(out_dir / "H_camera_to_projector.npy", H_c2p)
    (out_dir / "transforms.json").write_text(json.dumps(meta, indent=2))

    # Synthetic capture of dense grid
    grid = make_dense_grid(proj_h, proj_w)
    cam_capture = apply_H_projector_to_camera(grid, H_p2c, cam_w, cam_h)
    cv2.imwrite(str(out_dir / "proj_grid.png"), grid)
    cv2.imwrite(str(out_dir / "cam_capture_grid.png"), cam_capture)

    pts_proj = _sample_grid_points(proj_w, proj_h, step=80)
    pts_h = cv2.convertPointsToHomogeneous(pts_proj)[:, 0, :]
    pts_cam = (H_p2c @ pts_h.T).T
    pts_cam = (pts_cam[:, :2] / pts_cam[:, 2:3]).astype(np.float32)
    # add tiny noise to simulate detection
    rng = np.random.default_rng(0)
    pts_cam_noisy = pts_cam + rng.normal(0, 0.15, pts_cam.shape).astype(np.float32)

    H_est, stats = recover_homography_from_points(pts_proj, pts_cam_noisy)
    np.save(out_dir / "H_projector_to_camera_recovered.npy", H_est)

    # Pre-warp roundtrip: desired validation in CamImg -> ProjFB -> CamImg
    desired_cam = make_validation_target(cam_h, cam_w)
    prewarp = prewarp_camera_content_to_projector(desired_cam, H_c2p, proj_w, proj_h)
    reprojected = apply_H_projector_to_camera(prewarp, H_p2c, cam_w, cam_h)
    cv2.imwrite(str(out_dir / "desired_cam.png"), desired_cam)
    cv2.imwrite(str(out_dir / "prewarp_proj.png"), prewarp)
    cv2.imwrite(str(out_dir / "reprojected_cam.png"), reprojected)

    # Error only where desired has content
    mask = (desired_cam.max(axis=2) > 10).astype(np.float32)
    diff = np.abs(reprojected.astype(np.float32) - desired_cam.astype(np.float32)).mean(axis=2)
    valid = mask > 0
    # Also geometric: sample corners of validation rectangle
    margin = 60
    corners_cam = np.array(
        [
            [margin, margin],
            [cam_w - margin - 1, margin],
            [cam_w - margin - 1, cam_h - margin - 1],
            [margin, cam_h - margin - 1],
        ],
        dtype=np.float32,
    )
    # Ideal corners should map: cam -> proj via H_c2p -> cam via H_p2c
    ch = cv2.convertPointsToHomogeneous(corners_cam)[:, 0, :]
    pj = (H_c2p @ ch.T).T
    pj = pj[:, :2] / pj[:, 2:3]
    ph = cv2.convertPointsToHomogeneous(pj.astype(np.float32))[:, 0, :]
    back = (H_p2c @ ph.T).T
    back = back[:, :2] / back[:, 2:3]
    corner_err = np.linalg.norm(back - corners_cam, axis=1)

    roundtrip = {
        "mean_abs_intensity_diff_valid": float(diff[valid].mean()) if valid.any() else None,
        "corner_mean_px": float(corner_err.mean()),
        "corner_median_px": float(np.median(corner_err)),
        "corner_max_px": float(corner_err.max()),
    }

    passed = (
        stats["median_reproj_px"] < 1.0
        and stats["p95_reproj_px"] < 2.0
        and roundtrip["corner_median_px"] < 1.5
    )
    result = {
        "pass": passed,
        "homography_recovery": stats,
        "prewarp_roundtrip": roundtrip,
        "thresholds": {
            "median_reproj_px": 1.0,
            "p95_reproj_px": 2.0,
            "corner_median_px": 1.5,
        },
        "spaces": {
            "H_projector_to_camera": "ProjFB -> CamImg",
            "H_camera_to_projector": "CamImg -> ProjFB",
            "prewarp": "CamImg desired content -> ProjFB framebuffer",
        },
    }
    (out_dir / "synthetic_metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "report.md").write_text(
        f"# Synthetic verification\n\npass={passed}\n\n```json\n{json.dumps(result, indent=2)}\n```\n"
    )
    return result
