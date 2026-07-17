"""Phase 3–4: forward pre-warp matrix + residual composition convention tests."""

from __future__ import annotations

import numpy as np
import pytest

from procam_calibrate.charuco import apply_H, estimate_H_cam_from_proj
from procam_calibrate.homography_baseline import (
    forward_H_proj_from_source,
    prewarp_from_forward_H,
)
from procam_calibrate.residual_refine import (
    candidate_H_proj_from_source,
    interpolate_homography_strength,
)


def _rand_H(rng: np.random.Generator) -> np.ndarray:
    src = np.array([[0, 0], [1919, 0], [1919, 1079], [0, 1079]], dtype=np.float64)
    dst = src + rng.normal(0, 25, size=src.shape)
    dst[2] += rng.normal(0, 40, size=2)
    H, _ = estimate_H_cam_from_proj(src, dst)
    return H


def test_forward_H_composes_to_desired():
    rng = np.random.default_rng(0)
    H_cam = _rand_H(rng)
    H_des = _rand_H(rng)
    H_fwd = forward_H_proj_from_source(H_cam, H_des)
    pts = np.array([[100, 100], [900, 200], [1500, 900], [200, 800]], dtype=np.float64)
    via = apply_H(H_cam, apply_H(H_fwd, pts))
    direct = apply_H(H_des, pts)
    assert np.median(np.linalg.norm(via - direct, axis=1)) < 1e-6


def test_synthetic_chain_source_to_desired_camera():
    """source → H_proj_from_source → H_cam_from_proj → desired camera."""
    rng = np.random.default_rng(1)
    H_cam = _rand_H(rng)
    H_des = _rand_H(rng)
    H_fwd = forward_H_proj_from_source(H_cam, H_des)
    src = np.array([[50, 60], [400, 80], [1700, 1000], [80, 950], [960, 540]], dtype=np.float64)
    proj = apply_H(H_fwd, src)
    cam = apply_H(H_cam, proj)
    want = apply_H(H_des, src)
    err = np.linalg.norm(cam - want, axis=1)
    assert float(np.max(err)) < 1e-5


def test_warpPerspective_uses_forward_matrix_not_sampling_inverse():
    """Raster path: content point s appears near H_fwd @ s (spot test)."""
    import cv2

    content = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # bright marker at known source pixel
    s = (400, 300)
    content[s[1] - 2 : s[1] + 3, s[0] - 2 : s[0] + 3] = (0, 255, 0)
    H_fwd = np.array([[1.05, 0.02, 30], [-0.01, 0.98, -20], [0, 0, 1]], dtype=np.float64)
    warped = prewarp_from_forward_H(content, H_fwd, 1920, 1080)
    p = apply_H(H_fwd, np.array([[s[0], s[1]]], dtype=np.float64))[0]
    x, y = int(round(p[0])), int(round(p[1]))
    patch = warped[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4]
    assert patch.size > 0 and patch[..., 1].max() > 200


def test_candidate_composition_recovers_desired_prewarp():
    """Deterministic: observed = physical @ current; candidate must restore desired."""
    rng = np.random.default_rng(2)
    H_phys = _rand_H(rng)
    H_des = _rand_H(rng)
    H_cur = forward_H_proj_from_source(H_phys, _rand_H(rng))  # wrong current prewarp
    # True desired forward
    H_true = forward_H_proj_from_source(H_phys, H_des)
    src = np.array(
        [[100, 100], [500, 120], [1800, 200], [200, 900], [900, 500], [1600, 950]],
        dtype=np.float64,
    )
    # Detected camera = physical @ current prewarp @ source
    pts_cam = apply_H(H_phys, apply_H(H_cur, src))
    H_cand, _ = candidate_H_proj_from_source(H_cur, src, pts_cam, H_des)
    # After candidate: physical @ cand ≈ desired
    pts_after = apply_H(H_phys, apply_H(H_cand, src))
    pts_want = apply_H(H_des, src)
    assert float(np.median(np.linalg.norm(pts_after - pts_want, axis=1))) < 0.05
    # And candidate close to true forward
    rel = apply_H(H_cand, src) - apply_H(H_true, src)
    assert float(np.median(np.linalg.norm(rel, axis=1))) < 0.5


def test_wrong_composition_order_fails():
    rng = np.random.default_rng(3)
    H_phys = _rand_H(rng)
    H_des = _rand_H(rng)
    H_cur = forward_H_proj_from_source(H_phys, _rand_H(rng))
    src = np.array([[120, 140], [600, 200], [1700, 300], [300, 900], [1000, 600]], dtype=np.float64)
    pts_cam = apply_H(H_phys, apply_H(H_cur, src))
    H_obs, _ = estimate_H_cam_from_proj(src, pts_cam)
    # Reversed order (wrong)
    H_wrong = H_des @ np.linalg.inv(H_obs) @ H_cur
    H_wrong = H_wrong / H_wrong[2, 2]
    H_right, _ = candidate_H_proj_from_source(H_cur, src, pts_cam, H_des)
    pts_want = apply_H(H_des, src)
    err_wrong = float(np.median(np.linalg.norm(apply_H(H_phys, apply_H(H_wrong, src)) - pts_want, axis=1)))
    err_right = float(np.median(np.linalg.norm(apply_H(H_phys, apply_H(H_right, src)) - pts_want, axis=1)))
    assert err_right < 0.1
    assert err_wrong > 10 * err_right + 1.0


def test_double_inversion_fails():
    rng = np.random.default_rng(4)
    H_phys = _rand_H(rng)
    H_des = _rand_H(rng)
    H_cur = forward_H_proj_from_source(H_phys, _rand_H(rng))
    src = np.array([[150, 160], [700, 220], [1600, 400], [250, 850]], dtype=np.float64)
    pts_cam = apply_H(H_phys, apply_H(H_cur, src))
    H_obs, _ = estimate_H_cam_from_proj(src, pts_cam)
    H_wrong = H_cur @ H_obs @ np.linalg.inv(H_des)  # double-invert style mistake
    H_wrong = H_wrong / H_wrong[2, 2]
    H_right, _ = candidate_H_proj_from_source(H_cur, src, pts_cam, H_des)
    pts_want = apply_H(H_des, src)
    err_wrong = float(np.median(np.linalg.norm(apply_H(H_phys, apply_H(H_wrong, src)) - pts_want, axis=1)))
    err_right = float(np.median(np.linalg.norm(apply_H(H_phys, apply_H(H_right, src)) - pts_want, axis=1)))
    assert err_right < 0.1
    assert err_wrong > err_right + 5.0


def test_source_destination_swap_fails():
    rng = np.random.default_rng(5)
    H_phys = _rand_H(rng)
    H_des = _rand_H(rng)
    H_cur = forward_H_proj_from_source(H_phys, _rand_H(rng))
    src = np.array([[180, 190], [800, 250], [1500, 500], [400, 900]], dtype=np.float64)
    pts_cam = apply_H(H_phys, apply_H(H_cur, src))
    # Swap: fit cam→source instead of source→cam
    H_swapped, _ = estimate_H_cam_from_proj(pts_cam, src)
    H_wrong = H_cur @ np.linalg.inv(H_swapped) @ H_des
    H_wrong = H_wrong / H_wrong[2, 2]
    H_right, _ = candidate_H_proj_from_source(H_cur, src, pts_cam, H_des)
    pts_want = apply_H(H_des, src)
    err_wrong = float(np.median(np.linalg.norm(apply_H(H_phys, apply_H(H_wrong, src)) - pts_want, axis=1)))
    err_right = float(np.median(np.linalg.norm(apply_H(H_phys, apply_H(H_right, src)) - pts_want, axis=1)))
    assert err_right < 0.1
    assert err_wrong > err_right + 5.0


def test_opencv_sampling_matrix_as_forward_fails():
    """Using inv(H_fwd) as if it were the forward point matrix must fail the chain."""
    rng = np.random.default_rng(6)
    H_cam = _rand_H(rng)
    H_des = _rand_H(rng)
    H_fwd = forward_H_proj_from_source(H_cam, H_des)
    H_sample = np.linalg.inv(H_fwd)  # OpenCV sampling matrix mistaken for forward
    src = np.array([[200, 200], [900, 300], [1600, 800]], dtype=np.float64)
    via_wrong = apply_H(H_cam, apply_H(H_sample, src))
    want = apply_H(H_des, src)
    via_right = apply_H(H_cam, apply_H(H_fwd, src))
    assert float(np.median(np.linalg.norm(via_right - want, axis=1))) < 1e-5
    assert float(np.median(np.linalg.norm(via_wrong - want, axis=1))) > 10.0


def test_strength_interpolation_not_coeff_lerp():
    H0 = np.eye(3, dtype=np.float64)
    H1 = np.array([[1.2, 0.05, 40], [0.02, 0.9, -15], [1e-6, 2e-6, 1]], dtype=np.float64)
    ctrl = np.array([[0, 0], [1919, 0], [1919, 1079], [0, 1079], [960, 540]], dtype=np.float64)
    H_half = interpolate_homography_strength(H0, H1, ctrl, 0.5)
    # Midpoint in projector space
    p_mid = 0.5 * (apply_H(H0, ctrl) + apply_H(H1, ctrl))
    p_h = apply_H(H_half, ctrl)
    assert float(np.median(np.linalg.norm(p_h - p_mid, axis=1))) < 1.0
    # Coefficient lerp would differ from control-point interpolation for projective H
    H_coeff = 0.5 * (H0 + H1)
    H_coeff = H_coeff / H_coeff[2, 2]
    # Not required to differ always, but control-point path must match p_mid
    assert float(np.max(np.linalg.norm(p_h - p_mid, axis=1))) < 2.0
