"""Explicit projector–camera–source coordinate conventions.

All production homographies must name their mapping direction:

  H_<dst>_from_<src>   means   p_dst ~ H @ p_src   (homogeneous column vectors)

Spaces:
  source   — content layout matching ProjFB resolution (pre-warp input)
  proj     — projector framebuffer pixels (OpenCV top-left, +u right, +v down)
  cam      — camera image pixels as captured (same axis convention)
  desired  — predetermined camera-space target (not derived from corrected capture)
"""

from __future__ import annotations

import numpy as np

from .charuco import apply_H
from .homography_baseline import forward_H_proj_from_source, prewarp_from_forward_H


CONVENTION_DOC = {
    "pixel_origin": "top-left",
    "axis": "+u right, +v down",
    "homography_form": "p_dst ~ H_dst_from_src @ p_src",
    "opencv_warpPerspective": (
        "dst(p) = src(H^{-1} p); pass H_proj_from_source so content point s "
        "appears at projector pixel p = H_proj_from_source @ s"
    ),
}


def verify_forward_composition(
    H_cam_from_proj: np.ndarray,
    H_desired_cam_from_source: np.ndarray,
    pts_source: np.ndarray,
    tol_px: float = 1e-3,
) -> dict:
    """Check H_proj_from_source then H_cam_from_proj ≈ H_desired."""
    H_fwd = forward_H_proj_from_source(H_cam_from_proj, H_desired_cam_from_source)
    pred_proj = apply_H(H_fwd, pts_source)
    pred_cam = apply_H(H_cam_from_proj, pred_proj)
    des_cam = apply_H(H_desired_cam_from_source, pts_source)
    err = np.linalg.norm(pred_cam - des_cam, axis=1)
    return {
        "ok": bool(float(err.max()) <= tol_px),
        "max_err_px": float(err.max()),
        "median_err_px": float(np.median(err)),
        "H_proj_from_source": H_fwd,
    }


def detect_mirrored_or_inverted(
    H_proj_from_source: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> dict:
    """Flag mirrored / transposed / inverted corner mappings."""
    corners = np.array(
        [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
        dtype=np.float64,
    )
    mapped = apply_H(H_proj_from_source, corners)
    # Identity-like should keep order; detect x flip if left corners map right of right corners
    left_x = 0.5 * (mapped[0, 0] + mapped[3, 0])
    right_x = 0.5 * (mapped[1, 0] + mapped[2, 0])
    top_y = 0.5 * (mapped[0, 1] + mapped[1, 1])
    bot_y = 0.5 * (mapped[2, 1] + mapped[3, 1])
    mirrored_x = bool(left_x > right_x)
    mirrored_y = bool(top_y > bot_y)
    # det of linear part
    A = H_proj_from_source[:2, :2]
    det = float(np.linalg.det(A))
    return {
        "mirrored_x": mirrored_x,
        "mirrored_y": mirrored_y,
        "det_linear": det,
        "inverted_orientation": bool(det < 0),
        "ok": not mirrored_x and not mirrored_y,
    }


def wrong_direction_error(
    H_cam_from_proj: np.ndarray,
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
) -> dict:
    """Compare correct H vs using inv(H) as if directions were swapped."""
    err_ok = np.linalg.norm(apply_H(H_cam_from_proj, pts_proj) - pts_cam, axis=1)
    H_wrong = np.linalg.inv(H_cam_from_proj)
    H_wrong = H_wrong / H_wrong[2, 2]
    err_bad = np.linalg.norm(apply_H(H_wrong, pts_proj) - pts_cam, axis=1)
    return {
        "correct_median": float(np.median(err_ok)),
        "wrong_direction_median": float(np.median(err_bad)),
        "wrong_is_worse": bool(np.median(err_bad) > np.median(err_ok) * 5 + 1.0),
    }
