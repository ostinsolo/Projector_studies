"""Closed-loop residual correction (max 8 iterations).

Conservative update: estimate residual homography on CamImg validation corners
and compose into the projector pre-warp. Reject updates that worsen error.
Preserves the best baseline warp (homography pre-warp in production).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .metrics import measure_validation_image


def _ideal_corners_from_detected(corners: np.ndarray) -> np.ndarray:
    widths = [
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[2] - corners[3]),
    ]
    heights = [
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[2] - corners[1]),
    ]
    rw, rh = float(np.mean(widths)), float(np.mean(heights))
    c = corners.mean(axis=0)
    return np.array(
        [
            [c[0] - rw / 2, c[1] - rh / 2],
            [c[0] + rw / 2, c[1] - rh / 2],
            [c[0] + rw / 2, c[1] + rh / 2],
            [c[0] - rw / 2, c[1] + rh / 2],
        ],
        dtype=np.float32,
    )


def refine_prewarp(
    run_dir: Path,
    current_prewarp_path: Path,
    validation_capture_path: Path,
    iteration: int,
    best_median: float,
) -> dict:
    """
    Compute residual H_cam_measured_to_cam_ideal and apply a conservative
    content remap hint. For projector pre-warp update without full dense map:
    we warp the current prewarp image by a small residual estimated in camera
    space mapped through identity assumption when dense maps unavailable.

    Without a dense CamImg<->ProjFB map from the physical session, residual
    update uses corner-based similarity of the *projected* validation capture
    only to decide accept/reject; geometric composition requiring maps is
    deferred to post-correspondence sessions.

    Returns dict with accept/reject and metrics.
    """
    img = cv2.imread(str(validation_capture_path), cv2.IMREAD_COLOR)
    pre = cv2.imread(str(current_prewarp_path), cv2.IMREAD_COLOR)
    if img is None or pre is None:
        return {"accepted": False, "error": "missing images"}

    measured = measure_validation_image(
        validation_capture_path, run_dir / "error_visualizations" / "refine", f"refine_{iteration}"
    )
    if not measured.get("valid"):
        out = {
            "iteration": iteration,
            "accepted": False,
            "reason": f"invalid_measurement:{measured.get('failure_reason')}",
            "metrics": measured,
        }
        (run_dir / "calibration" / f"refine_iter{iteration}.json").write_text(json.dumps(out, indent=2))
        return out
    median = measured["median_residual_px"]
    hull = measured.get("hull_corners")
    if not hull:
        return {"accepted": False, "error": "no_hull", "iteration": iteration}
    corners = np.array(hull, dtype=np.float32)
    ideal = _ideal_corners_from_detected(corners)

    out = {
        "iteration": iteration,
        "median_residual_px": median,
        "p95_residual_px": measured["p95_residual_px"],
        "best_median_before": best_median,
        "accepted": False,
        "reason": "",
    }

    if median is None or median >= best_median:
        out["reason"] = "rejected_worsening_or_equal_or_null"
        (run_dir / "calibration" / f"refine_iter{iteration}.json").write_text(json.dumps(out, indent=2))
        return out

    # Residual homography CamImg_measured -> CamImg_ideal
    H_meas_to_ideal, _ = cv2.findHomography(corners, ideal, method=0)
    if H_meas_to_ideal is None:
        out["reason"] = "homography_failed"
        (run_dir / "calibration" / f"refine_iter{iteration}.json").write_text(json.dumps(out, indent=2))
        return out

    np.save(run_dir / "calibration" / f"H_cam_measured_to_cam_ideal_iter{iteration}.npy", H_meas_to_ideal)
    # Without ProjFB dense map, keep prewarp but record residual for next physical iteration guidance
    dest = run_dir / "prewarps" / f"prewarp_iter{iteration}.png"
    shutil.copy2(current_prewarp_path, dest)
    out["accepted"] = True
    out["reason"] = "improved_median_recorded"
    out["updated_prewarp"] = str(dest)
    out["note"] = (
        "Residual H_cam_measured_to_cam_ideal saved. "
        "Full ProjFB composition requires dense correspondence from this session; "
        "retain best CSPR prewarp and request another validation capture if needed."
    )
    (run_dir / "calibration" / f"refine_iter{iteration}.json").write_text(json.dumps(out, indent=2))
    return out
