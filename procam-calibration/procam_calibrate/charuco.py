"""Shared ChArUco board definition, detection, and correspondence metrics.

Single implementation used by homography calibration, validation, refinement,
and acceptance reporting. Do not duplicate detectors elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# --- Canonical board (must match the successful flat-wall baseline) ---
CHARUCO_SQUARES = (10, 7)  # (columns, rows) of chessboard squares
CHARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
CHARUCO_DICT_NAME = "DICT_4X4_50"
SQUARE_LENGTH = 1.0
MARKER_LENGTH = 0.7
GENERATE_MARGIN = 20
GENERATE_BORDER_BITS = 1
RANSAC_REPROJ_THRESHOLD = 3.0
MIN_CORNERS_CALIB = 12
MIN_CORNERS_VALIDATE = 8
# Expected inner corners for (10,7) squares: 9 x 6 = 54
EXPECTED_CHARUCO_CORNERS = (CHARUCO_SQUARES[0] - 1) * (CHARUCO_SQUARES[1] - 1)


@dataclass
class BoardSpec:
    squares: tuple[int, int] = CHARUCO_SQUARES
    dictionary_id: int = CHARUCO_DICT_ID
    dictionary_name: str = CHARUCO_DICT_NAME
    square_length: float = SQUARE_LENGTH
    marker_length: float = MARKER_LENGTH
    generate_margin: int = GENERATE_MARGIN
    generate_border_bits: int = GENERATE_BORDER_BITS
    expected_charuco_corners: int = EXPECTED_CHARUCO_CORNERS
    ransac_reproj_threshold: float = RANSAC_REPROJ_THRESHOLD


DEFAULT_BOARD_SPEC = BoardSpec()


def make_board(spec: BoardSpec = DEFAULT_BOARD_SPEC) -> tuple[Any, Any, BoardSpec]:
    dictionary = cv2.aruco.getPredefinedDictionary(spec.dictionary_id)
    board = cv2.aruco.CharucoBoard(
        spec.squares,
        squareLength=spec.square_length,
        markerLength=spec.marker_length,
        dictionary=dictionary,
    )
    return dictionary, board, spec


def generate_charuco_image(
    proj_w: int,
    proj_h: int,
    spec: BoardSpec = DEFAULT_BOARD_SPEC,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> tuple[np.ndarray, dict[int, np.ndarray], Any, BoardSpec, dict]:
    """Generate BGR ChArUco pattern in ProjFB.

    roi: optional (x0,y0,x1,y1) inclusive pixel box — board is generated into
    that sub-rectangle; outside is black. Used when inverse warp clips edges.
    """
    _, board, spec = make_board(spec)
    if roi is None:
        x0, y0, x1, y1 = 0, 0, proj_w - 1, proj_h - 1
        gen_w, gen_h = proj_w, proj_h
    else:
        x0, y0, x1, y1 = roi
        gen_w, gen_h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    gray = board.generateImage(
        (gen_w, gen_h), marginSize=spec.generate_margin, borderBits=spec.generate_border_bits
    )
    canvas = np.zeros((proj_h, proj_w), dtype=np.uint8)
    canvas[y0 : y0 + gen_h, x0 : x0 + gen_w] = gray
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    ids_to_xy, det = detect_charuco(bgr, board=board, spec=spec)
    # Shift already in full-frame coords because we detected on canvas
    meta = {
        "proj_resolution": [proj_w, proj_h],
        "roi": [x0, y0, x1, y1],
        "board": asdict(spec),
        "self_detect": det,
        "n_proj_corners": len(ids_to_xy),
    }
    if len(ids_to_xy) < MIN_CORNERS_CALIB:
        raise RuntimeError(
            f"Self-detection on ChArUco pattern found only {len(ids_to_xy)} corners "
            f"(expected up to {spec.expected_charuco_corners})"
        )
    return bgr, ids_to_xy, board, spec, meta


def detect_charuco(
    img_bgr: np.ndarray,
    board: Any | None = None,
    spec: BoardSpec = DEFAULT_BOARD_SPEC,
) -> tuple[dict[int, np.ndarray], dict]:
    """Detect ChArUco corners. Returns id→(u,v) in image pixels and rich metadata."""
    if board is None:
        _, board, spec = make_board(spec)
    if img_bgr.ndim == 2:
        gray = img_bgr
        h, w = gray.shape[:2]
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

    # Preprocessing: grayscale only (matches successful baseline). No adaptive EQ.
    preprocessing = {
        "color_conversion": "BGR2GRAY" if img_bgr.ndim == 3 else "none",
        "adaptive_histogram": False,
        "blur": None,
        "resize": None,
    }

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    n_markers = 0 if marker_ids is None else int(len(marker_ids))
    n_corners = 0 if charuco_ids is None else int(len(charuco_ids))
    out: dict[int, np.ndarray] = {}
    rejection_reasons: list[str] = []
    if marker_ids is None or n_markers == 0:
        rejection_reasons.append("no_aruco_markers_detected")
    if charuco_ids is None or n_corners == 0:
        rejection_reasons.append("no_charuco_corners_interpolated")
    else:
        for pt, i in zip(charuco_corners.reshape(-1, 2), charuco_ids.reshape(-1)):
            out[int(i)] = np.array(pt, dtype=np.float64)

    missing_ids = sorted(set(range(spec.expected_charuco_corners)) - set(out.keys()))
    meta = {
        "valid": len(out) >= MIN_CORNERS_VALIDATE,
        "image_resolution": [w, h],
        "board": asdict(spec),
        "dictionary": spec.dictionary_name,
        "preprocessing": preprocessing,
        "expected_charuco_corners": spec.expected_charuco_corners,
        "n_markers_detected": n_markers,
        "n_charuco_corners_interpolated": n_corners,
        "n_corners_returned": len(out),
        "corner_ids": sorted(out.keys()),
        "missing_corner_ids": missing_ids,
        "n_missing_corners": len(missing_ids),
        "rejection_reasons": rejection_reasons,
        "min_corners_required": MIN_CORNERS_VALIDATE,
    }
    return out, meta


def match_correspondences(
    proj_ids: dict[int, np.ndarray],
    cam_ids: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    common = sorted(set(proj_ids) & set(cam_ids))
    only_proj = sorted(set(proj_ids) - set(cam_ids))
    only_cam = sorted(set(cam_ids) - set(proj_ids))
    meta = {
        "n_proj_ids": len(proj_ids),
        "n_cam_ids": len(cam_ids),
        "n_matched": len(common),
        "matched_ids": common,
        "unmatched_proj_ids": only_proj,
        "unmatched_cam_ids": only_cam,
        "rejection_reasons": [],
    }
    if not common:
        meta["rejection_reasons"].append("no_shared_corner_ids")
        return np.zeros((0, 2)), np.zeros((0, 2)), [], meta
    pts_p = np.array([proj_ids[i] for i in common], dtype=np.float64)
    pts_c = np.array([cam_ids[i] for i in common], dtype=np.float64)
    return pts_p, pts_c, common, meta


def apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ph = cv2.convertPointsToHomogeneous(pts.astype(np.float64))[:, 0, :]
    out = (H @ ph.T).T
    return out[:, :2] / np.clip(out[:, 2:3], 1e-12, None)


def estimate_H_cam_from_proj(
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
    ransac_threshold: float = RANSAC_REPROJ_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """H_cam_from_proj: ProjFB homogeneous → CamImg homogeneous."""
    if len(pts_proj) < 4:
        raise RuntimeError(f"Need >=4 correspondences, got {len(pts_proj)}")
    H, mask = cv2.findHomography(
        pts_proj, pts_cam, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold
    )
    if H is None:
        raise RuntimeError("findHomography failed")
    mask_flat = mask.ravel().astype(bool) if mask is not None else np.ones(len(pts_proj), dtype=bool)
    inliers = int(mask_flat.sum())
    cam_est = apply_H(H, pts_proj)
    err = np.linalg.norm(cam_est - pts_cam, axis=1)
    err_inl = err[mask_flat]
    outlier_ids_idx = np.where(~mask_flat)[0].tolist()
    stats = {
        "n_correspondences": int(len(pts_proj)),
        "n_inliers": inliers,
        "n_outliers": int((~mask_flat).sum()),
        "outlier_indices": outlier_ids_idx,
        "ransac_reproj_threshold": ransac_threshold,
        "homography_condition_number": float(np.linalg.cond(H)),
        "mean_reproj_px": float(err_inl.mean()) if len(err_inl) else None,
        "median_reproj_px": float(np.median(err_inl)) if len(err_inl) else None,
        "p95_reproj_px": float(np.percentile(err_inl, 95)) if len(err_inl) else None,
        "max_reproj_px": float(err_inl.max()) if len(err_inl) else None,
        "all_point_mean_reproj_px": float(err.mean()),
        "inlier_mask": mask_flat.astype(int).tolist(),
    }
    return H.astype(np.float64), stats


def line_deviation(points: np.ndarray) -> float:
    """Max orthogonal distance from best-fit line (px)."""
    if len(points) < 2:
        return float("nan")
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
    d = np.abs((points[:, 0] - x0) * vy - (points[:, 1] - y0) * vx)
    return float(np.max(d))


def _axis_angle_deviations(pts_proj: np.ndarray, pts_cam: np.ndarray) -> dict:
    """Row/col angle deviation from image horizontal/vertical (keystone signal)."""
    h_angs: list[float] = []
    v_angs: list[float] = []
    # Group by rounded projector coords; use <=0.5 so .5 values map to their round bin.
    for yv in np.unique(np.round(pts_proj[:, 1], 0)):
        idx = np.where(np.abs(pts_proj[:, 1] - float(yv)) <= 0.5 + 1e-9)[0]
        if len(idx) < 3:
            continue
        vx, vy, _x0, _y0 = cv2.fitLine(pts_cam[idx].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        ang = float(np.degrees(np.arctan2(float(vy[0]), float(vx[0]))))
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        h_angs.append(abs(ang))
    for xv in np.unique(np.round(pts_proj[:, 0], 0)):
        idx = np.where(np.abs(pts_proj[:, 0] - float(xv)) <= 0.5 + 1e-9)[0]
        if len(idx) < 3:
            continue
        vx, vy, _x0, _y0 = cv2.fitLine(pts_cam[idx].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        ang = float(np.degrees(np.arctan2(float(vy[0]), float(vx[0]))))
        v_angs.append(abs(abs(ang) - 90.0))
    return {
        "horizontal_axis_deviation_deg_median": float(np.median(h_angs)) if h_angs else None,
        "horizontal_axis_deviation_deg_max": float(np.max(h_angs)) if h_angs else None,
        "vertical_axis_deviation_deg_median": float(np.median(v_angs)) if v_angs else None,
        "vertical_axis_deviation_deg_max": float(np.max(v_angs)) if v_angs else None,
        "n_horizontal_rows_measured": len(h_angs),
        "n_vertical_cols_measured": len(v_angs),
    }


def _affine_reproj_from_proj(pts_proj: np.ndarray, pts_cam: np.ndarray) -> dict:
    """Affine map from normalized known ProjFB coords → CamImg; residual is primary geometry."""
    pmin = pts_proj.min(axis=0)
    pmax = pts_proj.max(axis=0)
    span = np.clip(pmax - pmin, 1e-12, None)
    ideal = (pts_proj - pmin) / span
    A = np.hstack([ideal, np.ones((len(ideal), 1), dtype=np.float64)])
    M, _, _, _ = np.linalg.lstsq(A, pts_cam.astype(np.float64), rcond=None)
    pred = A @ M
    err = np.linalg.norm(pred - pts_cam, axis=1)
    return {
        "affine_median_reproj_px": float(np.median(err)),
        "affine_p95_reproj_px": float(np.percentile(err, 95)),
        "affine_max_reproj_px": float(err.max()),
        "affine_mean_reproj_px": float(err.mean()),
    }


def measure_independent_residuals(
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
    H_cam_from_proj: np.ndarray | None,
    matched_ids: list[int],
    tag: str,
    det_meta: Optional[dict] = None,
    match_meta: Optional[dict] = None,
    fit_stats: Optional[dict] = None,
    refit_homography: bool = True,
) -> dict:
    """Primary acceptance metrics from known ProjFB coords vs detected CamImg coords.

    Homography residual is from a per-capture refit by default (matches the successful
    baseline's reproj_before / reproj_after_refit). Fixed calibration H may be supplied
    for diagnostics but is not the acceptance residual after pre-warp.
    """
    if len(pts_proj) < MIN_CORNERS_VALIDATE:
        return {
            "valid": False,
            "pass": False,
            "tag": tag,
            "failure_reason": f"insufficient_matched_corners:{len(pts_proj)}",
            "n_valid_measurements": int(len(pts_proj)),
            "median_reproj_px": None,
            "p95_reproj_px": None,
            "max_reproj_px": None,
            "mean_reproj_px": None,
            "detection": det_meta,
            "matching": match_meta,
        }

    H_use = H_cam_from_proj
    local_fit = fit_stats
    if refit_homography or H_use is None:
        H_use, local_fit = estimate_H_cam_from_proj(pts_proj, pts_cam)

    cam_est = apply_H(H_use, pts_proj)
    disp = cam_est - pts_cam
    err = np.linalg.norm(disp, axis=1)

    # Collinearity residual within board rows/cols (lines stay lines under perspective)
    h_devs, v_devs = [], []
    for yv in np.unique(np.round(pts_proj[:, 1], 0)):
        idx = np.where(np.abs(pts_proj[:, 1] - float(yv)) <= 0.5 + 1e-9)[0]
        if len(idx) >= 3:
            h_devs.append(line_deviation(pts_cam[idx]))
    for xv in np.unique(np.round(pts_proj[:, 0], 0)):
        idx = np.where(np.abs(pts_proj[:, 0] - float(xv)) <= 0.5 + 1e-9)[0]
        if len(idx) >= 3:
            v_devs.append(line_deviation(pts_cam[idx]))

    axis = _axis_angle_deviations(pts_proj, pts_cam)
    affine = _affine_reproj_from_proj(pts_proj, pts_cam)
    secondary = secondary_aa_hull_metric(pts_cam)

    # Primary residual for gates: homography refit (known proj ↔ detected cam)
    result = {
        "valid": True,
        "pass": True,
        "tag": tag,
        "metric_family": "independent_proj_cam_reprojection_refit",
        "n_valid_measurements": int(len(pts_proj)),
        "matched_ids": matched_ids,
        "mean_reproj_px": float(err.mean()),
        "median_reproj_px": float(np.median(err)),
        "p95_reproj_px": float(np.percentile(err, 95)),
        "max_reproj_px": float(err.max()),
        "corner_displacement_mean_px": float(err.mean()),
        "corner_displacement_median_px": float(np.median(err)),
        "horizontal_line_deviation_px": float(np.median(h_devs)) if h_devs else None,
        "vertical_line_deviation_px": float(np.median(v_devs)) if v_devs else None,
        "horizontal_line_deviation_max_px": float(np.max(h_devs)) if h_devs else None,
        "vertical_line_deviation_max_px": float(np.max(v_devs)) if v_devs else None,
        # Axis alignment (degrees from image axes) — strong keystone signal
        **axis,
        # Affine residual from normalized known projector coords
        **affine,
        "mean_residual_px": float(err.mean()),
        "median_residual_px": float(np.median(err)),
        "p95_residual_px": float(np.percentile(err, 95)),
        "max_residual_px": float(err.max()),
        "detection": det_meta,
        "matching": match_meta,
        "homography_fit": local_fit,
        "secondary_aa_hull_metric": secondary,
        "secondary_metric_note": (
            "AA-hull / contour-style metric is secondary visualization only; "
            "not used for acceptance."
        ),
    }
    if H_cam_from_proj is not None and refit_homography:
        err_fixed = np.linalg.norm(apply_H(H_cam_from_proj, pts_proj) - pts_cam, axis=1)
        result["fixed_calib_H_median_reproj_px"] = float(np.median(err_fixed))
        result["fixed_calib_H_note"] = (
            "Diagnostic only: residual vs calibration H. After pre-warp this is "
            "expected to be large; acceptance uses per-capture refit + axis/affine metrics."
        )
    return result


def secondary_aa_hull_metric(pts_cam: np.ndarray) -> dict:
    if len(pts_cam) < 4:
        return {"valid": False, "reason": "need>=4"}
    rect = cv2.minAreaRect(pts_cam.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float64)
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    ordered = np.stack(
        [box[np.argmin(s)], box[np.argmin(diff)], box[np.argmax(s)], box[np.argmax(diff)]],
        axis=0,
    )
    widths = [np.linalg.norm(ordered[1] - ordered[0]), np.linalg.norm(ordered[2] - ordered[3])]
    heights = [np.linalg.norm(ordered[3] - ordered[0]), np.linalg.norm(ordered[2] - ordered[1])]
    rw, rh = float(np.mean(widths)), float(np.mean(heights))
    c = ordered.mean(axis=0)
    ideal = np.array(
        [
            [c[0] - rw / 2, c[1] - rh / 2],
            [c[0] + rw / 2, c[1] - rh / 2],
            [c[0] + rw / 2, c[1] + rh / 2],
            [c[0] - rw / 2, c[1] + rh / 2],
        ],
        dtype=np.float64,
    )
    corner_err = np.linalg.norm(ordered - ideal, axis=1)
    return {
        "valid": True,
        "label": "secondary_aa_hull",
        "aa_corner_median_err_px": float(np.median(corner_err)),
        "aa_corner_mean_err_px": float(corner_err.mean()),
        "aspect_ratio": rw / rh if rh else None,
        "ordered_corners": ordered.tolist(),
    }


def define_desired_target(
    H_cam_from_proj: np.ndarray,
    proj_w: int,
    proj_h: int,
    proj_ids: dict[int, np.ndarray],
) -> dict:
    """Define independent desired CamImg geometry for production correction metrics.

    Desired target = axis-aligned CamImg rectangle that is the bounding box of
    H_cam_from_proj applied to ProjFB corners. Built BEFORE analysing corrected capture.
    Source: ProjFB. Destination: CamImg.
    """
    corners_p = np.array(
        [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
        dtype=np.float32,
    )
    cam = apply_H(H_cam_from_proj, corners_p.astype(np.float64)).astype(np.float32)
    x, y, w, h = cv2.boundingRect(cam.reshape(-1, 1, 2))
    w, h = max(int(w), 2), max(int(h), 2)
    desired_corners = np.array(
        [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
        dtype=np.float32,
    )
    H_desired = cv2.getPerspectiveTransform(corners_p, desired_corners).astype(np.float64)
    desired_by_id = {
        int(i): apply_H(H_desired, np.asarray(p, dtype=np.float64).reshape(1, 2))[0]
        for i, p in proj_ids.items()
    }
    aspect = float(w) / float(h)
    return {
        "metric_category": "production_desired_target",
        "source_space": "ProjFB",
        "destination_space": "CamImg",
        "proj_resolution": [proj_w, proj_h],
        "desired_framebuffer_corners_cam": desired_corners.tolist(),
        "desired_aspect_ratio": aspect,
        "H_desired_cam_from_proj": H_desired,
        "desired_cam_by_id": desired_by_id,
        "pixel_centre_convention": "OpenCV pixel coords; top-left origin; +u right +v down",
        "distortion_state": "none_applied",
        "note": "Independent of corrected capture contour; matches pre-warp AA destination.",
    }


def save_desired_target(path: Path, target: dict) -> None:
    serial = {
        k: v
        for k, v in target.items()
        if k not in ("H_desired_cam_from_proj", "desired_cam_by_id")
    }
    serial["H_desired_cam_from_proj"] = np.asarray(target["H_desired_cam_from_proj"]).tolist()
    serial["desired_cam_by_id"] = {
        str(i): np.asarray(p).tolist() for i, p in target["desired_cam_by_id"].items()
    }
    path.write_text(json.dumps(serial, indent=2))


def load_desired_target(path: Path) -> dict:
    raw = json.loads(path.read_text())
    raw["H_desired_cam_from_proj"] = np.array(raw["H_desired_cam_from_proj"], dtype=np.float64)
    raw["desired_cam_by_id"] = {
        int(k): np.array(v, dtype=np.float64) for k, v in raw["desired_cam_by_id"].items()
    }
    return raw


def expected_visible_from_pattern(
    pattern_bgr: np.ndarray,
    full_board_ids: Optional[dict[int, np.ndarray]] = None,
) -> dict:
    """Corners actually present on a displayed (possibly pre-warped) pattern."""
    ids, det = detect_charuco(pattern_bgr)
    all_ids = set(full_board_ids or {})
    visible = sorted(ids.keys())
    excluded = sorted(all_ids - set(visible)) if all_ids else []
    return {
        "expected_visible_ids": visible,
        "expected_visible_corners": len(visible),
        "intentionally_excluded_ids": excluded,
        "intentionally_excluded_reason": (
            "not_present_on_displayed_pattern_after_prewarp_or_roi"
            if excluded
            else None
        ),
        "pattern_self_detect": det,
    }


def detection_coverage(
    cam_ids: dict[int, np.ndarray],
    image_shape: tuple[int, int],
    expected_visible: int,
    intentionally_excluded: Optional[list[int]] = None,
    region_pts: Optional[np.ndarray] = None,
) -> dict:
    """Spatial coverage of detections (Gate 5). image_shape = (h, w).

    Quadrants are relative to the board/detection region (not full image), so a
    board in the upper half of the frame can still occupy all four board quadrants.
    """
    h, w = image_shape[:2]
    pts = np.array(list(cam_ids.values()), dtype=np.float64) if cam_ids else np.zeros((0, 2))
    if region_pts is not None and len(region_pts) >= 2:
        rmin = region_pts.min(axis=0)
        rmax = region_pts.max(axis=0)
    elif len(pts) >= 2:
        rmin = pts.min(axis=0)
        rmax = pts.max(axis=0)
    else:
        rmin = np.array([0.0, 0.0])
        rmax = np.array([float(w), float(h)])
    cx = 0.5 * (rmin[0] + rmax[0])
    cy = 0.5 * (rmin[1] + rmax[1])
    quads = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
    for p in pts:
        left = p[0] < cx
        top = p[1] < cy
        key = ("t" if top else "b") + ("l" if left else "r")
        quads[key] += 1
    hull_area = 0.0
    if len(pts) >= 3:
        hull = cv2.convexHull(pts.astype(np.float32))
        hull_area = float(cv2.contourArea(hull))
    min_border = None
    if len(pts):
        d = np.minimum.reduce([pts[:, 0], pts[:, 1], w - 1 - pts[:, 0], h - 1 - pts[:, 1]])
        min_border = float(d.min())
    matched = len(cam_ids)
    exp = max(int(expected_visible), 1)
    frac = matched / exp
    excluded = list(intentionally_excluded or [])
    n_quads_occupied = sum(1 for v in quads.values() if v > 0)
    coverage_pass = bool(frac >= 0.80)
    # Documented intentional exclusion path: still require enough matched points
    if not coverage_pass and excluded and matched >= MIN_CORNERS_VALIDATE and frac >= 0.80:
        coverage_pass = True
    spatial_pass = bool(n_quads_occupied >= 3)
    return {
        "expected_visible_corners": int(expected_visible),
        "detected_corners": matched,
        "matched_corners": matched,
        "coverage_fraction": float(frac),
        "quadrant_counts": quads,
        "quadrant_basis": "board_region",
        "n_quadrants_occupied": n_quads_occupied,
        "hull_area_fraction": float(hull_area / (w * h)) if w * h else 0.0,
        "min_boundary_distance_px": min_border,
        "intentionally_excluded_ids": excluded,
        "coverage_pass": coverage_pass,
        "spatial_spread_pass": spatial_pass,
        "coverage_gate_pass": bool(coverage_pass and spatial_pass),
    }


def measure_against_desired_target(
    cam_ids: dict[int, np.ndarray],
    desired: dict,
    tag: str,
    det_meta: Optional[dict] = None,
    match_meta: Optional[dict] = None,
    pts_proj: Optional[np.ndarray] = None,
    matched_ids: Optional[list[int]] = None,
    proj_ids: Optional[dict[int, np.ndarray]] = None,
) -> dict:
    """Production correction metrics: detected CamImg vs predetermined desired CamImg."""
    desired_ids: dict[int, np.ndarray] = desired["desired_cam_by_id"]
    common = sorted(set(cam_ids) & set(desired_ids))
    if len(common) < MIN_CORNERS_VALIDATE:
        return {
            "valid": False,
            "pass": False,
            "tag": tag,
            "metric_category": "B_production_correction",
            "failure_reason": f"insufficient_matched_to_desired:{len(common)}",
            "n_valid_measurements": len(common),
            "target_median_err_px": None,
            "target_p95_err_px": None,
            "target_max_err_px": None,
            "detection": det_meta,
            "matching": match_meta,
        }
    pts_c = np.array([cam_ids[i] for i in common], dtype=np.float64)
    pts_d = np.array([desired_ids[i] for i in common], dtype=np.float64)
    err = np.linalg.norm(pts_c - pts_d, axis=1)
    # Rebuild projector coords in the same ID order as pts_c (never trust external ordering)
    if proj_ids is not None:
        pts_proj = np.array([proj_ids[i] for i in common], dtype=np.float64)
    elif pts_proj is not None and matched_ids is not None and list(matched_ids) == list(common):
        pass  # already aligned
    else:
        # Fall back: use desired Cam positions as a rectangular proxy grid for row/col grouping
        pts_proj = pts_d.copy()

    axis = _axis_angle_deviations(pts_proj, pts_c)

    # Orthogonality: mean row vs col directions
    ortho = None
    h_dirs, v_dirs = [], []
    for yv in np.unique(np.round(pts_proj[:, 1], 0)):
        idx = np.where(np.abs(pts_proj[:, 1] - float(yv)) <= 0.5 + 1e-9)[0]
        if len(idx) >= 2:
            d = pts_c[idx[-1]] - pts_c[idx[0]]
            h_dirs.append(d / (np.linalg.norm(d) + 1e-12))
    for xv in np.unique(np.round(pts_proj[:, 0], 0)):
        idx = np.where(np.abs(pts_proj[:, 0] - float(xv)) <= 0.5 + 1e-9)[0]
        if len(idx) >= 2:
            d = pts_c[idx[-1]] - pts_c[idx[0]]
            v_dirs.append(d / (np.linalg.norm(d) + 1e-12))
    if h_dirs and v_dirs:
        hm = np.mean(h_dirs, axis=0)
        vm = np.mean(v_dirs, axis=0)
        cos = np.clip(np.abs(np.dot(hm, vm)), 0, 1)
        ang = float(np.degrees(np.arccos(cos)))
        ortho = abs(90.0 - ang)

    # Outer corners: extreme desired ids by position
    des_all = np.array(list(desired_ids.values()), dtype=np.float64)
    # Parallelism / aspect from detected hull
    rect = cv2.minAreaRect(pts_c.astype(np.float32))
    (_cxy), (rw, rh), _ang = rect
    rw, rh = float(max(rw, rh)), float(min(rw, rh)) if min(rw, rh) > 1 else float(max(rw, 1))
    # keep width>=height naming as longer/shorter for aspect vs desired
    measured_aspect = float(max(rect[1][0], rect[1][1]) / max(min(rect[1][0], rect[1][1]), 1e-6))
    desired_aspect = float(desired.get("desired_aspect_ratio") or 1.0)
    # Prefer comparing board extent aspect in desired space
    dmin, dmax = des_all.min(0), des_all.max(0)
    desired_aspect = float((dmax[0] - dmin[0]) / max(dmax[1] - dmin[1], 1e-6))
    cmin, cmax = pts_c.min(0), pts_c.max(0)
    measured_aspect = float((cmax[0] - cmin[0]) / max(cmax[1] - cmin[1], 1e-6))
    aspect_err = abs(measured_aspect - desired_aspect) / max(desired_aspect, 1e-6)

    # Four outer corners by desired extremes
    corner_errs = []
    for selector in (
        lambda a: (a[:, 0] + a[:, 1]).argmin(),
        lambda a: (a[:, 0] - a[:, 1]).argmax(),
        lambda a: (a[:, 0] + a[:, 1]).argmax(),
        lambda a: (-a[:, 0] + a[:, 1]).argmax(),
    ):
        # pick among common by desired positions
        di = np.array([desired_ids[i] for i in common])
        j = int(selector(di))
        corner_errs.append(float(err[j]))

    box = cv2.boxPoints(rect).astype(np.float64)
    # opposite edge parallelism
    e01 = box[1] - box[0]
    e12 = box[2] - box[1]
    e23 = box[3] - box[2]
    e30 = box[0] - box[3]
    def _ang(a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))

    parallelism = float(np.mean([_ang(e01, e23), _ang(e12, e30)]))

    secondary = secondary_aa_hull_metric(pts_c)
    result = {
        "valid": True,
        "pass": True,
        "tag": tag,
        "metric_category": "B_production_correction",
        "n_valid_measurements": len(common),
        "matched_ids": common,
        "target_median_err_px": float(np.median(err)),
        "target_p95_err_px": float(np.percentile(err, 95)),
        "target_max_err_px": float(err.max()),
        "target_mean_err_px": float(err.mean()),
        "corner_position_err_px": float(np.mean(corner_errs)),
        "grid_intersection_err_px": float(err.mean()),
        "orthogonality_error_deg": ortho,
        "opposite_edge_parallelism_deg": parallelism,
        "aspect_ratio_error": float(aspect_err),
        "measured_aspect_ratio": measured_aspect,
        "desired_aspect_ratio": desired_aspect,
        "horizontal_axis_deviation_deg_median": axis.get("horizontal_axis_deviation_deg_median"),
        "vertical_axis_deviation_deg_median": axis.get("vertical_axis_deviation_deg_median"),
        "horizontal_axis_deviation_deg_max": axis.get("horizontal_axis_deviation_deg_max"),
        "vertical_axis_deviation_deg_max": axis.get("vertical_axis_deviation_deg_max"),
        # aliases for compare helpers / older keys
        "median_reproj_px": float(np.median(err)),
        "p95_reproj_px": float(np.percentile(err, 95)),
        "max_reproj_px": float(err.max()),
        "mean_reproj_px": float(err.mean()),
        "median_residual_px": float(np.median(err)),
        "p95_residual_px": float(np.percentile(err, 95)),
        "detection": det_meta,
        "matching": match_meta,
        "secondary_aa_hull_metric": secondary,
        "secondary_metric_note": "AA-hull is secondary visualization only; not used for acceptance.",
    }
    return result


def compare_reproj(unc: dict, cor: dict) -> dict:
    """Acceptance from production target metrics (category B), not fit residuals alone."""
    if not unc.get("valid") or not cor.get("valid"):
        reasons = []
        if not unc.get("valid"):
            reasons.append(f"uncorrected:{unc.get('failure_reason')}")
        if not cor.get("valid"):
            reasons.append(f"corrected:{cor.get('failure_reason')}")
        return {
            "valid": False,
            "improvement": {},
            "absolute_gate_pass": False,
            "relative_gate_pass": False,
            "acceptance_pass": False,
            "failure_reason": "; ".join(reasons),
        }

    def _get(d: dict, *keys: str):
        for k in keys:
            if d.get(k) is not None:
                return d[k]
        return None

    med_u = _get(unc, "target_median_err_px", "median_reproj_px")
    med_c = _get(cor, "target_median_err_px", "median_reproj_px")
    p95_u = _get(unc, "target_p95_err_px", "p95_reproj_px")
    p95_c = _get(cor, "target_p95_err_px", "p95_reproj_px")
    improvement = {
        "target_median_err_px": {
            "uncorrected": med_u,
            "corrected": med_c,
            "relative_improvement": None if not med_u else (med_u - med_c) / med_u,
        },
        "target_p95_err_px": {
            "uncorrected": p95_u,
            "corrected": p95_c,
            "relative_improvement": None if not p95_u else (p95_u - p95_c) / p95_u,
        },
    }
    target_improve = (
        isinstance(med_u, (int, float))
        and isinstance(med_c, (int, float))
        and isinstance(p95_u, (int, float))
        and isinstance(p95_c, (int, float))
        and med_c < med_u
        and p95_c < p95_u
    )
    already_good = (
        isinstance(med_u, (int, float))
        and isinstance(med_c, (int, float))
        and isinstance(p95_c, (int, float))
        and med_u <= 2.0
        and med_c <= 2.0
        and p95_c <= 5.0
    )
    ax_u = unc.get("horizontal_axis_deviation_deg_median")
    ax_c = cor.get("horizontal_axis_deviation_deg_median")
    axis_improve = (
        isinstance(ax_u, (int, float))
        and isinstance(ax_c, (int, float))
        and (ax_c < ax_u or ax_c <= 1.0)
    )
    ortho_u = unc.get("orthogonality_error_deg")
    ortho_c = cor.get("orthogonality_error_deg")
    ortho_improve = (
        isinstance(ortho_u, (int, float))
        and isinstance(ortho_c, (int, float))
        and ortho_c <= ortho_u + 1e-6
    )
    geo_ok = bool(
        (isinstance(ax_c, (int, float)) and ax_c <= 2.0)
        or (isinstance(ortho_c, (int, float)) and ortho_c <= 5.0)
        or axis_improve
        or ortho_improve
    )
    abs_pass = bool(geo_ok and med_c <= 5.0 and p95_c <= 12.0 and (target_improve or already_good))
    rel_pass = bool(
        (target_improve or already_good)
        and geo_ok
        and (
            already_good
            or (
                improvement["target_median_err_px"]["relative_improvement"] is not None
                and improvement["target_median_err_px"]["relative_improvement"] >= 0.5
                and improvement["target_p95_err_px"]["relative_improvement"] is not None
                and improvement["target_p95_err_px"]["relative_improvement"] >= 0.4
            )
        )
    )
    cov_ok = cor.get("coverage", {}).get("coverage_gate_pass", True)
    sec_u = (unc.get("secondary_aa_hull_metric") or {}).get("aa_corner_median_err_px")
    sec_c = (cor.get("secondary_aa_hull_metric") or {}).get("aa_corner_median_err_px")
    acceptance = bool((abs_pass or rel_pass) and cov_ok)
    return {
        "valid": True,
        "improvement": improvement,
        "target_median_before": med_u,
        "target_median_after": med_c,
        "target_p95_before": p95_u,
        "target_p95_after": p95_c,
        "axis_deviation_before_deg": ax_u,
        "axis_deviation_after_deg": ax_c,
        "axis_improvement_pass": axis_improve,
        "orthogonality_before_deg": ortho_u,
        "orthogonality_after_deg": ortho_c,
        "absolute_gate_pass": abs_pass,
        "relative_gate_pass": rel_pass,
        "geometry_gate_pass": geo_ok,
        "coverage_gate_pass": cov_ok,
        "acceptance_pass": acceptance,
        "failure_reason": None if acceptance else "acceptance_thresholds_not_met",
        "secondary_aa_before": sec_u,
        "secondary_aa_after": sec_c,
        "secondary_aa_note": "diagnostic only; not used for acceptance",
        "acceptance_metric_family": "B_production_desired_target",
    }


def valid_destination_roi(
    H_cam_from_proj: np.ndarray,
    proj_w: int,
    proj_h: int,
    cam_w: int,
    cam_h: int,
    margin_frac: float = 0.08,
) -> tuple[int, int, int, int]:
    """Estimate ProjFB ROI that maps inside the camera frame after pre-warp considerations.

    Samples a grid in ProjFB, keeps points whose H maps inside CamImg, returns
    bounding box shrunk by margin for safe validation board placement.
    """
    xs = np.linspace(0, proj_w - 1, 40)
    ys = np.linspace(0, proj_h - 1, 30)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    cam = apply_H(H_cam_from_proj, grid)
    inside = (
        (cam[:, 0] >= 0)
        & (cam[:, 0] < cam_w)
        & (cam[:, 1] >= 0)
        & (cam[:, 1] < cam_h)
    )
    if inside.sum() < 10:
        m = int(min(proj_w, proj_h) * margin_frac)
        return m, m, proj_w - 1 - m, proj_h - 1 - m
    pts = grid[inside]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    mx = (x1 - x0) * margin_frac
    my = (y1 - y0) * margin_frac
    return (
        int(max(0, np.floor(x0 + mx))),
        int(max(0, np.floor(y0 + my))),
        int(min(proj_w - 1, np.ceil(x1 - mx))),
        int(min(proj_h - 1, np.ceil(y1 - my))),
    )


def save_board_spec(path: Path, spec: BoardSpec = DEFAULT_BOARD_SPEC) -> None:
    path.write_text(json.dumps(asdict(spec), indent=2))


def load_proj_ids_json(path: Path) -> dict[int, np.ndarray]:
    raw = json.loads(path.read_text())
    return {int(k): np.array(v, dtype=np.float64) for k, v in raw.items()}
