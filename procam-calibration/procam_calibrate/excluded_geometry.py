"""Active wall vs excluded plane (ceiling) classification and usable masks.

Production content remains two-plane. A third planar cluster may be detected
only for exclusion (ceiling), never as an artistic surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import apply_H, estimate_H_cam_from_proj
from .two_plane import MIN_POINTS_PER_PLANE, fit_single_homography, fit_two_homographies
from .two_plane_observations import canonicalize_plane_labels


def _reproj_err(H: np.ndarray, pts_p: np.ndarray, pts_c: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_H(H, pts_p) - pts_c, axis=1)


def classify_active_and_excluded_planes(
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
    ids: Optional[list] = None,
    ransac_threshold: float = 3.0,
    ceiling_y_frac: float = 0.35,
    proj_ceiling_frac: float = 0.28,
) -> dict:
    """Fit up to three planar clusters; label ceiling as excluded.

    Strategy: peel likely ceiling points (upper projector band and/or upper
    camera band) *before* fitting the two active wall homographies so ceiling
    correspondences cannot contaminate walls A/B.
    """
    pts_proj = np.asarray(pts_proj, dtype=np.float64)
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    n = len(pts_proj)
    if ids is None:
        ids = list(range(n))

    labels = np.full(n, -1, dtype=np.int32)  # -1 outlier, 0 A, 1 B, 2 ceiling
    H_ceil = None
    ceiling_detected = False

    # Candidate ceiling: upper projector band OR strongly upper camera band
    proj_h_est = float(max(pts_proj[:, 1].max(), 1.0))
    upper_p = pts_proj[:, 1] <= proj_ceiling_frac * proj_h_est
    y_cut_cam = np.percentile(pts_cam[:, 1], max(5.0, 100 * ceiling_y_frac * 0.5))
    upper_c = pts_cam[:, 1] <= y_cut_cam
    ceil_cand = upper_p | (upper_c & (pts_proj[:, 1] <= 0.45 * proj_h_est))

    if ceil_cand.sum() >= 8:
        H_ceil, e_c, _ = fit_single_homography(
            pts_proj[ceil_cand], pts_cam[ceil_cand], ransac_threshold
        )
        # Keep as ceiling those well explained by H_ceil
        err_c = _reproj_err(H_ceil, pts_proj, pts_cam)
        # Tentative: candidates with good ceiling fit
        ceil_keep = ceil_cand & (err_c < max(6.0, ransac_threshold * 2.5))
        if ceil_keep.sum() >= 8:
            labels[ceil_keep] = 2
            ceiling_detected = True
        else:
            H_ceil = None

    wall_mask = labels != 2
    if wall_mask.sum() < 2 * MIN_POINTS_PER_PLANE:
        return {
            "n_clusters": int(ceiling_detected),
            "model": "failed",
            "active_walls": 0,
            "ceiling_detected": ceiling_detected,
            "labels": labels.tolist(),
            "assignment": {
                str(ids[i]): ("ceiling" if labels[i] == 2 else "outlier") for i in range(n)
            },
            "H_cam_from_proj_plane_A": None,
            "H_cam_from_proj_plane_B": None,
            "H_cam_from_proj_ceiling": H_ceil,
            "valid_active": False,
            "reason": "insufficient_points_after_ceiling_peel",
            "note": "Ceiling is excluded from content; not a third artistic plane",
            "diagnostic": "Need more wall coverage below the ceiling band",
        }

    two = fit_two_homographies(
        pts_proj[wall_mask],
        pts_cam[wall_mask],
        ids=[ids[i] for i in np.where(wall_mask)[0]],
        ransac_threshold=ransac_threshold,
    )

    if two.get("model") == "one_plane":
        labels[wall_mask] = 0
        return {
            "n_clusters": 1 + int(ceiling_detected),
            "model": "one_plane_with_excluded_ceiling" if ceiling_detected else "one_plane",
            "active_walls": 1,
            "ceiling_detected": ceiling_detected,
            "labels": labels.tolist(),
            "assignment": {
                str(ids[i]): ("ceiling" if labels[i] == 2 else "A") for i in range(n)
            },
            "H_cam_from_proj_plane_A": two.get("H_cam_from_proj_single"),
            "H_cam_from_proj_plane_B": None,
            "H_cam_from_proj_ceiling": H_ceil,
            "valid_active": True,
            "reason": "one_plane_after_ceiling_peel",
            "note": "Ceiling is excluded from content; not a third artistic plane",
        }

    if two.get("model") != "two_plane" or not two.get("valid"):
        # Fallback: camera-upper peel (may already have failed projector peel)
        return _three_cluster_fallback(pts_proj, pts_cam, ids, ransac_threshold, ceiling_y_frac)

    Ha = two["H_cam_from_proj_plane_A"]
    Hb = two["H_cam_from_proj_plane_B"]
    err_a = _reproj_err(Ha, pts_proj, pts_cam)
    err_b = _reproj_err(Hb, pts_proj, pts_cam)
    lab = (err_b < err_a).astype(np.int32)
    lab = canonicalize_plane_labels(pts_proj, lab)
    # Preserve ceiling labels
    lab[labels == 2] = 2
    # Optionally promote high-residual upper points to ceiling
    err = np.minimum(err_a, err_b)
    residual = (labels != 2) & (err > max(ransac_threshold * 3.0, 10.0)) & (
        pts_cam[:, 1] < np.percentile(pts_cam[:, 1], 40)
    )
    if residual.sum() >= 8 and H_ceil is not None:
        err_c = _reproj_err(H_ceil, pts_proj, pts_cam)
        promote = residual & (err_c < err * 0.75)
        lab[promote] = 2
        ceiling_detected = True
    labels = lab

    # Refit A/B excluding ceiling
    mask_a = labels == 0
    mask_b = labels == 1
    if mask_a.sum() >= MIN_POINTS_PER_PLANE and mask_b.sum() >= MIN_POINTS_PER_PLANE:
        Ha, _, _ = fit_single_homography(pts_proj[mask_a], pts_cam[mask_a], ransac_threshold)
        Hb, _, _ = fit_single_homography(pts_proj[mask_b], pts_cam[mask_b], ransac_threshold)
        ab = (labels == 0) | (labels == 1)
        can = canonicalize_plane_labels(pts_proj[ab], labels[ab])
        if not np.array_equal(can, labels[ab]):
            Ha, Hb = Hb, Ha
            new = labels.copy()
            new[ab] = can
            labels = new

    assignment = {
        str(ids[i]): {0: "A", 1: "B", 2: "ceiling", -1: "outlier"}.get(int(labels[i]), "outlier")
        for i in range(n)
    }
    n_a = int((labels == 0).sum())
    n_b = int((labels == 1).sum())
    valid = n_a >= MIN_POINTS_PER_PLANE and n_b >= MIN_POINTS_PER_PLANE

    return {
        "n_clusters": 2 + int(ceiling_detected),
        "model": "two_plane_with_excluded_ceiling" if ceiling_detected else "two_plane",
        "active_walls": 2 if valid else 0,
        "ceiling_detected": ceiling_detected,
        "labels": labels.tolist(),
        "assignment": assignment,
        "H_cam_from_proj_plane_A": Ha,
        "H_cam_from_proj_plane_B": Hb,
        "H_cam_from_proj_ceiling": H_ceil,
        "n_plane_A": n_a,
        "n_plane_B": n_b,
        "n_ceiling": int((labels == 2).sum()),
        "n_outlier": int((labels == -1).sum()),
        "valid_active": valid,
        "reason": "ok" if valid else "insufficient_active_wall_support",
        "note": "Ceiling is excluded from content; not a third artistic plane",
    }


def _three_cluster_fallback(pts_proj, pts_cam, ids, ransac_threshold, ceiling_y_frac) -> dict:
    """When two-plane fit fails, attempt sequential peel of upper cluster as ceiling."""
    n = len(pts_proj)
    # Peel upper camera points first
    y_cut = np.percentile(pts_cam[:, 1], 100 * ceiling_y_frac)
    upper = pts_cam[:, 1] <= y_cut
    labels = np.full(n, -1, dtype=np.int32)
    H_ceil = None
    if upper.sum() >= 8:
        H_ceil, _, _ = fit_single_homography(pts_proj[upper], pts_cam[upper], ransac_threshold)
        labels[upper] = 2
    rest = labels != 2
    if rest.sum() < 2 * MIN_POINTS_PER_PLANE:
        return {
            "n_clusters": int(upper.sum() >= 8),
            "model": "failed",
            "active_walls": 0,
            "ceiling_detected": bool(upper.sum() >= 8),
            "labels": labels.tolist(),
            "assignment": {str(ids[i]): ("ceiling" if labels[i] == 2 else "outlier") for i in range(n)},
            "H_cam_from_proj_plane_A": None,
            "H_cam_from_proj_plane_B": None,
            "H_cam_from_proj_ceiling": H_ceil,
            "valid_active": False,
            "reason": "cannot_form_two_active_walls_after_ceiling_peel",
            "diagnostic": "Third/upper cluster may be contaminating walls; check fold coverage",
            "note": "Ceiling is excluded from content; not a third artistic plane",
        }
    two = fit_two_homographies(
        pts_proj[rest], pts_cam[rest], ids=[ids[i] for i in np.where(rest)[0]], ransac_threshold=ransac_threshold
    )
    if two.get("model") != "two_plane" or not two.get("valid"):
        return {
            "n_clusters": 1 + int(H_ceil is not None),
            "model": "failed",
            "active_walls": 0,
            "ceiling_detected": H_ceil is not None,
            "labels": labels.tolist(),
            "assignment": {str(ids[i]): ("ceiling" if labels[i] == 2 else "outlier") for i in range(n)},
            "H_cam_from_proj_plane_A": None,
            "H_cam_from_proj_plane_B": None,
            "H_cam_from_proj_ceiling": H_ceil,
            "valid_active": False,
            "reason": "two_wall_fit_failed_after_ceiling_peel",
            "diagnostic": two.get("reason"),
            "note": "Ceiling is excluded from content; not a third artistic plane",
        }
    Ha = two["H_cam_from_proj_plane_A"]
    Hb = two["H_cam_from_proj_plane_B"]
    err_a = _reproj_err(Ha, pts_proj, pts_cam)
    err_b = _reproj_err(Hb, pts_proj, pts_cam)
    lab = (err_b < err_a).astype(np.int32)
    lab = canonicalize_plane_labels(pts_proj, lab)
    lab[labels == 2] = 2
    assignment = {
        str(ids[i]): {0: "A", 1: "B", 2: "ceiling", -1: "outlier"}[int(lab[i])] for i in range(n)
    }
    return {
        "n_clusters": 3,
        "model": "two_plane_with_excluded_ceiling",
        "active_walls": 2,
        "ceiling_detected": True,
        "labels": lab.tolist(),
        "assignment": assignment,
        "H_cam_from_proj_plane_A": Ha,
        "H_cam_from_proj_plane_B": Hb,
        "H_cam_from_proj_ceiling": H_ceil,
        "n_plane_A": int((lab == 0).sum()),
        "n_plane_B": int((lab == 1).sum()),
        "n_ceiling": int((lab == 2).sum()),
        "valid_active": True,
        "reason": "ok_after_ceiling_peel",
        "note": "Ceiling is excluded from content; not a third artistic plane",
    }


def estimate_wall_ceiling_boundary(
    pts_cam: np.ndarray,
    labels: np.ndarray,
    cam_w: int,
    cam_h: int,
    arch_horizontal_lines: Optional[list[dict]] = None,
) -> dict:
    """Estimate a straight wall–ceiling boundary in camera coordinates.

    Prefer interface between wall points and ceiling points; architectural
    horizontals are priors only.
    """
    labels = np.asarray(labels)
    wall = (labels == 0) | (labels == 1)
    ceil = labels == 2
    if ceil.sum() >= 4 and wall.sum() >= 4:
        # Boundary y ≈ midpoint between lowest ceiling and highest wall in each x band
        y_ceil = float(np.percentile(pts_cam[ceil, 1], 75))
        y_wall = float(np.percentile(pts_cam[wall, 1], 25))
        y_b = 0.5 * (y_ceil + y_wall)
        # Fit horizontal line y = y_b (MVP); allow slight tilt from ceiling x trend
        if ceil.sum() >= 8:
            xc, yc = pts_cam[ceil, 0], pts_cam[ceil, 1]
            A = np.vstack([xc, np.ones_like(xc)]).T
            slope, intercept = np.linalg.lstsq(A, yc, rcond=None)[0]
            # Use nearly-horizontal: clamp slope
            if abs(slope) > 0.25:
                slope = 0.0
                intercept = y_b
            else:
                intercept = float(intercept)
                slope = float(slope)
        else:
            slope, intercept = 0.0, y_b
        conf = float(min(1.0, ceil.sum() / 20.0))
        source = "correspondence_ceiling_cluster"
    else:
        # Prior from longest high horizontal architectural line in upper third
        slope, intercept = 0.0, cam_h * 0.22
        conf = 0.2
        source = "weak_prior_default"
        if arch_horizontal_lines:
            upper = [
                ln
                for ln in arch_horizontal_lines
                if 0.5 * (ln["y1"] + ln["y2"]) < 0.4 * cam_h and ln["length"] > 0.2 * cam_w
            ]
            if upper:
                best = max(upper, key=lambda L: L["length"] * L.get("confidence", 1.0))
                intercept = 0.5 * (best["y1"] + best["y2"])
                slope = (best["y2"] - best["y1"]) / (best["x2"] - best["x1"] + 1e-9)
                if abs(slope) > 0.25:
                    slope = 0.0
                conf = 0.35
                source = "architectural_horizontal_prior"
    # Endpoints across image width
    x0, x1 = 0.0, float(cam_w - 1)
    y0 = slope * x0 + intercept
    y1 = slope * x1 + intercept
    return {
        "type": "line",
        "slope": float(slope),
        "intercept": float(intercept),
        "endpoint_0": [x0, float(y0)],
        "endpoint_1": [x1, float(y1)],
        "confidence": conf,
        "source": source,
        "coordinate_space": "camera_pixels",
        "convention": "pixels with y > boundary (below line in image) are wall-side when slope≈0",
        "note": "Ceiling is above the boundary (smaller y). Not authoritative alone.",
    }


def build_usable_and_exclusion_masks(
    seam: dict,
    ceiling_boundary: dict,
    H_a: np.ndarray,
    H_b: np.ndarray,
    proj_w: int,
    proj_h: int,
    cam_w: int,
    cam_h: int,
    margin_px: float = 8.0,
    target_region_cam: Optional[list] = None,
) -> dict:
    """Build camera/projector usable masks and ceiling exclusion masks."""
    # Projector masks from seam (walls only)
    yy, xx = np.mgrid[0:proj_h, 0:proj_w]
    a, b, c = seam["line_abc"]
    side = a * xx + b * yy + c
    mask_a = (side < 0).astype(np.uint8) * 255
    mask_b = (side >= 0).astype(np.uint8) * 255

    # Map every projector pixel → camera via plane H; mark ceiling
    # Sample grid for speed then upsample classification
    step = 4
    xs = np.arange(0, proj_w, step)
    ys = np.arange(0, proj_h, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
    side_s = a * pts[:, 0] + b * pts[:, 1] + c
    cam_a = apply_H(H_a, pts)
    cam_b = apply_H(H_b, pts)
    cam = np.where(side_s[:, None] < 0, cam_a, cam_b)
    slope = ceiling_boundary["slope"]
    intercept = ceiling_boundary["intercept"]
    # Ceiling: camera y above boundary (smaller y) with margin
    y_bound = slope * cam[:, 0] + intercept + margin_px
    is_ceiling = cam[:, 1] < y_bound
    in_frame = (
        (cam[:, 0] >= 0)
        & (cam[:, 0] < cam_w)
        & (cam[:, 1] >= 0)
        & (cam[:, 1] < cam_h)
    )
    usable_s = (~is_ceiling) & in_frame

    usable_proj = np.zeros((proj_h, proj_w), dtype=np.uint8)
    ceil_proj = np.zeros((proj_h, proj_w), dtype=np.uint8)
    for i, (x, y) in enumerate(pts.astype(int)):
        x1 = min(proj_w, x + step)
        y1 = min(proj_h, y + step)
        if usable_s[i]:
            usable_proj[y:y1, x:x1] = 255
        if is_ceiling[i] and in_frame[i]:
            ceil_proj[y:y1, x:x1] = 255

    # Restrict usable to wall masks
    usable_proj = cv2.bitwise_and(usable_proj, cv2.bitwise_or(mask_a, mask_b))

    # Camera-space masks via warp of projector masks (approx densify)
    usable_cam = np.zeros((cam_h, cam_w), dtype=np.uint8)
    ceil_cam = np.zeros((cam_h, cam_w), dtype=np.uint8)
    # Rasterize sampled camera points
    for i, (u, v) in enumerate(cam.astype(int)):
        if not (0 <= u < cam_w and 0 <= v < cam_h):
            continue
        if usable_s[i]:
            cv2.circle(usable_cam, (u, v), step, 255, -1)
        if is_ceiling[i]:
            cv2.circle(ceil_cam, (u, v), step, 255, -1)
    # Fill: wall below boundary
    yy_c, xx_c = np.mgrid[0:cam_h, 0:cam_w]
    below = (yy_c >= (slope * xx_c + intercept + margin_px)).astype(np.uint8) * 255
    # Usable camera = below boundary AND (approx) projector-reachable fill
    if usable_cam.max() == 0:
        usable_cam = below
    else:
        usable_cam = cv2.bitwise_and(cv2.dilate(usable_cam, np.ones((7, 7), np.uint8)), below)
    ceil_cam = cv2.bitwise_and(
        ((yy_c < (slope * xx_c + intercept + margin_px)).astype(np.uint8) * 255),
        np.ones((cam_h, cam_w), dtype=np.uint8) * 255,
    )

    if target_region_cam is not None:
        poly = np.array(target_region_cam, dtype=np.int32)
        tr = np.zeros((cam_h, cam_w), dtype=np.uint8)
        cv2.fillPoly(tr, [poly], 255)
        usable_cam = cv2.bitwise_and(usable_cam, tr)

    confidence = np.zeros((cam_h, cam_w), dtype=np.float32)
    confidence[usable_cam > 0] = 0.8
    confidence[ceil_cam > 0] = 0.1
    if ceiling_boundary.get("confidence", 0) < 0.3:
        # shrink usable near boundary
        er = cv2.erode(usable_cam, np.ones((int(margin_px) * 2 + 1, int(margin_px) * 2 + 1), np.uint8))
        usable_cam = er

    return {
        "usable_mask_projector": usable_proj,
        "usable_mask_camera": usable_cam,
        "excluded_ceiling_mask_projector": ceil_proj,
        "excluded_ceiling_mask_camera": ceil_cam,
        "mask_plane_A": mask_a,
        "mask_plane_B": mask_b,
        "confidence_map_camera": confidence,
        "margin_px": margin_px,
        "stats": {
            "usable_proj_frac": float((usable_proj > 0).mean()),
            "usable_cam_frac": float((usable_cam > 0).mean()),
            "ceiling_cam_frac": float((ceil_cam > 0).mean()),
        },
    }


def save_exclusion_artifacts(out_dir: Path, masks: dict, boundary: dict, classification: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in (
        "usable_mask_projector",
        "usable_mask_camera",
        "excluded_ceiling_mask_projector",
        "excluded_ceiling_mask_camera",
        "mask_plane_A",
        "mask_plane_B",
    ):
        if key in masks and masks[key] is not None:
            cv2.imwrite(str(out_dir / f"{key}.png"), masks[key])
    if "confidence_map_camera" in masks:
        conf = (np.clip(masks["confidence_map_camera"], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / "confidence_map_camera.png"), conf)
    (out_dir / "wall_ceiling_boundary.json").write_text(json.dumps(boundary, indent=2))
    serial = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in classification.items()
        if k
        not in (
            "H_cam_from_proj_plane_A",
            "H_cam_from_proj_plane_B",
            "H_cam_from_proj_ceiling",
            "two_plane_fit",
        )
    }
    if classification.get("H_cam_from_proj_plane_A") is not None:
        np.save(out_dir / "H_cam_from_proj_plane_A.npy", classification["H_cam_from_proj_plane_A"])
        np.save(out_dir / "H_cam_from_proj_plane_B.npy", classification["H_cam_from_proj_plane_B"])
    if classification.get("H_cam_from_proj_ceiling") is not None:
        np.save(out_dir / "H_cam_from_proj_ceiling.npy", classification["H_cam_from_proj_ceiling"])
    (out_dir / "plane_classification.json").write_text(json.dumps(serial, indent=2, default=str))
    (out_dir / "mask_stats.json").write_text(json.dumps(masks.get("stats", {}), indent=2))
