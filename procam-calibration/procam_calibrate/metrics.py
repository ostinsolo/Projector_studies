"""Geometric validation metrics — production uses independent desired target (category B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .charuco import (
    compare_reproj,
    define_desired_target,
    detect_charuco,
    detection_coverage,
    expected_visible_from_pattern,
    load_desired_target,
    load_proj_ids_json,
    match_correspondences,
    measure_against_desired_target,
    measure_independent_residuals,
)


def detect_outer_rectangle_corners(img_bgr: np.ndarray) -> np.ndarray | None:
    """Secondary diagnostic only."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
    else:
        box = approx.reshape(4, 2).astype(np.float32)
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    tl, br = box[np.argmin(s)], box[np.argmax(s)]
    tr, bl = box[np.argmin(diff)], box[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def measure_charuco_against_projector(
    capture_path: Path,
    proj_ids: dict[int, np.ndarray],
    H_cam_from_proj: Optional[np.ndarray],
    out_dir: Path,
    tag: str,
    fit_stats: Optional[dict] = None,
    refit_homography: bool = True,
    desired: Optional[dict] = None,
    expected_visible: Optional[dict] = None,
) -> dict:
    """Measure capture. With desired target → category B production metrics."""
    del refit_homography
    img = cv2.imread(str(capture_path), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "valid": False,
            "pass": False,
            "tag": tag,
            "failure_reason": f"unreadable:{capture_path}",
            "target_median_err_px": None,
            "median_reproj_px": None,
        }
    cam_ids, det = detect_charuco(img)
    pts_p, pts_c, common, match_meta = match_correspondences(proj_ids, cam_ids)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_health = measure_independent_residuals(
        pts_p,
        pts_c,
        H_cam_from_proj,
        common,
        tag=f"{tag}_fit_health",
        det_meta=det,
        match_meta=match_meta,
        fit_stats=fit_stats,
        refit_homography=True,
    )

    if desired is None:
        result = {
            **fit_health,
            "valid": False,
            "pass": False,
            "failure_reason": "no_desired_target_for_production_metrics",
            "target_median_err_px": None,
            "target_p95_err_px": None,
            "target_max_err_px": None,
            "calibration_fit_health": {
                "median_fit_residual_px": fit_health.get("median_reproj_px"),
                "p95_fit_residual_px": fit_health.get("p95_reproj_px"),
                "note": "Category A only — production requires desired_target.json",
            },
        }
    else:
        result = measure_against_desired_target(
            cam_ids,
            desired,
            tag=tag,
            det_meta=det,
            match_meta=match_meta,
            proj_ids=proj_ids,
        )
        if expected_visible is not None:
            exp_n = int(expected_visible["expected_visible_corners"])
            excl = list(expected_visible.get("intentionally_excluded_ids") or [])
            vis = set(expected_visible.get("expected_visible_ids") or cam_ids.keys())
            cam_for_cov = {i: p for i, p in cam_ids.items() if i in vis}
            result["expected_visible"] = expected_visible
        else:
            exp_n = len(desired.get("desired_cam_by_id") or proj_ids)
            excl = []
            cam_for_cov = cam_ids
        region = np.array(list((desired.get("desired_cam_by_id") or {}).values()), dtype=np.float64)
        result["coverage"] = detection_coverage(
            cam_for_cov,
            img.shape[:2],
            expected_visible=exp_n,
            intentionally_excluded=excl,
            region_pts=region if len(region) else None,
        )
        result["calibration_fit_health"] = {
            "median_fit_residual_px": fit_health.get("median_reproj_px"),
            "p95_fit_residual_px": fit_health.get("p95_reproj_px"),
            "max_fit_residual_px": fit_health.get("max_reproj_px"),
            "n_inliers": (fit_health.get("homography_fit") or {}).get("n_inliers"),
            "note": "Category A — not sole acceptance criterion",
        }

    result["capture_path"] = str(capture_path)
    result["image_size"] = [img.shape[1], img.shape[0]]
    vis_img = img.copy()
    for p in pts_c.astype(int):
        cv2.circle(vis_img, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
    if desired is not None:
        for i in common:
            d = desired["desired_cam_by_id"][i]
            cv2.circle(vis_img, (int(d[0]), int(d[1])), 4, (0, 0, 255), 1)
            c = cam_ids[i]
            cv2.line(vis_img, (int(c[0]), int(c[1])), (int(d[0]), int(d[1])), (255, 128, 0), 1)
    cv2.imwrite(str(out_dir / f"{tag}_charuco_overlay.png"), vis_img)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def measure_validation_image(path: Path, out_dir: Path, tag: str) -> dict:
    """Validate a capture using run-dir ChArUco context + desired target when present."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "valid": False,
            "pass": False,
            "tag": tag,
            "failure_reason": f"unreadable:{path}",
            "target_median_err_px": None,
            "median_reproj_px": None,
        }

    run_dir = (
        path.parent.parent
        if path.parent.name
        in {"captured_validation", "captures", "camera_captures_validated", "camera_captures_raw"}
        else path.parent
    )
    candidates = [
        run_dir / "homography_baseline",
        run_dir / "calibration" / "homography",
        run_dir,
    ]
    proj_ids = None
    H = None
    desired = None
    expected_visible = None
    base_used = None
    for base in candidates:
        ids_path = base / "proj_corner_ids.json"
        h_path = base / "H_cam_from_proj.npy"
        dt_path = base / "desired_target.json"
        if ids_path.exists() and h_path.exists():
            proj_ids = load_proj_ids_json(ids_path)
            H = np.load(h_path)
            base_used = base
            if dt_path.exists():
                desired = load_desired_target(dt_path)
            break

    if proj_ids is not None and H is not None:
        proj_w, proj_h = 1920, 1080
        for base in candidates:
            pat = base / "patterns" / "charuco_calib.png"
            if pat.exists():
                im = cv2.imread(str(pat))
                if im is not None:
                    proj_h, proj_w = im.shape[:2]
                    break
        if desired is None:
            desired = define_desired_target(H, proj_w, proj_h, proj_ids)
        if base_used is not None:
            pre = base_used / "prewarp_charuco.png"
            if pre.exists():
                pre_img = cv2.imread(str(pre))
                if pre_img is not None and tag == "corrected":
                    expected_visible = expected_visible_from_pattern(pre_img, proj_ids)
        return measure_charuco_against_projector(
            path, proj_ids, H, out_dir, tag, desired=desired, expected_visible=expected_visible
        )

    cam_ids, det = detect_charuco(img)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = img.copy()
    for p in cam_ids.values():
        cv2.circle(vis, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
    result = {
        "valid": False,
        "pass": False,
        "tag": tag,
        "failure_reason": "no_H_cam_from_proj_for_independent_target",
        "n_valid_measurements": len(cam_ids),
        "detection": det,
        "target_median_err_px": None,
        "target_p95_err_px": None,
        "target_max_err_px": None,
        "median_reproj_px": None,
        "p95_reproj_px": None,
        "max_reproj_px": None,
        "mean_residual_px": None,
        "median_residual_px": None,
        "p95_residual_px": None,
        "max_residual_px": None,
        "note": "ChArUco detector ran (shared module); acceptance requires desired target + H.",
    }
    cv2.imwrite(str(out_dir / f"{tag}_overlay.png"), vis)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def compare_uncorrected_corrected(unc: dict, cor: dict) -> dict:
    return compare_reproj(unc, cor)


def measure_pair_with_homography(
    unc_path: Path,
    cor_path: Path,
    proj_ids: dict[int, np.ndarray],
    H_cam_from_proj: np.ndarray,
    out_dir: Path,
    desired: Optional[dict] = None,
    proj_w: int = 1920,
    proj_h: int = 1080,
    prewarp_pattern_path: Optional[Path] = None,
) -> dict:
    """Full before/after measurement against independent desired target."""
    if desired is None:
        desired = define_desired_target(H_cam_from_proj, proj_w, proj_h, proj_ids)
    pat = unc_path.parent.parent / "patterns" / "charuco_calib.png"
    if pat.exists():
        exp_unc = expected_visible_from_pattern(cv2.imread(str(pat)), proj_ids)
    else:
        exp_unc = {
            "expected_visible_ids": sorted(proj_ids.keys()),
            "expected_visible_corners": len(proj_ids),
            "intentionally_excluded_ids": [],
        }
    exp_cor = exp_unc
    if prewarp_pattern_path is not None and Path(prewarp_pattern_path).exists():
        exp_cor = expected_visible_from_pattern(cv2.imread(str(prewarp_pattern_path)), proj_ids)
        mask_path = Path(prewarp_pattern_path).parent / "expected_visible_mask.json"
        mask_path.write_text(json.dumps(exp_cor, indent=2, default=str))
    unc = measure_charuco_against_projector(
        unc_path,
        proj_ids,
        H_cam_from_proj,
        out_dir,
        "uncorrected",
        desired=desired,
        expected_visible=exp_unc,
    )
    cor = measure_charuco_against_projector(
        cor_path,
        proj_ids,
        H_cam_from_proj,
        out_dir,
        "corrected",
        desired=desired,
        expected_visible=exp_cor,
    )
    comparison = compare_reproj(unc, cor)
    return {
        "uncorrected": unc,
        "corrected": cor,
        "comparison": comparison,
        "desired_target_summary": {
            "proj_resolution": desired.get("proj_resolution"),
            "desired_aspect_ratio": desired.get("desired_aspect_ratio"),
            "n_desired_ids": len(desired.get("desired_cam_by_id") or {}),
        },
        "calib_H_shape": list(H_cam_from_proj.shape),
    }
