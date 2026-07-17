"""Flat-wall production path: ChArUco + planar homography (uses shared charuco module)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .camera_control import CameraController, list_avfoundation_video_devices, select_iphone_camera
from .charuco import (
    MIN_CORNERS_CALIB,
    MIN_CORNERS_VALIDATE,
    compare_reproj,
    define_desired_target,
    detect_charuco,
    estimate_H_cam_from_proj,
    expected_visible_from_pattern,
    generate_charuco_image,
    match_correspondences,
    measure_independent_residuals,
    save_board_spec,
    save_desired_target,
    valid_destination_roi,
)
from .display_control import ProjectorController, list_displays, select_projector_display
from .metrics import measure_charuco_against_projector, measure_pair_with_homography
from .patterns import make_validation_target


def forward_H_proj_from_source(
    H_cam_from_proj: np.ndarray,
    H_desired_cam_from_proj: np.ndarray,
) -> np.ndarray:
    """Forward point transform: source content (ProjFB layout) → ProjFB display coords.

    Column-vector: p_proj ~ H_proj_from_source @ p_source
    Chosen so H_cam_from_proj @ H_proj_from_source ≈ H_desired_cam_from_proj.
    """
    H = np.linalg.inv(H_cam_from_proj) @ H_desired_cam_from_proj
    return (H / H[2, 2]).astype(np.float64)


def prewarp_from_forward_H(
    content_projfb: np.ndarray,
    H_proj_from_source: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> np.ndarray:
    """Rasterize content with OpenCV warpPerspective using forward H_proj_from_source.

    OpenCV: dst(p) = src(H^{-1} p). Passing H_proj_from_source makes content point s
    appear at projector pixel p = H_proj_from_source @ s.
    """
    return cv2.warpPerspective(
        content_projfb,
        H_proj_from_source.astype(np.float64),
        (proj_w, proj_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_forward_prewarp_meta(
    out_dir: Path,
    H_proj_from_source: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> Path:
    np.save(out_dir / "H_proj_from_source_current.npy", H_proj_from_source)
    meta = {
        "matrix_name": "H_proj_from_source_current",
        "source_space": "SourceContent_ProjFB_layout",
        "destination_space": "ProjFB",
        "source_resolution": [proj_w, proj_h],
        "projector_resolution": [proj_w, proj_h],
        "matrix": H_proj_from_source.tolist(),
        "matrix_normalization": "H /= H[2,2]",
        "opencv_warpPerspective_convention": (
            "dst(p)=src(H^{-1} p); matrix passed is H_proj_from_source (forward point map)"
        ),
        "interpolation": "INTER_LINEAR",
        "border_mode": "BORDER_CONSTANT",
        "border_value": [0, 0, 0],
        "pixel_centre_convention": "OpenCV top-left origin; +u right +v down",
        "distortion_state": "none_applied",
    }
    path = out_dir / "H_proj_from_source_current.json"
    path.write_text(json.dumps(meta, indent=2))
    return path


def prewarp_projfb_content(
    content_projfb: np.ndarray,
    H_cam_from_proj: np.ndarray,
    proj_w: int,
    proj_h: int,
    H_desired_cam_from_proj: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build ProjFB pre-warp so physical H maps content toward desired CamImg AA rect."""
    if H_desired_cam_from_proj is None:
        corners_p = np.array(
            [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
            dtype=np.float32,
        )
        from .charuco import apply_H

        cam = apply_H(H_cam_from_proj, corners_p.astype(np.float64)).astype(np.float32)
        x, y, w, h = cv2.boundingRect(cam.reshape(-1, 1, 2))
        w, h = max(int(w), 2), max(int(h), 2)
        desired = np.array(
            [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
            dtype=np.float32,
        )
        H_desired_cam_from_proj = cv2.getPerspectiveTransform(corners_p, desired).astype(np.float64)
    H_fwd = forward_H_proj_from_source(H_cam_from_proj, H_desired_cam_from_proj)
    return prewarp_from_forward_H(content_projfb, H_fwd, proj_w, proj_h)


def homography_dir(run_dir: Path) -> Path:
    d = run_dir / "homography_baseline"
    d.mkdir(parents=True, exist_ok=True)
    (d / "patterns").mkdir(exist_ok=True)
    (d / "captures").mkdir(exist_ok=True)
    return d


def save_proj_ids(path: Path, proj_ids: dict[int, np.ndarray]) -> None:
    path.write_text(json.dumps({str(k): v.tolist() for k, v in proj_ids.items()}, indent=2))


def ensure_production_homography_artifacts(run_dir: Path) -> dict[str, str]:
    """Copy H / ids / pre-warp into production locations used by auto-wall."""
    out = homography_dir(run_dir)
    cal = run_dir / "calibration" / "homography"
    cal.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in ("H_cam_from_proj.npy", "H_proj_from_cam.npy", "proj_corner_ids.json", "board_spec.json"):
        src = out / name
        if src.exists():
            dst = cal / name
            dst.write_bytes(src.read_bytes())
            copied[name] = str(dst)
    # Prefer charuco prewarp for production projection; fall back to validation target warp
    pre_src = out / "prewarp_charuco.png"
    if not pre_src.exists():
        pre_src = out / "prewarp_validation.png"
    if pre_src.exists():
        prewarps = run_dir / "prewarps"
        prewarps.mkdir(exist_ok=True)
        (prewarps / "prewarp_best.png").write_bytes(pre_src.read_bytes())
        pv = run_dir / "projected_validation"
        pv.mkdir(exist_ok=True)
        (pv / "prewarp_to_project.png").write_bytes(pre_src.read_bytes())
        copied["prewarp"] = str(prewarps / "prewarp_best.png")
    cor = out / "captures" / "charuco_corrected.png"
    if cor.exists():
        cap_val = run_dir / "captured_validation"
        cap_val.mkdir(exist_ok=True)
        (cap_val / "validation_corrected.png").write_bytes(cor.read_bytes())
        copied["validation_corrected"] = str(cap_val / "validation_corrected.png")
    return copied


def write_homography_final_report(
    run_dir: Path,
    metrics: dict,
    acceptance: bool,
    state: Optional[dict] = None,
) -> Path:
    """Write reproducibility report for the production homography MVP."""
    out = homography_dir(run_dir)
    unc = metrics.get("uncorrected") or {}
    cor = metrics.get("corrected") or {}
    comp = metrics.get("comparison") or {}
    baseline = {}
    br = out / "baseline_result.json"
    if br.exists():
        try:
            baseline = json.loads(br.read_text())
        except Exception:
            baseline = {}
    board = unc.get("detection", {}).get("board") or cor.get("detection", {}).get("board")
    report = run_dir / "report.md"
    repro = run_dir / "FINAL_REPRODUCIBILITY_REPORT.md"
    body = f"""# Flat-wall homography MVP — final report

## Acceptance: {'PASS / DONE' if acceptance else 'FAIL'}

## Pipeline

Production path (no neural training):

```
discover projector and camera
→ display ChArUco calibration board
→ capture
→ detect correspondences
→ estimate H_projector_to_camera
→ invert to H_camera_to_projector
→ generate projector pre-warp
→ display corrected validation board
→ capture
→ independently measure residual error
→ save result
→ DONE
```

## Reproduce

```bash
procam-calibrate auto-wall --run-dir {run_dir}
```

Or one-shot baseline:

```bash
procam-calibrate flatwall-homography --run-dir {run_dir}
```

## Primary metrics (independent known ProjFB ↔ detected CamImg)

Per-capture homography refit residual, affine residual from normalized projector
coords, and axis-alignment of board rows/cols (not contour AA).

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| median_reproj_px (H refit) | {unc.get('median_reproj_px')} | {cor.get('median_reproj_px')} |
| p95_reproj_px | {unc.get('p95_reproj_px')} | {cor.get('p95_reproj_px')} |
| max_reproj_px | {unc.get('max_reproj_px')} | {cor.get('max_reproj_px')} |
| corner_displacement_median_px | {unc.get('corner_displacement_median_px')} | {cor.get('corner_displacement_median_px')} |
| affine_median_reproj_px | {unc.get('affine_median_reproj_px')} | {cor.get('affine_median_reproj_px')} |
| horizontal_axis_deviation_deg | {unc.get('horizontal_axis_deviation_deg_median')} | {cor.get('horizontal_axis_deviation_deg_median')} |
| vertical_axis_deviation_deg | {unc.get('vertical_axis_deviation_deg_median')} | {cor.get('vertical_axis_deviation_deg_median')} |
| n_matched | {unc.get('n_valid_measurements')} | {cor.get('n_valid_measurements')} |

## Secondary visual metric (not used for acceptance)

| Metric | Uncorrected | Corrected |
|--------|-------------|-----------|
| aa_corner_median_err_px | {(unc.get('secondary_aa_hull_metric') or {}).get('aa_corner_median_err_px')} | {(cor.get('secondary_aa_hull_metric') or {}).get('aa_corner_median_err_px')} |

## Detection / board

- dictionary: `{(unc.get('detection') or {}).get('dictionary')}`
- board: `{json.dumps(board)}`
- expected markers/corners: see detection blocks below
- RANSAC inliers (fit): `{(baseline.get('homography_fit') or {}).get('n_inliers')}`

## Comparison / gates

```json
{json.dumps(comp, indent=2, default=str)}
```

## Artifacts preserved

- `{out / 'H_cam_from_proj.npy'}`
- `{out / 'H_proj_from_cam.npy'}`
- `{out / 'prewarp_charuco.png'}`
- `{run_dir / 'calibration' / 'homography'}`
- `{run_dir / 'prewarps' / 'prewarp_best.png'}`
- `{run_dir / 'metrics.json'}`

## Full metrics JSON

```json
{json.dumps(metrics, indent=2, default=str)}
```

## State history

```json
{json.dumps((state or {}).get('history', []), indent=2)}
```
"""
    report.write_text(body)
    repro.write_text(body)
    return report


def remeasure_saved_captures(run_dir: Path) -> dict:
    """Re-run independent metrics on saved uncorrected/corrected ChArUco captures."""
    out = homography_dir(run_dir)
    ids_path = out / "proj_corner_ids.json"
    h_path = out / "H_cam_from_proj.npy"
    unc = out / "captures" / "charuco_uncorrected.png"
    cor = out / "captures" / "charuco_corrected.png"
    if not all(p.exists() for p in (ids_path, h_path, unc, cor)):
        return {
            "valid": False,
            "error": "missing_saved_homography_artifacts",
            "comparison": {"valid": False, "acceptance_pass": False, "failure_reason": "missing_artifacts"},
            "paths": {
                "ids": str(ids_path),
                "H": str(h_path),
                "unc": str(unc),
                "cor": str(cor),
            },
        }
    from .charuco import load_proj_ids_json

    proj_ids = load_proj_ids_json(ids_path)
    H = np.load(h_path)
    immutable = run_dir.name in {
        "flatwall_20260717_141610",
        "flatwall_live_20260717_154312",
    }
    # Resolve projector resolution from pattern if present
    proj_w, proj_h = 1920, 1080
    pat = out / "patterns" / "charuco_calib.png"
    if pat.exists():
        im = cv2.imread(str(pat))
        if im is not None:
            proj_h, proj_w = im.shape[:2]
    from .charuco import load_desired_target

    dt_path = out / "desired_target.json"
    if dt_path.exists():
        desired = load_desired_target(dt_path)
    else:
        if immutable:
            return {
                "valid": False,
                "error": "missing_desired_target_on_immutable_run",
                "comparison": {
                    "valid": False,
                    "acceptance_pass": False,
                    "failure_reason": "immutable_missing_desired_target",
                },
            }
        desired = define_desired_target(H, proj_w, proj_h, proj_ids)
        save_desired_target(dt_path, desired)
    if immutable:
        metrics_dir = run_dir / "analysis_remeasure_replay" / "metrics"
    else:
        metrics_dir = out / "remeasure"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Prefer ROI ids for corrected if present (board was constrained)
    ids_roi = out / "proj_corner_ids_roi.json"
    if ids_roi.exists():
        from .charuco import load_proj_ids_json as _load

        proj_roi = _load(ids_roi)
        desired_roi = define_desired_target(H, proj_w, proj_h, proj_roi)
        unc_m = measure_charuco_against_projector(
            unc, proj_ids, H, metrics_dir, "uncorrected", desired=desired
        )
        cor_m = measure_charuco_against_projector(
            cor, proj_roi, H, metrics_dir, "corrected", desired=desired_roi
        )
        result = {
            "uncorrected": unc_m,
            "corrected": cor_m,
            "comparison": compare_reproj(unc_m, cor_m),
            "note": "corrected measured against ROI board projector IDs + desired target",
        }
    else:
        result = measure_pair_with_homography(
            unc,
            cor,
            proj_ids,
            H,
            metrics_dir,
            desired=desired,
            proj_w=proj_w,
            proj_h=proj_h,
            prewarp_pattern_path=out / "prewarp_charuco.png",
        )
    result["valid"] = bool(result.get("comparison", {}).get("valid"))
    if immutable:
        # Never overwrite original metrics / remeasure on preserved MVP runs
        side = run_dir / "analysis_remeasure_replay"
        side.mkdir(parents=True, exist_ok=True)
        (side / "remeasure_metrics.json").write_text(json.dumps(result, indent=2, default=str))
        result["immutable_run"] = True
        result["metrics_write_path"] = str(side / "remeasure_metrics.json")
    else:
        (out / "remeasure_metrics.json").write_text(json.dumps(result, indent=2, default=str))
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def remeasure_or_recapture(
    run_dir: Path,
    proj_w: int = 1920,
    proj_h: int = 1080,
    settle_s: float = 0.8,
    projector: Any = None,
    camera: Any = None,
    bind_devices: Any = None,
) -> dict:
    """Remeasure saved captures; if corrected corners too few, ROI board + recapture once."""
    result = remeasure_saved_captures(run_dir)
    if result.get("error") == "missing_saved_homography_artifacts":
        return result
    cor = result.get("corrected") or {}
    n = int(cor.get("n_valid_measurements") or 0)
    if cor.get("valid") and n >= MIN_CORNERS_VALIDATE:
        ensure_production_homography_artifacts(run_dir)
        return result

    # Need ROI recapture
    out = homography_dir(run_dir)
    h_path = out / "H_cam_from_proj.npy"
    unc = out / "captures" / "charuco_uncorrected.png"
    if not h_path.exists():
        result["roi_recapture"] = {"attempted": False, "reason": "missing_H"}
        return result

    if projector is None or camera is None:
        if bind_devices is None:
            result["roi_recapture"] = {
                "attempted": False,
                "reason": "devices_unavailable_for_recapture",
                "corrected_corners": n,
            }
            return result
        try:
            projector, camera = bind_devices()
        except Exception as e:
            result["roi_recapture"] = {
                "attempted": False,
                "reason": f"bind_failed:{e}",
                "corrected_corners": n,
            }
            return result

    from .charuco import load_proj_ids_json

    H = np.load(h_path)
    cor_img = cv2.imread(str(out / "captures" / "charuco_corrected.png"))
    if cor_img is None:
        # use uncorrected size as fallback for ROI estimate
        cor_img = cv2.imread(str(unc))
    cam_h, cam_w = cor_img.shape[:2]
    roi = valid_destination_roi(H, proj_w, proj_h, cam_w, cam_h)
    pattern2, proj_ids2, _board2, _, gen2 = generate_charuco_image(proj_w, proj_h, roi=roi)
    prewarp_board2 = prewarp_projfb_content(pattern2, H, proj_w, proj_h)
    pre_board_path2 = out / "prewarp_charuco_roi.png"
    cv2.imwrite(str(pre_board_path2), prewarp_board2)
    (out / "patterns").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "patterns" / "charuco_calib_roi.png"), pattern2)
    save_proj_ids(out / "proj_corner_ids_roi.json", proj_ids2)

    projector.show_black()
    time.sleep(0.3)
    projector.show_path(pre_board_path2)
    time.sleep(settle_s)
    cor_path = out / "captures" / "charuco_corrected.png"
    meta_cor = camera.capture_frame(cor_path, settle_s=0.35, flush_n=20, expect_texture=True)
    result = remeasure_saved_captures(run_dir)
    result["roi_recapture"] = {
        "attempted": True,
        "roi": list(roi),
        "generation": gen2,
        "capture_meta": meta_cor.get("metrics"),
    }
    ensure_production_homography_artifacts(run_dir)
    return result


def run_flatwall_homography_baseline(
    run_dir: Path,
    proj_w: int = 1920,
    proj_h: int = 1080,
    settle_s: float = 0.8,
    screen_id: Optional[int] = None,
    reuse_devices: Optional[tuple[Any, Any]] = None,
) -> dict:
    """Full automatic production path."""
    out = homography_dir(run_dir)
    patterns = out / "patterns"
    captures = out / "captures"

    pattern, proj_ids, board, spec, gen_meta = generate_charuco_image(proj_w, proj_h)
    pattern_path = patterns / "charuco_calib.png"
    cv2.imwrite(str(pattern_path), pattern)
    save_proj_ids(out / "proj_corner_ids.json", proj_ids)
    save_board_spec(out / "board_spec.json", spec)
    # Also copy ids next to calibration for measure-validation discovery
    cal = run_dir / "calibration" / "homography"
    cal.mkdir(parents=True, exist_ok=True)

    projector = camera = None
    owns_devices = False
    if reuse_devices:
        projector, camera = reuse_devices
    else:
        owns_devices = True
        displays = list_displays()
        selected = select_projector_display(
            displays, prefer_resolution=(proj_w, proj_h), prefer_screen_id=screen_id
        )
        if selected is None:
            return {"pass": False, "valid": False, "error": "no_projector_display", "case": "B"}
        devices = list_avfoundation_video_devices()
        cam_dev = select_iphone_camera(devices)
        if cam_dev is None:
            return {"pass": False, "valid": False, "error": "no_iphone_camera", "case": "B"}
        projector = ProjectorController(selected, displays)
        camera = CameraController(cam_dev)
        projector.start()

    result: dict[str, Any] = {
        "valid": False,
        "pass": False,
        "pipeline": "flatwall_homography",
        "board_generation": gen_meta,
        "n_proj_self_detect": len(proj_ids),
    }
    try:
        projector.show_black()
        time.sleep(0.3)
        projector.show_path(pattern_path)
        time.sleep(settle_s)
        unc_path = captures / "charuco_uncorrected.png"
        meta_unc = camera.capture_frame(unc_path, settle_s=0.35, flush_n=20, expect_texture=True)
        unc_img = cv2.imread(str(unc_path))
        cam_ids_unc, det_unc = detect_charuco(unc_img, board=board)
        pts_p, pts_c, common, match_unc = match_correspondences(proj_ids, cam_ids_unc)
        result["before_detection"] = {**det_unc, "matching": match_unc, "capture_meta": meta_unc.get("metrics")}
        if len(common) < MIN_CORNERS_CALIB:
            result["error"] = f"Insufficient matches before: {len(common)}"
            result["case"] = "B"
            return result

        H_cam_from_proj, h_stats = estimate_H_cam_from_proj(pts_p, pts_c)
        H_proj_from_cam = np.linalg.inv(H_cam_from_proj)
        np.save(out / "H_cam_from_proj.npy", H_cam_from_proj)
        np.save(out / "H_proj_from_cam.npy", H_proj_from_cam)
        np.save(cal / "H_cam_from_proj.npy", H_cam_from_proj)
        np.save(cal / "H_proj_from_cam.npy", H_proj_from_cam)
        save_proj_ids(cal / "proj_corner_ids.json", proj_ids)
        result["homography_fit"] = h_stats

        # Define independent desired target BEFORE corrected capture analysis
        desired = define_desired_target(H_cam_from_proj, proj_w, proj_h, proj_ids)
        save_desired_target(out / "desired_target.json", desired)
        save_desired_target(cal / "desired_target.json", desired)
        result["desired_target"] = {
            "proj_resolution": desired["proj_resolution"],
            "desired_aspect_ratio": desired["desired_aspect_ratio"],
            "n_ids": len(desired["desired_cam_by_id"]),
        }

        from .metrics import measure_charuco_against_projector as _meas

        exp_unc = expected_visible_from_pattern(pattern, proj_ids)
        unc_metrics = _meas(
            unc_path,
            proj_ids,
            H_cam_from_proj,
            out / "remeasure",
            "uncorrected",
            desired=desired,
            expected_visible=exp_unc,
        )
        vis_b = unc_img.copy()
        for p in pts_c.astype(int):
            cv2.circle(vis_b, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        cv2.imwrite(str(out / "before_corners.png"), vis_b)

        # Pre-warp full board + validation target (same H_desired as metrics)
        content = make_validation_target(proj_h, proj_w)
        Hd = desired["H_desired_cam_from_proj"]
        H_fwd = forward_H_proj_from_source(H_cam_from_proj, Hd)
        save_forward_prewarp_meta(out, H_fwd, proj_w, proj_h)
        save_forward_prewarp_meta(cal, H_fwd, proj_w, proj_h)
        prewarp = prewarp_from_forward_H(content, H_fwd, proj_w, proj_h)
        prewarp_board = prewarp_from_forward_H(pattern, H_fwd, proj_w, proj_h)
        pre_path = out / "prewarp_validation.png"
        pre_board_path = out / "prewarp_charuco.png"
        cv2.imwrite(str(pre_path), prewarp)
        cv2.imwrite(str(pre_board_path), prewarp_board)
        # Production copies
        prewarps = run_dir / "prewarps"
        prewarps.mkdir(exist_ok=True)
        cv2.imwrite(str(prewarps / "prewarp_best.png"), prewarp_board)
        pv = run_dir / "projected_validation"
        pv.mkdir(exist_ok=True)
        cv2.imwrite(str(pv / "prewarp_to_project.png"), prewarp_board)
        result["H_proj_from_source_current"] = {
            "path": str(out / "H_proj_from_source_current.npy"),
            "meta": str(out / "H_proj_from_source_current.json"),
        }

        # --- corrected board ---
        projector.show_black()
        time.sleep(0.3)
        projector.show_path(pre_board_path)
        time.sleep(settle_s)
        cor_path = captures / "charuco_corrected.png"
        meta_cor = camera.capture_frame(cor_path, settle_s=0.35, flush_n=20, expect_texture=True)
        cor_img = cv2.imread(str(cor_path))
        cam_ids_cor, det_cor = detect_charuco(cor_img, board=board)
        pts_p2, pts_c2, common2, match_cor = match_correspondences(proj_ids, cam_ids_cor)
        result["after_detection"] = {**det_cor, "matching": match_cor, "capture_meta": meta_cor.get("metrics")}

        # If too few corners after warp, regenerate board in valid ROI and recapture once
        if len(common2) < MIN_CORNERS_VALIDATE:
            cam_h, cam_w = cor_img.shape[:2]
            roi = valid_destination_roi(H_cam_from_proj, proj_w, proj_h, cam_w, cam_h)
            result["roi_recapture"] = {"roi": list(roi), "reason": f"only_{len(common2)}_corners"}
            pattern2, proj_ids2, board2, _, gen2 = generate_charuco_image(proj_w, proj_h, roi=roi)
            # Re-estimate is still with original H; validation board uses ROI self-ids
            prewarp_board2 = prewarp_projfb_content(pattern2, H_cam_from_proj, proj_w, proj_h, Hd)
            pre_board_path2 = out / "prewarp_charuco_roi.png"
            cv2.imwrite(str(pre_board_path2), prewarp_board2)
            cv2.imwrite(str(patterns / "charuco_calib_roi.png"), pattern2)
            save_proj_ids(out / "proj_corner_ids_roi.json", proj_ids2)
            projector.show_black()
            time.sleep(0.3)
            projector.show_path(pre_board_path2)
            time.sleep(settle_s)
            cor_path = captures / "charuco_corrected.png"
            meta_cor = camera.capture_frame(cor_path, settle_s=0.35, flush_n=20, expect_texture=True)
            cor_img = cv2.imread(str(cor_path))
            cam_ids_cor, det_cor = detect_charuco(cor_img, board=board2)
            pts_p2, pts_c2, common2, match_cor = match_correspondences(proj_ids2, cam_ids_cor)
            result["after_detection"] = {
                **det_cor,
                "matching": match_cor,
                "capture_meta": meta_cor.get("metrics"),
                "roi_board": gen2,
            }
            proj_ids_cor = proj_ids2
            desired_cor = define_desired_target(H_cam_from_proj, proj_w, proj_h, proj_ids_cor)
            save_desired_target(out / "desired_target_roi.json", desired_cor)
            projected_prewarp_path = pre_board_path2
        else:
            proj_ids_cor = proj_ids
            desired_cor = desired
            projected_prewarp_path = pre_board_path

        if len(common2) < MIN_CORNERS_VALIDATE:
            result["error"] = f"Insufficient matches after correction: {len(common2)}"
            result["case"] = "B"
            result["uncorrected_metrics"] = unc_metrics
            return result

        # Expected-visible from the pre-warped board actually projected (not full 54)
        prewarp_for_mask = cv2.imread(str(projected_prewarp_path))
        if prewarp_for_mask is None:
            prewarp_for_mask = prewarp_board
        exp_cor = expected_visible_from_pattern(prewarp_for_mask, proj_ids_cor)
        (out / "expected_visible_mask.json").write_text(
            json.dumps(exp_cor, indent=2, default=str)
        )
        cor_metrics = _meas(
            cor_path,
            proj_ids_cor,
            H_cam_from_proj,
            out / "remeasure",
            "corrected",
            desired=desired_cor,
            expected_visible=exp_cor,
        )
        comparison = compare_reproj(unc_metrics, cor_metrics)

        vis_a = cor_img.copy()
        for p in pts_c2.astype(int):
            cv2.circle(vis_a, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        cv2.imwrite(str(out / "after_corners.png"), vis_a)

        # Copy corrected capture to standard location
        cap_val = run_dir / "captured_validation"
        cap_val.mkdir(exist_ok=True)
        cv2.imwrite(str(cap_val / "validation_corrected.png"), cor_img)

        result["uncorrected_metrics"] = unc_metrics
        result["corrected_metrics"] = cor_metrics
        result["comparison"] = comparison
        result["valid"] = bool(comparison.get("valid"))
        result["pass"] = bool(comparison.get("acceptance_pass"))
        result["case"] = "A" if result["pass"] else "B"
        if not result["pass"]:
            result["error"] = comparison.get("failure_reason") or "acceptance_failed"

        metrics_out = {
            "uncorrected": unc_metrics,
            "corrected": cor_metrics,
            "comparison": comparison,
        }
        (out / "baseline_metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))
        (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))

    finally:
        if owns_devices and projector is not None:
            try:
                projector.shutdown()
            except Exception:
                pass

    (out / "baseline_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
