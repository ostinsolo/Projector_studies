"""Planar residual diagnosis and homography-only refinement (production).

Does not use CSPR. All geometry uses shared charuco evaluation_ids.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import (
    MIN_CORNERS_VALIDATE,
    apply_H,
    compare_reproj,
    define_desired_target,
    detect_charuco,
    estimate_H_cam_from_proj,
    expected_visible_from_pattern,
    load_desired_target,
    load_proj_ids_json,
    match_correspondences,
)
from .homography_baseline import (
    forward_H_proj_from_source,
    homography_dir,
    prewarp_from_forward_H,
    prewarp_projfb_content,
    save_forward_prewarp_meta,
)
from .metrics import measure_charuco_against_projector, measure_pair_with_homography


STRENGTHS = (1.0, 0.75, 0.5, 0.25)
MAX_PHYSICAL_ITERS = 4
BURST_N = 3


def classify_residual(
    residuals: np.ndarray,
    after_translation: float,
    after_similarity: float,
    after_affine: float,
    after_homography: float,
) -> str:
    """Classify dominant residual model from remaining median magnitudes."""
    base = float(np.median(np.linalg.norm(residuals, axis=1)))
    if base < 1e-6:
        return "none"
    dx = residuals[:, 0]
    dy = residuals[:, 1]
    # Constant translation if mag of mean vector ≈ median mag and variance small
    mean_v = residuals.mean(axis=0)
    mean_mag = float(np.linalg.norm(mean_v))
    std = float(np.linalg.norm(residuals - mean_v, axis=1).std())
    if mean_mag > 0.85 * base and std < 0.35 * base and after_translation < 0.4 * base:
        return "translation"
    if after_similarity < 0.4 * base and after_similarity <= after_affine * 1.05:
        # scale/rotation
        scales = np.linalg.norm(residuals, axis=1)
        if abs(np.std(dx) - np.std(dy)) < 0.2 * max(np.std(dx), np.std(dy), 1e-6):
            return "scale" if after_similarity < after_homography * 1.1 else "similarity"
        return "rotation"
    if after_affine < 0.4 * base and after_affine <= after_homography * 1.15:
        return "affine"
    if after_homography < 0.4 * base:
        return "projective"
    if after_homography > 0.7 * base:
        return "inconsistent/noisy"
    return "lens_distortion_or_higher_order"


def diagnose_residuals(
    cam_ids: dict[int, np.ndarray],
    desired: dict,
    evaluation_ids: list[int],
    out_dir: Path,
    image_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Phase 2: residual field diagnosis on evaluation_ids only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    des = desired["desired_cam_by_id"]
    pts_c = np.array([cam_ids[i] for i in evaluation_ids], dtype=np.float64)
    pts_d = np.array([des[i] for i in evaluation_ids], dtype=np.float64)
    res = pts_c - pts_d
    mag = np.linalg.norm(res, axis=1)

    vectors = {
        str(i): {"detected": pts_c[k].tolist(), "desired": pts_d[k].tolist(),
                 "dx": float(res[k, 0]), "dy": float(res[k, 1]), "mag": float(mag[k])}
        for k, i in enumerate(evaluation_ids)
    }
    (out_dir / "residual_vectors.json").write_text(json.dumps(vectors, indent=2))

    hist_bins = np.linspace(0, float(mag.max()) + 1e-6, 21)
    counts, edges = np.histogram(mag, bins=hist_bins)
    (out_dir / "residual_magnitude_histogram.json").write_text(
        json.dumps({"bin_edges": edges.tolist(), "counts": counts.tolist()}, indent=2)
    )

    # Fit models: translation / similarity / affine / homography from desired→detected
    # translation
    t = res.mean(axis=0)
    after_t = float(np.median(np.linalg.norm(res - t, axis=1)))
    # similarity (Umeyama)
    c_d, c_c = pts_d.mean(0), pts_c.mean(0)
    X, Y = pts_d - c_d, pts_c - c_c
    U, S, Vt = np.linalg.svd(X.T @ Y / max(len(X), 1))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var = np.mean(np.sum(X**2, axis=1))
    scale = float(np.sum(S) / max(var, 1e-12))
    pred_sim = scale * (pts_d - c_d) @ R.T + c_c
    after_sim = float(np.median(np.linalg.norm(pts_c - pred_sim, axis=1)))
    # affine
    A = np.hstack([pts_d, np.ones((len(pts_d), 1))])
    M_aff, _, _, _ = np.linalg.lstsq(A, pts_c, rcond=None)
    pred_aff = A @ M_aff
    after_aff = float(np.median(np.linalg.norm(pts_c - pred_aff, axis=1)))
    # homography
    H_obs, hstats = estimate_H_cam_from_proj(pts_d, pts_c)
    pred_h = apply_H(H_obs, pts_d)
    after_h = float(np.median(np.linalg.norm(pts_c - pred_h, axis=1)))

    # Spatial grid (3x3 mean mag)
    xmin, ymin = pts_d.min(0)
    xmax, ymax = pts_d.max(0)
    grid = np.zeros((3, 3))
    grid_n = np.zeros((3, 3))
    for k in range(len(pts_d)):
        gx = min(2, int(3 * (pts_d[k, 0] - xmin) / max(xmax - xmin, 1e-6)))
        gy = min(2, int(3 * (pts_d[k, 1] - ymin) / max(ymax - ymin, 1e-6)))
        grid[gy, gx] += mag[k]
        grid_n[gy, gx] += 1
    with np.errstate(invalid="ignore"):
        spatial = np.where(grid_n > 0, grid / np.maximum(grid_n, 1), np.nan)

    cls = classify_residual(res, after_t, after_sim, after_aff, after_h)

    if image_bgr is not None:
        vis = image_bgr.copy()
        for k in range(len(pts_c)):
            c = tuple(np.round(pts_c[k]).astype(int))
            d = tuple(np.round(pts_d[k]).astype(int))
            cv2.arrowedLine(vis, d, c, (0, 140, 255), 2, tipLength=0.25)
            cv2.circle(vis, d, 3, (0, 0, 255), -1)
            cv2.circle(vis, c, 3, (0, 255, 0), -1)
        cv2.imwrite(str(out_dir / "residual_vector_overlay.png"), vis)

    report = {
        "n_ids": len(evaluation_ids),
        "evaluation_ids": evaluation_ids,
        "median_mag_px": float(np.median(mag)),
        "p95_mag_px": float(np.percentile(mag, 95)),
        "median_dx": float(np.median(res[:, 0])),
        "median_dy": float(np.median(res[:, 1])),
        "p95_dx": float(np.percentile(np.abs(res[:, 0]), 95)),
        "p95_dy": float(np.percentile(np.abs(res[:, 1]), 95)),
        "mean_translation": t.tolist(),
        "remaining_after_translation_median_px": after_t,
        "remaining_after_similarity_median_px": after_sim,
        "remaining_after_affine_median_px": after_aff,
        "remaining_after_homography_median_px": after_h,
        "similarity_scale": scale,
        "spatial_residual_grid_median_mag": np.where(np.isnan(spatial), None, spatial).tolist(),
        "dominant_residual_class": cls,
        "homography_fit_stats": hstats,
        "constant_translation_hypothesis": bool(
            abs(float(np.median(res[:, 0])) - t[0]) < 1.0
            and abs(float(np.median(res[:, 1])) - t[1]) < 1.0
            and after_t < 0.4 * float(np.median(mag))
        ),
    }
    (out_dir / "residual_diagnosis.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def interpolate_homography_strength(
    H_current: np.ndarray,
    H_full: np.ndarray,
    source_pts: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Interpolate in projector control-point space, then refit H (not coeff lerp)."""
    if strength >= 1.0 - 1e-9:
        return H_full.astype(np.float64)
    if strength <= 1e-9:
        return H_current.astype(np.float64)
    p_cur = apply_H(H_current, source_pts)
    p_full = apply_H(H_full, source_pts)
    p_blend = (1.0 - strength) * p_cur + strength * p_full
    H_new, _ = estimate_H_cam_from_proj(source_pts, p_blend)
    # estimate_H maps src→dst; here src=source_pts, dst=projector blend → H_proj_from_source
    return H_new


def candidate_H_proj_from_source(
    H_proj_from_source_current: np.ndarray,
    pts_source: np.ndarray,
    pts_cam_detected: np.ndarray,
    H_desired_cam_from_source: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Derive full-strength candidate forward pre-warp.

    H_proj_candidate = H_proj_current @ inv(H_cam_from_source_obs) @ H_desired
    Verified against warpPerspective convention in tests.
    """
    H_cam_from_source_obs, stats = estimate_H_cam_from_proj(pts_source, pts_cam_detected)
    H_cand = (
        H_proj_from_source_current
        @ np.linalg.inv(H_cam_from_source_obs)
        @ H_desired_cam_from_source
    )
    H_cand = H_cand / H_cand[2, 2]
    return H_cand.astype(np.float64), {"H_cam_from_source_observed_fit": stats}


def capture_burst_median(
    camera: Any,
    out_dir: Path,
    stem: str,
    n: int = BURST_N,
    settle_s: float = 0.35,
) -> tuple[dict[int, np.ndarray], dict]:
    """Capture n frames; return median corner position per ID + variability stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[int, np.ndarray]] = []
    metas = []
    for i in range(n):
        path = out_dir / f"{stem}_burst_{i:02d}.png"
        meta = camera.capture_frame(path, settle_s=settle_s, flush_n=20, expect_texture=True)
        img = cv2.imread(str(path))
        ids, det = detect_charuco(img)
        frames.append(ids)
        metas.append({"path": str(path), "capture_meta": meta.get("metrics"), "detection": det})

    all_ids = sorted(set().union(*[set(f) for f in frames]))
    median_ids: dict[int, np.ndarray] = {}
    jitter = {}
    for i in all_ids:
        pts = [f[i] for f in frames if i in f]
        if not pts:
            continue
        arr = np.array(pts, dtype=np.float64)
        median_ids[i] = np.median(arr, axis=0)
        if len(arr) >= 2:
            d = np.linalg.norm(arr - median_ids[i], axis=1)
            jitter[str(i)] = {
                "n_frames": len(arr),
                "max_frame_to_frame_px": float(np.max(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
                if len(arr) > 1
                else 0.0,
                "max_dev_from_median_px": float(d.max()),
                "std_px": float(d.std()),
            }
    mags = [j["max_dev_from_median_px"] for j in jitter.values()] if jitter else [0.0]
    report = {
        "burst_metas": metas,
        "per_id_jitter": jitter,
        "aggregate_median_jitter_px": float(np.median(mags)),
        "aggregate_p95_jitter_px": float(np.percentile(mags, 95)),
        "n_ids_median": len(median_ids),
    }
    (out_dir / f"{stem}_burst_variability.json").write_text(json.dumps(report, indent=2, default=str))
    return median_ids, report


def accept_candidate(prev: dict, new: dict, jitter_median: float) -> tuple[bool, str]:
    """Conservative acceptance vs previous best."""
    if not new.get("valid") or not new.get("coverage", {}).get("coverage_gate_pass", False):
        return False, "invalid_or_coverage"
    keys = ("target_median_err_px", "target_p95_err_px")
    for k in keys:
        if new.get(k) is None or prev.get(k) is None:
            return False, f"missing_{k}"
        # must improve beyond jitter
        if new[k] >= prev[k] - max(jitter_median, 0.05):
            return False, f"no_improve_{k}"
    ax_p = prev.get("horizontal_axis_deviation_deg_median")
    ax_n = new.get("horizontal_axis_deviation_deg_median")
    if isinstance(ax_p, (int, float)) and isinstance(ax_n, (int, float)):
        if ax_n > ax_p + 0.25:
            return False, "axis_regression"
    o_p = prev.get("orthogonality_error_deg")
    o_n = new.get("orthogonality_error_deg")
    if isinstance(o_p, (int, float)) and isinstance(o_n, (int, float)):
        if o_n > o_p + 0.5:
            return False, "ortho_regression"
    return True, "accepted"


def remeasure_run_v2(run_dir: Path) -> dict:
    """Recalculate metrics with evaluation_ids; write analysis_remeasure_v2/ only."""
    out = homography_dir(run_dir)
    analysis = run_dir / "analysis_remeasure_v2"
    analysis.mkdir(parents=True, exist_ok=True)
    ids_path = out / "proj_corner_ids.json"
    h_path = out / "H_cam_from_proj.npy"
    unc = out / "captures" / "charuco_uncorrected.png"
    cor = out / "captures" / "charuco_corrected.png"
    proj_ids = load_proj_ids_json(ids_path)
    H = np.load(h_path)
    proj_w, proj_h = 1920, 1080
    pat = out / "patterns" / "charuco_calib.png"
    if pat.exists():
        im = cv2.imread(str(pat))
        if im is not None:
            proj_h, proj_w = im.shape[:2]
    dt_path = out / "desired_target.json"
    if dt_path.exists():
        desired = load_desired_target(dt_path)
    else:
        desired = define_desired_target(H, proj_w, proj_h, proj_ids)
    pre = out / "prewarp_charuco.png"
    result = measure_pair_with_homography(
        unc,
        cor,
        proj_ids,
        H,
        analysis / "metrics",
        desired=desired,
        proj_w=proj_w,
        proj_h=proj_h,
        prewarp_pattern_path=pre if pre.exists() else None,
    )
    (analysis / "metrics_v2.json").write_text(json.dumps(result, indent=2, default=str))

    # Diagnose corrected residuals
    cor_img = cv2.imread(str(cor))
    cam_ids, _ = detect_charuco(cor_img)
    exp = None
    if pre.exists():
        exp = expected_visible_from_pattern(cv2.imread(str(pre)), proj_ids)
    from .charuco import build_evaluation_ids

    id_sets = build_evaluation_ids(
        cam_ids,
        desired["desired_cam_by_id"],
        proj_ids,
        exp.get("expected_visible_ids") if exp else None,
    )
    diag = diagnose_residuals(
        cam_ids,
        desired,
        id_sets["evaluation_ids"],
        analysis / "residual_diagnosis",
        image_bgr=cor_img,
    )
    result["residual_diagnosis"] = diag
    result["id_sets_corrected"] = id_sets
    (analysis / "metrics_v2.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_homography_refinement(
    run_dir: Path,
    projector: Any,
    camera: Any,
    proj_w: int,
    proj_h: int,
    settle_s: float = 0.8,
    max_iters: int = MAX_PHYSICAL_ITERS,
) -> dict:
    """Physical residual refinement loop with strengths and rollback."""
    out = homography_dir(run_dir)
    refine_root = run_dir / "homography_refinement"
    refine_root.mkdir(parents=True, exist_ok=True)

    proj_ids = load_proj_ids_json(out / "proj_corner_ids.json")
    H_cam = np.load(out / "H_cam_from_proj.npy")
    desired = load_desired_target(out / "desired_target.json")
    H_des = desired["H_desired_cam_from_proj"]
    pattern = cv2.imread(str(out / "patterns" / "charuco_calib.png"))
    H_proj_cur = forward_H_proj_from_source(H_cam, H_des)
    save_forward_prewarp_meta(out, H_proj_cur, proj_w, proj_h)

    # Baseline measure from saved corrected (or recapture)
    pre_path = out / "prewarp_charuco.png"
    exp = expected_visible_from_pattern(cv2.imread(str(pre_path)), proj_ids) if pre_path.exists() else None
    base_metrics = measure_charuco_against_projector(
        out / "captures" / "charuco_corrected.png",
        proj_ids,
        H_cam,
        refine_root / "baseline",
        "corrected_baseline",
        desired=desired,
        expected_visible=exp,
    )
    cam0, _ = detect_charuco(cv2.imread(str(out / "captures" / "charuco_corrected.png")))
    diag0 = diagnose_residuals(
        cam0,
        desired,
        base_metrics.get("evaluation_ids") or [],
        refine_root / "diagnosis",
        image_bgr=cv2.imread(str(out / "captures" / "charuco_corrected.png")),
    )

    best = {
        "metrics": base_metrics,
        "H_proj_from_source": H_proj_cur.copy(),
        "prewarp_path": str(pre_path),
        "iteration": 0,
        "strength": 0.0,
    }
    history: list[dict] = [{"iteration": 0, "kind": "baseline", "metrics": {
        k: base_metrics.get(k) for k in (
            "target_median_err_px", "target_p95_err_px",
            "horizontal_axis_deviation_deg_median", "orthogonality_error_deg", "valid"
        )
    }, "diagnosis_class": diag0.get("dominant_residual_class")}]

    abs_pass = bool(
        base_metrics.get("valid")
        and (base_metrics.get("target_median_err_px") or 1e9) <= 5.0
        and (base_metrics.get("target_p95_err_px") or 1e9) <= 12.0
        and base_metrics.get("coverage", {}).get("coverage_gate_pass")
    )
    if abs_pass:
        result = {
            "status": "ABSOLUTE_PASS",
            "absolute_gate_pass": True,
            "best": best,
            "history": history,
            "residual_class": diag0.get("dominant_residual_class"),
        }
        (refine_root / "refinement_result.json").write_text(json.dumps(result, indent=2, default=str))
        return result

    # Source control points: evaluation / all proj ids
    src_ids = sorted(proj_ids.keys())
    pts_source_all = np.array([proj_ids[i] for i in src_ids], dtype=np.float64)

    for it in range(1, max_iters + 1):
        it_dir = refine_root / f"iter_{it:02d}"
        it_dir.mkdir(exist_ok=True)
        # Detect on current best projected board (re-show)
        projector.show_black()
        time.sleep(0.25)
        projector.show_path(Path(best["prewarp_path"]))
        time.sleep(settle_s)
        med_ids, var = capture_burst_median(camera, it_dir / "burst_current", "current", n=BURST_N)
        eval_ids = sorted(
            set(med_ids)
            & set(desired["desired_cam_by_id"])
            & set(proj_ids)
            & set((exp or {}).get("expected_visible_ids") or proj_ids.keys())
        )
        if len(eval_ids) < MIN_CORNERS_VALIDATE:
            history.append({"iteration": it, "error": "insufficient_ids", "n": len(eval_ids)})
            break
        pts_s = np.array([proj_ids[i] for i in eval_ids], dtype=np.float64)
        pts_c = np.array([med_ids[i] for i in eval_ids], dtype=np.float64)
        H_full, fit_meta = candidate_H_proj_from_source(
            best["H_proj_from_source"], pts_s, pts_c, H_des
        )
        (it_dir / "candidate_fit.json").write_text(json.dumps(fit_meta, indent=2, default=str))

        accepted_this_iter = False
        for strength in STRENGTHS:
            # Control points for interpolation: board corners in source
            corners = np.array(
                [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
                dtype=np.float64,
            )
            # Also include a few board points for stability
            ctrl = np.vstack([corners, pts_source_all[:: max(1, len(pts_source_all) // 8)]])
            H_try = interpolate_homography_strength(
                best["H_proj_from_source"], H_full, ctrl, strength
            )
            pre_try = prewarp_from_forward_H(pattern, H_try, proj_w, proj_h)
            pre_try_path = it_dir / f"prewarp_s{strength:.2f}.png"
            cv2.imwrite(str(pre_try_path), pre_try)
            np.save(it_dir / f"H_proj_from_source_s{strength:.2f}.npy", H_try)

            projector.show_black()
            time.sleep(0.25)
            projector.show_path(pre_try_path)
            time.sleep(settle_s)
            med_try, var_try = capture_burst_median(
                camera, it_dir / f"burst_s{strength:.2f}", f"s{strength:.2f}", n=BURST_N
            )
            # Write synthetic median image for measure helper: overlay points on last frame
            last = list((it_dir / f"burst_s{strength:.2f}").glob("*_burst_*.png"))
            measure_path = it_dir / f"measure_s{strength:.2f}.png"
            if last:
                base_img = cv2.imread(str(sorted(last)[-1]))
            else:
                base_img = np.zeros((1080, 1920, 3), np.uint8)
            # Draw median detections and save for metric path (detector will re-detect;
            # better: save last burst frame and measure with custom IDs)
            cv2.imwrite(str(measure_path), base_img)
            exp_try = expected_visible_from_pattern(pre_try, proj_ids)
            # Measure using median IDs directly
            from .charuco import measure_against_desired_target

            m = measure_against_desired_target(
                med_try,
                desired,
                tag=f"refined_s{strength:.2f}",
                proj_ids=proj_ids,
                expected_visible_ids=exp_try.get("expected_visible_ids"),
            )
            m["coverage"] = {
                "coverage_gate_pass": bool(
                    m.get("valid")
                    and len(m.get("evaluation_ids") or [])
                    >= 0.8 * max(1, exp_try.get("expected_visible_corners", 1))
                ),
                "coverage_fraction": len(m.get("evaluation_ids") or [])
                / max(1, exp_try.get("expected_visible_corners", 1)),
            }
            # spatial spread quick check
            if m.get("evaluation_ids"):
                from .charuco import detection_coverage

                m["coverage"] = detection_coverage(
                    {i: med_try[i] for i in m["evaluation_ids"]},
                    base_img.shape[:2],
                    expected_visible=exp_try.get("expected_visible_corners", len(m["evaluation_ids"])),
                    intentionally_excluded=exp_try.get("intentionally_excluded_ids"),
                    region_pts=np.array(
                        [desired["desired_cam_by_id"][i] for i in m["evaluation_ids"]]
                    ),
                )
            (it_dir / f"metrics_s{strength:.2f}.json").write_text(
                json.dumps(m, indent=2, default=str)
            )
            ok, reason = accept_candidate(
                best["metrics"], m, var_try.get("aggregate_median_jitter_px", 0.0)
            )
            history.append(
                {
                    "iteration": it,
                    "strength": strength,
                    "accepted": ok,
                    "reason": reason,
                    "metrics": {
                        k: m.get(k)
                        for k in (
                            "target_median_err_px",
                            "target_p95_err_px",
                            "horizontal_axis_deviation_deg_median",
                            "orthogonality_error_deg",
                        )
                    },
                    "jitter_median_px": var_try.get("aggregate_median_jitter_px"),
                }
            )
            if ok:
                # Promote best
                best_pre = refine_root / "prewarp_best.png"
                cv2.imwrite(str(best_pre), pre_try)
                np.save(refine_root / "H_proj_from_source_best.npy", H_try)
                # Also update production copies
                (run_dir / "prewarps").mkdir(exist_ok=True)
                cv2.imwrite(str(run_dir / "prewarps" / "prewarp_best.png"), pre_try)
                cv2.imwrite(str(out / "prewarp_charuco.png"), pre_try)
                np.save(out / "H_proj_from_source_current.npy", H_try)
                save_forward_prewarp_meta(out, H_try, proj_w, proj_h)
                # Save a corrected capture copy
                cap = run_dir / "captured_validation"
                cap.mkdir(exist_ok=True)
                if last:
                    cv2.imwrite(str(cap / "validation_corrected.png"), cv2.imread(str(sorted(last)[-1])))
                    cv2.imwrite(str(out / "captures" / "charuco_corrected.png"), cv2.imread(str(sorted(last)[-1])))
                best = {
                    "metrics": m,
                    "H_proj_from_source": H_try,
                    "prewarp_path": str(best_pre),
                    "iteration": it,
                    "strength": strength,
                }
                accepted_this_iter = True
                exp = exp_try
                break  # next iteration from new best

        if not accepted_this_iter:
            # No strength improved — converged physical limit
            break
        # Absolute check
        if (
            (best["metrics"].get("target_median_err_px") or 1e9) <= 5.0
            and (best["metrics"].get("target_p95_err_px") or 1e9) <= 12.0
            and best["metrics"].get("coverage", {}).get("coverage_gate_pass")
        ):
            result = {
                "status": "ABSOLUTE_PASS",
                "absolute_gate_pass": True,
                "relative_gate_pass": True,
                "physical_limit_documented": False,
                "best": {
                    **best,
                    "H_proj_from_source": best["H_proj_from_source"].tolist(),
                },
                "history": history,
                "residual_class": diag0.get("dominant_residual_class"),
            }
            (refine_root / "refinement_result.json").write_text(
                json.dumps(result, indent=2, default=str)
            )
            return result

    bm = best["metrics"]
    abs_ok = bool(
        (bm.get("target_median_err_px") or 1e9) <= 5.0
        and (bm.get("target_p95_err_px") or 1e9) <= 12.0
        and bm.get("coverage", {}).get("coverage_gate_pass")
    )
    result = {
        "status": "ABSOLUTE_PASS" if abs_ok else "CONVERGED_PHYSICAL_LIMIT",
        "absolute_gate_pass": abs_ok,
        "relative_gate_pass": bool(
            (bm.get("target_median_err_px") or 0)
            < (base_metrics.get("target_median_err_px") or 1e9)
        ),
        "physical_limit_documented": not abs_ok,
        "best_metrics": {
            k: bm.get(k)
            for k in (
                "target_median_err_px",
                "target_p95_err_px",
                "horizontal_axis_deviation_deg_median",
                "orthogonality_error_deg",
            )
        },
        "baseline_metrics": {
            k: base_metrics.get(k)
            for k in ("target_median_err_px", "target_p95_err_px")
        },
        "remaining_residual_class": diag0.get("dominant_residual_class"),
        "best_iteration": best.get("iteration"),
        "best_strength": best.get("strength"),
        "best_prewarp": best.get("prewarp_path"),
        "history": history,
    }
    (refine_root / "refinement_result.json").write_text(json.dumps(result, indent=2, default=str))
    # Update run metrics.json with best (do not touch immutable originals of other runs)
    metrics_out = {
        "uncorrected": None,
        "corrected": bm,
        "comparison": compare_reproj(
            # load uncorrected from remeasure if present
            measure_charuco_against_projector(
                out / "captures" / "charuco_uncorrected.png",
                proj_ids,
                H_cam,
                refine_root / "unc_final",
                "uncorrected",
                desired=desired,
                expected_visible=expected_visible_from_pattern(
                    cv2.imread(str(out / "patterns" / "charuco_calib.png")), proj_ids
                )
                if (out / "patterns" / "charuco_calib.png").exists()
                else None,
            ),
            bm,
        ),
        "refinement": result,
    }
    # Prefer not overwriting if this is a preserved run — only write refinement metrics sidecar
    (refine_root / "metrics_after_refinement.json").write_text(
        json.dumps(metrics_out, indent=2, default=str)
    )
    if run_dir.name not in ("flatwall_20260717_141610", "flatwall_live_20260717_154312"):
        (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))
    return result
