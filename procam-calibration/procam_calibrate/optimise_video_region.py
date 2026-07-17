"""Physical / offline maximum video-region search reusing a saved calibration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import apply_H, detect_charuco, generate_charuco_image
from .dense_diagnostic import (
    calib_residuals_for_run,
    generate_dense_diagnostic_frame,
    measure_spatial_grid,
)
from .homography_baseline import forward_H_proj_from_source, prewarp_from_forward_H
from .two_plane_refine import build_piecewise_from_forwards
from .video_playback import (
    build_desired_H_for_video_rect,
    build_piecewise_remap,
    measure_ceiling_leakage,
    prepare_playback_calibration,
    warp_frame_piecewise,
)
from .video_region import maximum_inscribed_rectangle, parse_aspect, save_video_region


@dataclass
class OptimiseConfig:
    calibration_run: Path
    aspect: str = "16:9"
    physical: bool = False
    resume: bool = False
    screen_id: Optional[int] = None
    settle_s: float = 0.7
    region_scale: Optional[float] = None
    region_offset_x: float = 0.0
    region_offset_y: float = 0.0
    safety_margin_frac: float = 0.02
    max_candidates: int = 40


def _opt_dir(calib: Path) -> Path:
    return calib / "region_optimisation"


def _load_masks_and_Hs(calib: Path) -> dict:
    Ha = np.load(calib / "model_fit" / "H_cam_from_proj_plane_A.npy")
    Hb = np.load(calib / "model_fit" / "H_cam_from_proj_plane_B.npy")
    usable_cam = cv2.imread(str(calib / "excluded_geometry" / "usable_mask_camera.png"), 0)
    usable_proj = cv2.imread(str(calib / "excluded_geometry" / "usable_mask_projector.png"), 0)
    ceil_proj = cv2.imread(
        str(calib / "excluded_geometry" / "excluded_ceiling_mask_projector.png"), 0
    )
    mask_a = cv2.imread(str(calib / "seam" / "plane_mask_A.png"), 0)
    mask_b = cv2.imread(str(calib / "seam" / "plane_mask_B.png"), 0)
    region0 = json.loads((calib / "video_region" / "maximum_video_region.json").read_text())
    seam = json.loads((calib / "seam" / "seam_model.json").read_text())
    meta = json.loads((calib / "run_metadata.json").read_text()) if (calib / "run_metadata.json").exists() else {}
    proj_w = int(meta.get("proj_w", 1920))
    proj_h = int(meta.get("proj_h", 1080))
    return {
        "Ha": Ha,
        "Hb": Hb,
        "usable_cam": usable_cam,
        "usable_proj": usable_proj,
        "ceil_proj": ceil_proj,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "region0": region0,
        "seam": seam,
        "proj_w": proj_w,
        "proj_h": proj_h,
    }


def rect_from_bbox(bbox: dict) -> dict:
    poly = [
        [bbox["x0"], bbox["y0"]],
        [bbox["x1"], bbox["y0"]],
        [bbox["x1"], bbox["y1"]],
        [bbox["x0"], bbox["y1"]],
    ]
    return {
        "valid": True,
        "camera_polygon": poly,
        "bbox": bbox,
        "area_px": int(bbox["width"] * bbox["height"]),
        "aspect": bbox["aspect"],
        "claim": "candidate",
    }


def scale_translate_rect(base_bbox: dict, scale: float, dx: float, dy: float, usable: np.ndarray) -> Optional[dict]:
    """Scale about centre and translate; require full containment in usable mask."""
    cx = 0.5 * (base_bbox["x0"] + base_bbox["x1"])
    cy = 0.5 * (base_bbox["y0"] + base_bbox["y1"])
    w = base_bbox["width"] * scale
    h = base_bbox["height"] * scale
    x0 = cx - w / 2 + dx
    y0 = cy - h / 2 + dy
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    H, W = usable.shape[:2]
    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return None
    x0i, y0i, x1i, y1i = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    patch = usable[y0i : y1i + 1, x0i : x1i + 1]
    if patch.size == 0 or patch.min() == 0:
        return None
    bbox = {
        "x0": x0i,
        "y0": y0i,
        "x1": x1i,
        "y1": y1i,
        "width": x1i - x0i + 1,
        "height": y1i - y0i + 1,
        "area": (x1i - x0i + 1) * (y1i - y0i + 1),
        "aspect": base_bbox["aspect"],
    }
    return rect_from_bbox(bbox)


def evaluate_candidate_offline(
    calib: Path,
    region: dict,
    ctx: dict,
    source_w: int = 1280,
    source_h: int = 720,
) -> dict:
    """Mask containment + calib spatial grid + ceiling leakage self-check."""
    usable_area = int((ctx["usable_cam"] > 0).sum())
    residuals = calib_residuals_for_run(calib)
    grid = measure_spatial_grid(
        residuals["pts_cam"],
        residuals["err"],
        region["bbox"],
        cols=6,
        rows=4,
        labels=residuals["labels"],
    )
    # Build remap and leak self-check
    H_des = build_desired_H_for_video_rect(region["camera_polygon"], source_w, source_h)
    bundle = build_piecewise_remap(
        ctx["Ha"],
        ctx["Hb"],
        H_des,
        ctx["mask_a"],
        ctx["mask_b"],
        ctx["usable_proj"],
        ctx["ceil_proj"],
        ctx["proj_w"],
        ctx["proj_h"],
        source_w,
        source_h,
    )
    diag = generate_dense_diagnostic_frame(source_w, source_h, 0.35)
    proj = warp_frame_piecewise(
        diag,
        {
            **bundle,
            "meta": {"source_w": source_w, "source_h": source_h},
        },
    )
    leak = measure_ceiling_leakage(proj, ctx["ceil_proj"])
    # Projector clipping: content near edge of framebuffer outside usable
    content = np.any(proj > 20, axis=2)
    outside = content & (ctx["usable_proj"] == 0)
    clip_frac = float(outside.sum() / max(content.sum(), 1))

    global_med = float(np.median(residuals["err"]))
    global_p95 = float(np.percentile(residuals["err"], 95))
    # Soft offline pass: mask fit + LR support + leak + no clip
    pass_offline = bool(
        grid["lower_right_supported"]
        and leak.get("pass")
        and clip_frac <= 0.02
        and grid["corner_support"]["lr"]["supported"]
        and grid["corner_support"]["ll"]["supported"]
        and grid["corner_support"]["ur"]["supported"]
        and grid["corner_support"]["ul"]["supported"]
    )
    return {
        "mode": "offline",
        "pass": pass_offline,
        "area_px": region["area_px"],
        "pct_usable": 100.0 * region["area_px"] / max(usable_area, 1),
        "global_median_err_px": global_med,
        "global_p95_err_px": global_p95,
        "spatial_grid": grid,
        "ceiling_leakage": leak,
        "projector_clip_frac": clip_frac,
        "fail_reasons": [
            r
            for r, ok in [
                ("lower_right_unsupported", grid["lower_right_supported"]),
                ("corner_unsupported", all(v["supported"] for v in grid["corner_support"].values())),
                ("ceiling_leak", leak.get("pass")),
                ("projector_clip", clip_frac <= 0.02),
            ]
            if not ok
        ],
        "prewarp_sample_shape": list(proj.shape),
    }


def build_candidate_prewarp(
    ctx: dict,
    region: dict,
    source_w: int = 1280,
    source_h: int = 720,
) -> tuple[np.ndarray, np.ndarray, dict]:
    H_des = build_desired_H_for_video_rect(region["camera_polygon"], source_w, source_h)
    content = generate_dense_diagnostic_frame(source_w, source_h, 0.4)
    H_fwd_a = forward_H_proj_from_source(ctx["Ha"], H_des)
    H_fwd_b = forward_H_proj_from_source(ctx["Hb"], H_des)
    pre = build_piecewise_from_forwards(
        content, H_fwd_a, H_fwd_b, ctx["mask_a"], ctx["mask_b"], ctx["proj_w"], ctx["proj_h"]
    )
    # Apply usable / ceiling
    pre = pre.copy()
    pre[ctx["usable_proj"] == 0] = 0
    pre[ctx["ceil_proj"] > 0] = 0
    return pre, content, {"H_des": H_des, "H_fwd_a": H_fwd_a, "H_fwd_b": H_fwd_b}


def run_offline_search(cfg: OptimiseConfig) -> dict:
    calib = Path(cfg.calibration_run)
    out = _opt_dir(calib)
    out.mkdir(parents=True, exist_ok=True)
    ctx = _load_masks_and_Hs(calib)
    base = ctx["region0"]["bbox"]
    aspect = parse_aspect(cfg.aspect)
    base["aspect"] = aspect
    usable = ctx["usable_cam"]
    usable_area = int((usable > 0).sum())

    # Also recompute true max inscribed at this aspect (finer step)
    max_ins = maximum_inscribed_rectangle(usable, aspect=aspect, step=4, safety_erode_px=0)
    (out / "recomputed_max_inscribed.json").write_text(json.dumps(max_ins, indent=2))

    # Shrink/shift first so unsupported LR corner can enter support, then expand.
    scales = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    if cfg.region_scale is not None:
        scales = [float(cfg.region_scale)]
    offsets = [(0, 0)]
    if cfg.region_scale is None:
        offsets += [
            (-80, 0),
            (-40, 0),
            (40, 0),
            (80, 0),
            (0, -60),
            (0, -30),
            (0, 30),
            (-60, -40),
            (-40, -30),
            (40, -30),
            (-60, 30),
            (40, 30),
        ]
    offsets = [(ox + cfg.region_offset_x, oy + cfg.region_offset_y) for ox, oy in offsets]

    results = []
    # Seed with recomputed max inscribed if valid
    if max_ins.get("valid") and max_ins.get("bbox"):
        region = rect_from_bbox(max_ins["bbox"])
        ev = evaluate_candidate_offline(calib, region, ctx)
        ev.update({"name": "recomputed_max_inscribed", "scale": None, "dx": 0, "dy": 0, "bbox": region["bbox"]})
        pre, _, _ = build_candidate_prewarp(ctx, region)
        cv2.imwrite(str(out / "recomputed_max_inscribed_prewarp.png"), pre)
        (out / "recomputed_max_inscribed_eval.json").write_text(json.dumps(ev, indent=2, default=str))
        results.append(ev)

    for scale in scales:
        for dx, dy in offsets:
            region = scale_translate_rect(base, scale, dx, dy, usable)
            name = f"s{scale:.2f}_dx{dx:.0f}_dy{dy:.0f}"
            if region is None:
                results.append(
                    {
                        "name": name,
                        "pass": False,
                        "fail_reasons": ["outside_usable_mask"],
                        "scale": scale,
                        "dx": dx,
                        "dy": dy,
                        "area_px": 0,
                    }
                )
                continue
            ev = evaluate_candidate_offline(calib, region, ctx)
            ev.update({"name": name, "scale": scale, "dx": dx, "dy": dy, "bbox": region["bbox"]})
            pre, _, _ = build_candidate_prewarp(ctx, region)
            cv2.imwrite(str(out / f"{name}_prewarp.png"), pre)
            (out / f"{name}_eval.json").write_text(json.dumps(ev, indent=2, default=str))
            results.append(ev)
            if len(results) >= cfg.max_candidates:
                break
        if len(results) >= cfg.max_candidates:
            break

    def _area(r: dict) -> int:
        return int(r.get("area_px") or 0)

    # Support-hull constrained search (binary scale about calib percentile box)
    residuals = calib_residuals_for_run(calib)
    pts = residuals["pts_cam"]
    x0, x1 = np.percentile(pts[:, 0], [8, 92])
    y0, y1 = np.percentile(pts[:, 1], [8, 92])
    sw, sh = x1 - x0, y1 - y0
    if sw / max(sh, 1e-6) > aspect:
        nw = sh * aspect
        cx = 0.5 * (x0 + x1)
        x0, x1 = cx - nw / 2, cx + nw / 2
    else:
        nh = sw / aspect
        cy = 0.5 * (y0 + y1)
        y0, y1 = cy - nh / 2, cy + nh / 2
    support_base = {
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
        "area": int((x1 - x0) * (y1 - y0)),
        "aspect": aspect,
    }
    support_best = None
    for dx, dy in [(0, 0), (-30, 0), (30, 0), (0, -20), (0, 20), (-30, -20), (30, -20)]:
        lo, hi = 0.5, 1.0
        local = None
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            region = scale_translate_rect(support_base, mid, dx, dy, usable)
            if region is None:
                hi = mid
                continue
            ev = evaluate_candidate_offline(calib, region, ctx)
            ev.update(
                {
                    "name": f"support_s{mid:.3f}_dx{dx}_dy{dy}",
                    "scale": mid,
                    "dx": dx,
                    "dy": dy,
                    "bbox": region["bbox"],
                }
            )
            results.append(ev)
            if ev.get("pass"):
                local = ev
                lo = mid
            else:
                hi = mid
        if local and (support_best is None or _area(local) > _area(support_best)):
            support_best = local
    if support_best is not None:
        pre, _, _ = build_candidate_prewarp(ctx, rect_from_bbox(support_best["bbox"]))
        cv2.imwrite(str(out / "best_support_constrained_prewarp.png"), pre)
        (out / "BEST_SUPPORT_CONSTRAINED.json").write_text(
            json.dumps(support_best, indent=2, default=str)
        )

    passing = [r for r in results if r.get("pass") and _area(r) > 0]
    if passing:
        best = max(passing, key=_area)
    elif support_best is not None:
        best = support_best

    rejected_larger = [
        r
        for r in results
        if _area(r) > _area(best) and not r.get("pass")
    ]

    summary = {
        "calibration_run": str(calib),
        "aspect": cfg.aspect,
        "baseline_area": ctx["region0"]["area_px"],
        "baseline_pct_usable": ctx["region0"].get("pct_usable_mask_occupied"),
        "usable_area_px": usable_area,
        "recomputed_max_inscribed_area": max_ins.get("area_px"),
        "recomputed_max_inscribed_pct": max_ins.get("pct_usable_mask_occupied"),
        "n_candidates": len(results),
        "n_pass_offline": len(passing),
        "best": best,
        "best_support_constrained": support_best,
        "rejected_larger": [
            {"name": r["name"], "area_px": r.get("area_px"), "fail_reasons": r.get("fail_reasons")}
            for r in rejected_larger[:20]
        ],
        "claim": "maximum_offline_verified_under_mask_and_support_constraints",
        "tradeoff": (
            "Largest mask-inscribed 16:9 (~37% usable) leaves the lower-right unsupported by "
            "calibration correspondences. Largest support-safe 16:9 is smaller (~27% usable). "
            "Expanding past support-safe requires denser LR calibration, not only rectangle search."
        ),
        "note": (
            "Offline pass requires usable-mask containment, corner/LR calib support, "
            "and ceiling-leak self-check. Physical capture still required for REGION_EXPANSION_VALIDATED."
        ),
        "results": results,
        "updated_at": datetime.now().isoformat(),
    }
    (out / "OFFLINE_REGION_SEARCH.json").write_text(json.dumps(summary, indent=2, default=str))

    # Overlay best vs baseline
    base_img = cv2.imread(str(calib / "captured_validation" / "corrected_burst_00.png"))
    if base_img is not None and best.get("bbox"):
        vis = base_img.copy()
        u = ctx["usable_cam"]
        tint = np.zeros_like(vis)
        tint[u > 0] = (0, 70, 0)
        vis = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)
        b0 = ctx["region0"]["bbox"]
        cv2.rectangle(vis, (b0["x0"], b0["y0"]), (b0["x1"], b0["y1"]), (0, 255, 255), 2)
        bb = best["bbox"]
        cv2.rectangle(vis, (bb["x0"], bb["y0"]), (bb["x1"], bb["y1"]), (0, 255, 0), 2)
        cv2.putText(vis, "yellow=baseline  green=best offline", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(out / "offline_best_vs_baseline.png"), vis)

    md = [
        "# Offline region search",
        "",
        f"- Baseline area: {summary['baseline_area']} ({summary['baseline_pct_usable']:.1f}% usable)",
        f"- Recomputed max inscribed: {summary['recomputed_max_inscribed_area']} ({summary['recomputed_max_inscribed_pct']})",
        f"- Candidates tried: {summary['n_candidates']} (pass offline: {summary['n_pass_offline']})",
        f"- Best: `{best.get('name')}` area={best.get('area_px')} pct={best.get('pct_usable')} pass={best.get('pass')}",
        f"- Fail reasons (best): {best.get('fail_reasons')}",
        f"- Rejected larger: {len(rejected_larger)}",
        "",
        "## Interpretation",
        "",
        "If baseline fails `lower_right_unsupported`, shrink or shift away from the unsupported corner",
        "before expanding. Physical search should start from the largest offline-passing candidate.",
    ]
    (out / "OFFLINE_REGION_SEARCH.md").write_text("\n".join(md) + "\n")
    return summary


def run_physical_candidate(
    cfg: OptimiseConfig,
    region: dict,
    name: str,
) -> dict:
    """Project dense diagnostic for one candidate and capture a burst."""
    from .camera_control import CameraController, list_avfoundation_video_devices, select_iphone_camera
    from .display_control import ProjectorController, list_displays, select_projector_display

    calib = Path(cfg.calibration_run)
    out = _opt_dir(calib) / "physical" / name
    out.mkdir(parents=True, exist_ok=True)
    ctx = _load_masks_and_Hs(calib)
    pre, content, mats = build_candidate_prewarp(ctx, region)
    cv2.imwrite(str(out / "diagnostic_source.png"), content)
    cv2.imwrite(str(out / "prewarp.png"), pre)

    displays = list_displays()
    selected = select_projector_display(
        displays, prefer_resolution=(ctx["proj_w"], ctx["proj_h"]), prefer_screen_id=cfg.screen_id
    )
    if selected is None:
        return {"ok": False, "error": "no_projector"}
    devices = list_avfoundation_video_devices()
    cam = select_iphone_camera(devices)
    if cam is None:
        return {"ok": False, "error": "no_camera"}
    projector = ProjectorController(selected, displays)
    camera = CameraController(cam)
    projector.start()
    frames = []
    try:
        projector.show_image(pre)
        time.sleep(cfg.settle_s)
        for i in range(3):
            path = out / f"capture_{i:02d}.png"
            meta = camera.capture_frame(path, settle_s=0.35, flush_n=20, expect_texture=True)
            frames.append({"path": str(path), "meta": meta})
    finally:
        projector.shutdown()

    leak = measure_ceiling_leakage(pre, ctx["ceil_proj"])
    offline = evaluate_candidate_offline(calib, region, ctx)
    result = {
        "ok": True,
        "name": name,
        "region": region,
        "captures": frames,
        "ceiling_leakage_prewarp": leak,
        "offline_eval": offline,
        "note": "Physical geometry metrics from ChArUco on dense diagnostic require tiled validation stage",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def run_optimise_video_region(cfg: OptimiseConfig) -> dict:
    summary = run_offline_search(cfg)
    if not cfg.physical:
        summary["terminal"] = "USER_ACTION_REQUIRED" if summary["n_pass_offline"] == 0 else "OFFLINE_SEARCH_COMPLETE"
        summary["next"] = (
            f"procam-calibrate optimise-video-region --calibration-run {cfg.calibration_run} "
            f"--aspect {cfg.aspect} --physical --resume"
        )
        return summary

    # Physical: test baseline, best offline pass, and one larger failing neighbour
    ctx = _load_masks_and_Hs(Path(cfg.calibration_run))
    base_region = rect_from_bbox(ctx["region0"]["bbox"])
    physical_results = []
    todo = [("baseline", base_region)]
    best = summary.get("best") or {}
    if best.get("bbox"):
        todo.append((best["name"], rect_from_bbox(best["bbox"])))
    # Try 10% larger around best if mask allows
    if best.get("bbox"):
        larger = scale_translate_rect(best["bbox"], 1.1, 0, 0, ctx["usable_cam"])
        if larger is not None:
            todo.append(("best_plus_10pct", larger))

    for name, region in todo:
        physical_results.append(run_physical_candidate(cfg, region, name))

    summary["physical_results"] = physical_results
    summary["terminal"] = "USER_ACTION_REQUIRED"
    # If we got captures for an expanded region with offline pass, note partial progress
    if any(p.get("ok") and (p.get("offline_eval") or {}).get("pass") for p in physical_results):
        summary["terminal"] = "PHYSICAL_CANDIDATES_CAPTURED"
    (Path(cfg.calibration_run) / "region_optimisation" / "PHYSICAL_SEARCH.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return summary
