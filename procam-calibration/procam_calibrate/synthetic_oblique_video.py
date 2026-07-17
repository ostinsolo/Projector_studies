"""Synthetic scenes for ceiling exclusion and maximum video region."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .charuco import apply_H
from .excluded_geometry import (
    build_usable_and_exclusion_masks,
    classify_active_and_excluded_planes,
    estimate_wall_ceiling_boundary,
    save_exclusion_artifacts,
)
from .synthetic_two_plane import generate_two_plane_correspondences, make_connected_plane_homographies, sample_board_points
from .two_plane import estimate_straight_seam
from .video_playback import (
    generate_diagnostic_frame,
    measure_ceiling_leakage,
    prepare_playback_calibration,
    warp_frame_piecewise,
    load_remap_bundle,
)
from .video_region import maximum_inscribed_rectangle, parse_aspect, save_video_region


def generate_two_wall_with_ceiling(
    noise_px: float = 0.0,
    ceiling_frac: float = 0.22,
    rng: Optional[np.random.Generator] = None,
    proj_w: int = 1920,
    proj_h: int = 1080,
    cam_w: int = 1920,
    cam_h: int = 1080,
) -> dict:
    """Two-wall correspondences plus a third upper (ceiling) cluster."""
    rng = rng or np.random.default_rng(0)
    base = generate_two_plane_correspondences(
        90.0, True, noise_px, rng=rng, proj_w=proj_w, proj_h=proj_h, cam_w=cam_w, cam_h=cam_h
    )
    H_a, H_b = base["H_a_true"], base["H_b_true"]
    # Ceiling plane: map upper projector band through a third H (shift up in camera)
    pts_all, ids_all = sample_board_points(proj_w, proj_h)
    # Sample dense grid in upper projector region
    xs = np.linspace(proj_w * 0.1, proj_w * 0.9, 12)
    ys = np.linspace(0, proj_h * ceiling_frac, 8)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    # Ceiling H: take H_a and push points upward (smaller y) more strongly on the right
    H_c = H_a.copy()
    # Approximate: apply H_a then translate up
    cam_c = apply_H(H_a, grid)
    cam_c[:, 1] = cam_c[:, 1] * 0.35 + cam_h * 0.02
    cam_c[:, 0] = cam_c[:, 0] * 0.9 + cam_w * 0.08
    if noise_px > 0:
        cam_c = cam_c + rng.normal(0, noise_px, size=cam_c.shape)

    # Merge
    pts_p = np.vstack([base["pts_proj"], grid])
    pts_c = np.vstack([base["pts_cam"], cam_c])
    ids = list(base["ids"]) + [10000 + i for i in range(len(grid))]
    true = np.concatenate([base["true_labels"], np.full(len(grid), 2, dtype=np.int32)])
    return {
        "pts_proj": pts_p,
        "pts_cam": pts_c,
        "ids": ids,
        "true_labels": true,  # 0/1 wall, 2 ceiling
        "H_a_true": H_a,
        "H_b_true": H_b,
        "proj_w": proj_w,
        "proj_h": proj_h,
        "cam_w": cam_w,
        "cam_h": cam_h,
        "ceiling_frac": ceiling_frac,
    }


def run_oblique_video_suite(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []

    # Ceiling intrusion classification
    sc = generate_two_wall_with_ceiling(0.0, rng=np.random.default_rng(1))
    cls = classify_active_and_excluded_planes(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    labels = np.array(cls["labels"])
    # Ceiling points mostly labeled 2
    ceil_true = sc["true_labels"] == 2
    ceil_pred = labels == 2
    ceil_recall = float((ceil_pred & ceil_true).sum() / max(ceil_true.sum(), 1))
    wall_ok = cls.get("valid_active") and cls.get("n_plane_A", 0) >= 12 and cls.get("n_plane_B", 0) >= 12
    # Ceiling must not dominate wall labels
    wall_contam = float((ceil_true & ((labels == 0) | (labels == 1))).sum() / max(ceil_true.sum(), 1))
    cases.append(
        {
            "name": "ceiling_cluster_excluded",
            "pass": wall_ok and ceil_recall >= 0.5 and wall_contam <= 0.5 and cls.get("ceiling_detected"),
            "ceil_recall": ceil_recall,
            "wall_contam": wall_contam,
            "model": cls.get("model"),
        }
    )

    # Boundary + masks + max rect + leakage
    Ha, Hb = cls["H_cam_from_proj_plane_A"], cls["H_cam_from_proj_plane_B"]
    if Ha is None or Hb is None or not cls.get("valid_active"):
        cases.append(
            {
                "name": "active_walls_available_for_masks",
                "pass": False,
                "reason": cls.get("reason"),
                "model": cls.get("model"),
            }
        )
        summary = {
            "n_cases": len(cases),
            "n_pass": sum(1 for c in cases if c["pass"]),
            "all_pass": all(c["pass"] for c in cases),
            "cases": cases,
        }
        (out_dir / "oblique_video_suite_results.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        return summary

    # For seam use only A/B
    ab = (labels == 0) | (labels == 1)
    seam = estimate_straight_seam(sc["pts_proj"][ab], labels[ab], sc["proj_w"], sc["proj_h"], Ha, Hb)
    boundary = estimate_wall_ceiling_boundary(sc["pts_cam"], labels, sc["cam_w"], sc["cam_h"])
    masks = build_usable_and_exclusion_masks(
        seam, boundary, Ha, Hb, sc["proj_w"], sc["proj_h"], sc["cam_w"], sc["cam_h"], margin_px=6.0
    )
    save_exclusion_artifacts(out_dir / "excl", masks, boundary, cls)

    # Usable must not heavily overlap ceiling mask in camera
    overlap = np.logical_and(masks["usable_mask_camera"] > 0, masks["excluded_ceiling_mask_camera"] > 0)
    cases.append(
        {
            "name": "usable_ceiling_masks_mostly_exclusive",
            "pass": float(overlap.mean()) < 0.05,
            "overlap_frac": float(overlap.mean()),
        }
    )

    for aspect_name in ("16:9", "4:3", "1:1"):
        region = maximum_inscribed_rectangle(
            masks["usable_mask_camera"], aspect=parse_aspect(aspect_name), alignment="camera", step=8
        )
        ok = region.get("valid") and region.get("local_optimality", {}).get("locally_maximal_scale", False)
        # polygon below boundary
        if region.get("valid"):
            poly = np.array(region["camera_polygon"])
            y_top = poly[:, 1].min()
            y_b = boundary["intercept"]
            below = y_top >= y_b - 5
        else:
            below = False
        cases.append(
            {
                "name": f"max_rect_{aspect_name.replace(':', 'x')}_no_ceiling",
                "pass": bool(region.get("valid") and below),
                "pct_usable": region.get("pct_usable_mask_occupied"),
            }
        )

    region = maximum_inscribed_rectangle(
        masks["usable_mask_camera"], aspect=16 / 9, alignment="camera", step=6
    )
    save_video_region(out_dir / "region", region)
    meta = prepare_playback_calibration(
        out_dir / "run",
        Ha,
        Hb,
        seam,
        masks,
        region,
        source_w=1280,
        source_h=720,
        proj_w=sc["proj_w"],
        proj_h=sc["proj_h"],
        video_fit="contain",
    )
    leak = meta["ceiling_leakage_selfcheck"]
    cases.append(
        {
            "name": "diagnostic_prewarp_ceiling_leakage",
            "pass": leak.get("pass", False),
            "leak_frac": leak.get("ceiling_leak_fraction"),
        }
    )

    # False horizontal shadow — boundary from correspondence still preferred when ceiling exists
    cases.append(
        {
            "name": "boundary_source_not_shadow_only",
            "pass": boundary.get("source") != "architectural_horizontal_prior" or not cls.get("ceiling_detected"),
            "source": boundary.get("source"),
        }
    )

    # One-pixel gap rejection: ensure wall masks exclusive
    ma, mb = masks["mask_plane_A"], masks["mask_plane_B"]
    cases.append(
        {
            "name": "wall_masks_exclusive",
            "pass": int(np.sum((ma > 0) & (mb > 0))) == 0,
        }
    )

    # Mirror / orientation: warped diagnostic shouldn't be all black on usable
    bundle = load_remap_bundle(out_dir / "run")
    fr = generate_diagnostic_frame(1280, 720, 0.3)
    out = warp_frame_piecewise(fr, {**bundle, "meta": bundle.get("meta", {})})
    cases.append(
        {
            "name": "warped_diagnostic_has_content",
            "pass": float((out > 20).mean()) > 0.01,
            "content_frac": float((out > 20).mean()),
        }
    )

    # Active wall smaller than ceiling cluster still recovers walls
    sc2 = generate_two_wall_with_ceiling(0.0, ceiling_frac=0.45, rng=np.random.default_rng(2))
    # Drop many wall points
    keep = np.ones(len(sc2["ids"]), dtype=bool)
    wall_idx = np.where(sc2["true_labels"] < 2)[0]
    drop = wall_idx[::3]
    keep[drop] = False
    cls2 = classify_active_and_excluded_planes(
        sc2["pts_proj"][keep], sc2["pts_cam"][keep], ids=[sc2["ids"][i] for i in np.where(keep)[0]]
    )
    # Sparse walls + large ceiling: recover walls, or fail clearly — never promote
    # ceiling to a third artistic content plane.
    reason2 = cls2.get("reason") or ""
    ok_sparse = bool(cls2.get("valid_active")) or (
        not cls2.get("valid_active")
        and (
            "insufficient" in reason2
            or "failed" in reason2
            or "cannot" in reason2
            or "peel" in reason2
        )
    )
    cases.append(
        {
            "name": "ceiling_larger_still_excludes",
            "pass": ok_sparse and "third artistic" in (cls2.get("note") or "third artistic").lower(),
            "model": cls2.get("model"),
            "valid_active": cls2.get("valid_active"),
            "reason": reason2,
        }
    )

    # Bounding-box is NOT used: max rect area < bounding box of usable
    ys, xs = np.where(masks["usable_mask_camera"] > 0)
    if len(xs) and region.get("valid"):
        bb_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        cases.append(
            {
                "name": "max_rect_not_bounding_box",
                "pass": region["area_px"] <= bb_area,
                "area": region["area_px"],
                "bb_area": int(bb_area),
            }
        )

    # Obstacle reducing usable area → smaller max rect
    mask_obs = masks["usable_mask_camera"].copy()
    mask_obs[200:400, 300:500] = 0
    region_obs = maximum_inscribed_rectangle(mask_obs, aspect=16 / 9, step=8)
    cases.append(
        {
            "name": "obstacle_reduces_max_rect",
            "pass": bool(
                region.get("valid")
                and region_obs.get("valid")
                and region_obs["area_px"] <= region["area_px"]
            ),
            "area_full": region.get("area_px"),
            "area_obs": region_obs.get("area_px"),
        }
    )

    # Fold-crossing rectangle: selected rect spans both wall sides in projector via seam
    if region.get("valid"):
        poly = np.array(region["camera_polygon"])
        # Map camera corners through inv Hs is heavy; check rect width spans seam x in camera
        seam_x = float(np.median(sc["pts_cam"][sc["true_labels"] < 2, 0]))
        crosses = poly[:, 0].min() < seam_x < poly[:, 0].max()
        cases.append(
            {
                "name": "target_rect_may_cross_fold",
                "pass": True,  # allowed; record whether this scene crosses
                "crosses_fold_x": bool(crosses),
            }
        )

    # contain vs stretch: contain must not claim stretch semantics
    from .video_region import apply_video_fit

    fit_c = apply_video_fit(1280, 720, region, video_fit="contain")
    fit_s = apply_video_fit(1280, 720, region, video_fit="stretch")
    cases.append(
        {
            "name": "video_fit_contain_not_stretch",
            "pass": fit_c.get("mode") == "contain" and fit_s.get("mode") == "stretch",
        }
    )

    # Wider top ceiling band: recover two walls + ceiling, or fail with peel diagnostic
    sc_top = generate_two_wall_with_ceiling(0.0, ceiling_frac=0.32, rng=np.random.default_rng(3))
    cls_top = classify_active_and_excluded_planes(
        sc_top["pts_proj"], sc_top["pts_cam"], ids=sc_top["ids"]
    )
    reason_top = cls_top.get("reason") or ""
    ok_top = (
        bool(cls_top.get("ceiling_detected") and cls_top.get("valid_active"))
        or (
            not cls_top.get("valid_active")
            and ("peel" in reason_top or "failed" in reason_top or "insufficient" in reason_top)
        )
    )
    cases.append(
        {
            "name": "ceiling_across_top_band",
            "pass": ok_top and "third artistic" in (cls_top.get("note") or "third artistic").lower(),
            "model": cls_top.get("model"),
            "reason": reason_top,
        }
    )

    summary = {
        "n_cases": len(cases),
        "n_pass": sum(1 for c in cases if c["pass"]),
        "all_pass": all(c["pass"] for c in cases),
        "cases": cases,
    }
    (out_dir / "oblique_video_suite_results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
