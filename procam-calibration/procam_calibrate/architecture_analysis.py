"""Classical architectural scene analysis: lines, vanishing points, seam priors.

Architectural detection is a *prior* only. Projected correspondences remain
authoritative for plane geometry and final seam verification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class TargetRegionConfig:
    margin_frac: float = 0.06
    prefer_two_wall: bool = True
    prefer_seam_near_centre: bool = True
    preferred_centre: Optional[tuple[float, float]] = None  # camera px
    preferred_width_frac: float = 0.85
    preferred_height_frac: float = 0.80
    max_perspective_skew: float = 0.25


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_line_segments(gray: np.ndarray) -> list[dict]:
    """LSD when available, else Probabilistic Hough."""
    g = _normalize_contrast(gray)
    lines_out: list[dict] = []
    if hasattr(cv2, "createLineSegmentDetector"):
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            segs, widths, prec, nfa = lsd.detect(g)
            if segs is not None:
                for i, s in enumerate(segs):
                    vals = np.asarray(s, dtype=np.float64).reshape(-1)
                    if vals.size < 4:
                        continue
                    x1, y1, x2, y2 = map(float, vals[:4])
                    length = float(np.hypot(x2 - x1, y2 - y1))
                    angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
                    conf = 1.0
                    if nfa is not None and len(nfa) > i:
                        nfa_v = np.asarray(nfa[i]).reshape(-1)
                        conf = float(min(1.0, max(0.05, (-float(nfa_v[0])) / 20.0)))
                    lines_out.append(
                        {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "length": length,
                            "angle_deg": angle,
                            "confidence": conf,
                            "source": "lsd",
                        }
                    )
        except cv2.error:
            lines_out = []
    if not lines_out:
        edges = cv2.Canny(g, 50, 150, apertureSize=3)
        segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=12)
        if segs is not None:
            for s in segs[:, 0]:
                x1, y1, x2, y2 = map(float, s)
                length = float(np.hypot(x2 - x1, y2 - y1))
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
                lines_out.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "length": length,
                        "angle_deg": angle,
                        "confidence": min(1.0, length / 400.0),
                        "source": "hough",
                    }
                )
    # Keep longest / higher confidence
    lines_out.sort(key=lambda d: d["length"] * d["confidence"], reverse=True)
    return lines_out[:400]


def classify_lines(lines: list[dict], angle_tol: float = 12.0) -> dict[str, list[dict]]:
    horiz, vert, oblique = [], [], []
    for ln in lines:
        a = ln["angle_deg"]
        # normalize near-horizontal: 0 or 180, near-vertical: 90
        d_h = min(abs(a - 0), abs(a - 180), abs(a - 360))
        d_v = abs(a - 90)
        if d_h <= angle_tol:
            horiz.append(ln)
        elif d_v <= angle_tol:
            vert.append(ln)
        else:
            oblique.append(ln)
    return {"horizontal": horiz, "vertical": vert, "oblique": oblique}


def _line_homogeneous(ln: dict) -> np.ndarray:
    p1 = np.array([ln["x1"], ln["y1"], 1.0])
    p2 = np.array([ln["x2"], ln["y2"], 1.0])
    L = np.cross(p1, p2)
    n = np.linalg.norm(L[:2]) + 1e-12
    return L / n


def estimate_vanishing_points(groups: dict[str, list[dict]], img_shape: tuple[int, int]) -> dict:
    """Intersect pairs within orientation groups; pick densest cluster (robust)."""
    h, w = img_shape[:2]
    result = {}
    for name, lines in groups.items():
        if name == "oblique" or len(lines) < 2:
            continue
        pts = []
        for i in range(min(len(lines), 40)):
            for j in range(i + 1, min(len(lines), 40)):
                Li = _line_homogeneous(lines[i])
                Lj = _line_homogeneous(lines[j])
                v = np.cross(Li, Lj)
                if abs(v[2]) < 1e-9:
                    continue
                x, y = float(v[0] / v[2]), float(v[1] / v[2])
                # keep finite VPs (including far outside image)
                if np.isfinite(x) and np.isfinite(y) and abs(x) < 50 * w and abs(y) < 50 * h:
                    pts.append([x, y])
        if not pts:
            result[name] = {"point": None, "confidence": 0.0, "n_intersections": 0}
            continue
        arr = np.array(pts, dtype=np.float64)
        # median as robust VP
        med = np.median(arr, axis=0)
        dist = np.linalg.norm(arr - med, axis=1)
        inliers = arr[dist <= np.percentile(dist, 60)]
        vp = inliers.mean(0) if len(inliers) else med
        conf = float(min(1.0, len(inliers) / 20.0))
        result[name] = {
            "point": [float(vp[0]), float(vp[1])],
            "confidence": conf,
            "n_intersections": len(pts),
            "n_inliers": int(len(inliers)),
        }
    return result


def propose_seam_candidates(
    groups: dict[str, list[dict]],
    img_shape: tuple[int, int],
    prefer_near_centre: bool = True,
) -> list[dict]:
    """Candidate vertical (or near-vertical) folds / columns as seam priors."""
    h, w = img_shape[:2]
    cx = w * 0.5
    cands = []
    for ln in groups.get("vertical", []) + [
        x for x in groups.get("oblique", []) if min(abs(x["angle_deg"] - 90), abs(x["angle_deg"] - 270)) < 25
    ]:
        mx = 0.5 * (ln["x1"] + ln["x2"])
        my = 0.5 * (ln["y1"] + ln["y2"])
        # Full-height score
        y_span = abs(ln["y2"] - ln["y1"]) / max(h, 1)
        centre_bonus = 1.0 - min(1.0, abs(mx - cx) / (0.5 * w)) if prefer_near_centre else 0.5
        score = ln["length"] / max(h, 1) * 0.4 + y_span * 0.35 + centre_bonus * 0.25
        score *= ln["confidence"]
        # Reject short texture edges
        if ln["length"] < 0.25 * h:
            reason = "too_short_for_wall_fold"
            cands.append(
                {
                    "type": "vertical_line",
                    "line": ln,
                    "mid_x": mx,
                    "mid_y": my,
                    "score": float(score * 0.2),
                    "confidence": float(score * 0.2),
                    "rejected_reason": reason,
                    "status": "rejected_prior",
                }
            )
            continue
        cands.append(
            {
                "type": "vertical_line",
                "line": ln,
                "mid_x": float(mx),
                "mid_y": float(my),
                "score": float(score),
                "confidence": float(min(1.0, score)),
                "rejected_reason": None,
                "status": "candidate",
                "note": "architectural prior only — verify with projected correspondences",
            }
        )
    cands.sort(key=lambda d: d["score"], reverse=True)
    # Deduplicate by x proximity
    kept = []
    for c in cands:
        if c["status"] != "candidate":
            kept.append(c)
            continue
        if any(abs(c["mid_x"] - k["mid_x"]) < 0.04 * w for k in kept if k["status"] == "candidate"):
            c = dict(c)
            c["status"] = "rejected_prior"
            c["rejected_reason"] = "duplicate_near_stronger_candidate"
        kept.append(c)
    return kept


def propose_axis_candidates(groups: dict[str, list[dict]], vps: dict) -> dict:
    """Dominant H/V axes from longest lines + VP directions."""
    def _dom(lines: list[dict]) -> Optional[dict]:
        if not lines:
            return None
        best = max(lines, key=lambda L: L["length"] * L["confidence"])
        return {
            "angle_deg": best["angle_deg"],
            "length": best["length"],
            "confidence": best["confidence"],
            "example_line": best,
        }

    return {
        "horizontal_axis": _dom(groups.get("horizontal", [])),
        "vertical_axis": _dom(groups.get("vertical", [])),
        "vanishing_points": vps,
    }


def propose_target_region(
    img_shape: tuple[int, int],
    seam_candidates: list[dict],
    cfg: Optional[TargetRegionConfig] = None,
) -> dict:
    """Axis-aligned (camera) proposed projection quadrilateral with margins."""
    cfg = cfg or TargetRegionConfig()
    h, w = img_shape[:2]
    m = cfg.margin_frac
    cw = cfg.preferred_width_frac * (1 - 2 * m) * w
    ch = cfg.preferred_height_frac * (1 - 2 * m) * h
    if cfg.preferred_centre:
        cx, cy = cfg.preferred_centre
    else:
        cx, cy = w * 0.5, h * 0.5
    # Pull centre toward best seam if two-wall preferred
    active = [c for c in seam_candidates if c.get("status") == "candidate"]
    if cfg.prefer_two_wall and cfg.prefer_seam_near_centre and active:
        cx = 0.65 * cx + 0.35 * float(active[0]["mid_x"])
    x0 = max(m * w, cx - cw / 2)
    x1 = min(w - 1 - m * w, cx + cw / 2)
    y0 = max(m * h, cy - ch / 2)
    y1 = min(h - 1 - m * h, cy + ch / 2)
    poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {
        "type": "quadrilateral",
        "camera_polygon": poly,
        "margin_frac": cfg.margin_frac,
        "prefer_two_wall": cfg.prefer_two_wall,
        "includes_best_seam_prior": bool(active),
        "best_seam_prior_x": float(active[0]["mid_x"]) if active else None,
        "confidence": float(active[0]["confidence"]) if active else 0.4,
        "coordinate_space": "camera_pixels_distorted_as_captured",
        "note": "Proposed region prior; refined after correspondence fit",
    }


def analyse_architecture(
    image_bgr: np.ndarray,
    out_dir: Optional[Path] = None,
    cfg: Optional[TargetRegionConfig] = None,
) -> dict:
    """Full classical analysis of an empty-scene camera frame."""
    cfg = cfg or TargetRegionConfig()
    if image_bgr is None or image_bgr.size == 0:
        return {"valid": False, "failure_reason": "empty_image"}
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < 15.0:
        # still analyse but flag
        usable = False
        reject = "low_sharpness"
    else:
        usable = True
        reject = None

    lines = detect_line_segments(gray)
    groups = classify_lines(lines)
    vps = estimate_vanishing_points(groups, (h, w))
    seams = propose_seam_candidates(groups, (h, w), prefer_near_centre=cfg.prefer_seam_near_centre)
    axes = propose_axis_candidates(groups, vps)
    region = propose_target_region((h, w), seams, cfg)

    # Occlusion heuristic: large dark connected components near bottom
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    dark = (gray < thr * 0.5).astype(np.uint8) * 255
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    occlusions = []
    for i in range(1, n_lab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 0.02 * w * h:
            continue
        x, y, bw, bh = [int(stats[i, k]) for k in range(4)]
        if y + bh > 0.55 * h:  # lower half furniture-like
            occlusions.append({"bbox": [x, y, bw, bh], "area": area, "confidence": min(1.0, area / (0.1 * w * h))})

    report = {
        "valid": usable,
        "failure_reason": reject,
        "image_size": [w, h],
        "blur_laplacian_var": blur,
        "n_lines": len(lines),
        "n_horizontal": len(groups["horizontal"]),
        "n_vertical": len(groups["vertical"]),
        "n_oblique": len(groups["oblique"]),
        "lines": lines[:200],
        "groups_counts": {k: len(v) for k, v in groups.items()},
        "vanishing_points": vps,
        "seam_candidates": seams[:30],
        "axis_candidates": axes,
        "target_region": region,
        "occlusions": occlusions[:20],
        "method": "classical_lsd_or_hough",
        "authority": "prior_only",
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "detected_lines.json").write_text(json.dumps({"lines": lines[:200]}, indent=2))
        (out_dir / "vanishing_points.json").write_text(json.dumps(vps, indent=2))
        (out_dir / "seam_candidates.json").write_text(json.dumps(seams[:30], indent=2))
        (out_dir / "axis_candidates.json").write_text(json.dumps(axes, indent=2, default=str))
        (out_dir / "target_region.json").write_text(json.dumps(region, indent=2))
        (out_dir / "architecture_report.json").write_text(json.dumps(report, indent=2, default=str))
        overlay = render_architecture_overlay(image_bgr, report)
        cv2.imwrite(str(out_dir / "architectural_overlay.png"), overlay)
    return report


def render_architecture_overlay(image_bgr: np.ndarray, report: dict) -> np.ndarray:
    vis = image_bgr.copy()
    for ln in report.get("lines", [])[:120]:
        color = (80, 80, 80)
        a = ln["angle_deg"]
        if min(abs(a), abs(a - 180)) <= 12:
            color = (0, 200, 255)  # H
        elif abs(a - 90) <= 12:
            color = (0, 255, 0)  # V
        else:
            color = (180, 180, 50)
        cv2.line(
            vis,
            (int(ln["x1"]), int(ln["y1"])),
            (int(ln["x2"]), int(ln["y2"])),
            color,
            1,
            cv2.LINE_AA,
        )
    for c in report.get("seam_candidates", []):
        if c.get("status") != "candidate":
            continue
        x = int(c["mid_x"])
        cv2.line(vis, (x, 0), (x, vis.shape[0] - 1), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"seam_prior {c['confidence']:.2f}", (x + 4, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    poly = report.get("target_region", {}).get("camera_polygon")
    if poly:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (255, 128, 0), 2, cv2.LINE_AA)
    return vis


def render_synthetic_two_wall_scene(
    cam_w: int = 1280,
    cam_h: int = 720,
    seam_x_frac: float = 0.5,
    wall_angle_hint: float = 0.0,
    add_shadow_line: bool = False,
    add_column: bool = False,
    add_furniture: bool = False,
    low_contrast_seam: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, dict]:
    """Render a simple synthetic empty room for architecture-analysis tests."""
    rng = rng or np.random.default_rng(0)
    img = np.full((cam_h, cam_w, 3), 210, dtype=np.uint8)
    # Floor / ceiling bands
    cv2.rectangle(img, (0, int(0.75 * cam_h)), (cam_w, cam_h), (160, 155, 150), -1)
    cv2.rectangle(img, (0, 0), (cam_w, int(0.12 * cam_h)), (190, 195, 200), -1)
    seam_x = int(seam_x_frac * cam_w)
    # Two wall tones
    left = img[:, :seam_x].copy()
    right = img[:, seam_x:].copy()
    left = (left.astype(np.float32) * 0.95).astype(np.uint8)
    right = (right.astype(np.float32) * 1.05).clip(0, 255).astype(np.uint8)
    img[:, :seam_x] = left
    img[:, seam_x:] = right
    # Structural H/V lines
    for y in (int(0.12 * cam_h), int(0.75 * cam_h)):
        cv2.line(img, (0, y), (cam_w - 1, y), (40, 40, 40), 2)
    for x in (40, cam_w - 40):
        cv2.line(img, (x, int(0.12 * cam_h)), (x, int(0.75 * cam_h)), (50, 50, 50), 2)
    # Fold
    fold_color = (90, 90, 90) if not low_contrast_seam else (195, 195, 195)
    cv2.line(img, (seam_x, int(0.12 * cam_h)), (seam_x, int(0.75 * cam_h)), fold_color, 2 if not low_contrast_seam else 1)
    gt = {"seam_x": seam_x, "seam_x_frac": seam_x_frac, "true_fold": True}
    if add_shadow_line:
        sx = int(0.22 * cam_w)
        cv2.line(img, (sx, int(0.2 * cam_h)), (sx, int(0.7 * cam_h)), (30, 30, 30), 3)
        gt["misleading_shadow_x"] = sx
    if add_column:
        cx0 = int(0.78 * cam_w)
        cv2.rectangle(img, (cx0, int(0.12 * cam_h)), (cx0 + 35, int(0.75 * cam_h)), (120, 120, 125), -1)
        gt["column_x"] = cx0 + 17
    if add_furniture:
        cv2.rectangle(
            img,
            (int(0.1 * cam_w), int(0.55 * cam_h)),
            (int(0.35 * cam_w), int(0.85 * cam_h)),
            (80, 60, 40),
            -1,
        )
        gt["furniture"] = True
    # mild noise
    noise = rng.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, gt


def verify_seam_with_architecture_prior(
    seam_corr: dict,
    seam_candidates: list[dict],
    proj_w: int,
    cam_w: Optional[int] = None,
) -> dict:
    """Compare correspondence-derived seam with architectural prior (confidence only)."""
    mid = seam_corr.get("midpoint")
    if not mid:
        return {"consistent": False, "reason": "no_corr_seam"}
    # Map projector seam x to approximate camera x if sizes known — else compare frac
    seam_frac = float(mid[0]) / max(proj_w - 1, 1)
    active = [c for c in seam_candidates if c.get("status") == "candidate"]
    if not active:
        return {
            "consistent": True,
            "reason": "no_arch_prior_available",
            "used_prior": False,
            "corr_seam_frac": seam_frac,
            "confidence": 0.5,
        }
    best = active[0]
    if cam_w is None:
        cam_w = 1
        prior_frac = None
        # without cam width, only report scores
        return {
            "consistent": True,
            "used_prior": True,
            "corr_seam_frac": seam_frac,
            "prior_confidence": best["confidence"],
            "note": "prior present; absolute x compare skipped without cam_w",
        }
    prior_frac = float(best["mid_x"]) / max(cam_w - 1, 1)
    delta = abs(seam_frac - prior_frac)
    # Strong prior disagreement → flag, but correspondence still wins
    consistent = delta <= 0.18 or best["confidence"] < 0.35
    return {
        "consistent": bool(consistent),
        "used_prior": True,
        "corr_seam_frac": seam_frac,
        "prior_seam_frac": prior_frac,
        "frac_delta": float(delta),
        "prior_confidence": best["confidence"],
        "authority": "correspondence_seam_final",
        "reason": None if consistent else "arch_prior_disagrees_kept_correspondence_seam",
    }
