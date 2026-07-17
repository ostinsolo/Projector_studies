"""Maximum straight video rectangle in camera coordinates (usable-mask constrained)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


ASPECT_PRESETS = {
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
}


def parse_aspect(aspect: str | float) -> float:
    if isinstance(aspect, (int, float)):
        return float(aspect)
    aspect = str(aspect).strip()
    if aspect in ASPECT_PRESETS:
        return ASPECT_PRESETS[aspect]
    if ":" in aspect:
        a, b = aspect.split(":", 1)
        return float(a) / float(b)
    return float(aspect)


def maximum_inscribed_rectangle(
    usable_mask: np.ndarray,
    aspect: float = 16 / 9,
    alignment: str = "camera",
    angle_deg: float = 0.0,
    min_height: int = 40,
    step: int = 4,
    safety_erode_px: int = 2,
) -> dict:
    """Largest axis-aligned (or rotated) rectangle of given aspect inside mask.

    Searches scales and positions. Reports local optimality probes.
    alignment: camera | architecture | custom-angle
    """
    mask = (usable_mask > 0).astype(np.uint8)
    if safety_erode_px > 0:
        k = 2 * safety_erode_px + 1
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    h, w = mask.shape[:2]
    if mask.max() == 0:
        return {
            "valid": False,
            "failure_reason": "empty_usable_mask",
            "camera_polygon": None,
        }

    angle = 0.0 if alignment == "camera" else float(angle_deg)
    # For architecture/custom small rotations, rotate mask into search space
    if abs(angle) > 0.05:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        search_mask = rot
    else:
        search_mask = mask
        M = None

    # Integral image for O(1) rectangle sums
    integ = cv2.integral(search_mask)

    def rect_full(x0, y0, x1, y1) -> bool:
        # sum of binary mask == area
        s = int(integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0])
        return s >= (x1 - x0 + 1) * (y1 - y0 + 1)

    # Candidate heights from large to small
    max_h = h
    best = None
    candidates_tried = 0
    rejected_larger = []

    for height in range(max_h, min_height - 1, -step):
        width = int(round(height * aspect))
        if width < min_height or width >= w:
            continue
        # Scan positions
        found_any = False
        for y0 in range(0, h - height + 1, step):
            y1 = y0 + height - 1
            for x0 in range(0, w - width + 1, step):
                x1 = x0 + width - 1
                candidates_tried += 1
                if rect_full(x0, y0, x1, y1):
                    area = width * height
                    best = {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "width": width,
                        "height": height,
                        "area": area,
                        "aspect": aspect,
                    }
                    found_any = True
                    break
            if found_any:
                break
        if found_any:
            # Try one step larger to record rejection
            taller = height + step
            wider = int(round(taller * aspect))
            if taller < h and wider < w:
                rejected_larger.append(
                    {
                        "height": taller,
                        "width": wider,
                        "reason": "no_valid_position_inside_mask",
                    }
                )
            break

    if best is None:
        return {
            "valid": False,
            "failure_reason": "no_inscribed_rectangle",
            "candidates_tried": candidates_tried,
            "aspect": aspect,
        }

    # Map back from rotated search space if needed
    poly = np.array(
        [
            [best["x0"], best["y0"]],
            [best["x1"], best["y0"]],
            [best["x1"], best["y1"]],
            [best["x0"], best["y1"]],
        ],
        dtype=np.float64,
    )
    if M is not None:
        Minv = cv2.invertAffineTransform(M)
        ones = np.ones((4, 1))
        pts = np.hstack([poly, ones])
        poly = (Minv @ pts.T).T

    # Local optimality: small translations / scale-up
    local = _local_optimality_test(search_mask if M is None else mask, best, aspect, step)

    usable_area = int((usable_mask > 0).sum())
    report = {
        "valid": True,
        "method": "max_inscribed_aspect_rectangle_search",
        "claim": "maximum_found_under_current_constraints",
        "alignment": alignment,
        "angle_deg": angle,
        "aspect": aspect,
        "camera_polygon": poly.tolist(),
        "bbox": best,
        "area_px": best["area"],
        "usable_mask_area_px": usable_area,
        "pct_usable_mask_occupied": float(100.0 * best["area"] / max(usable_area, 1)),
        "candidates_tried": candidates_tried,
        "rejected_larger": rejected_larger,
        "local_optimality": local,
        "coordinate_space": "camera_pixels",
        "straightness_definition": (
            "Rectangle is axis-aligned in the selected alignment space "
            "(default: camera image axes). Piecewise wall Hs make this "
            "rectangle appear on the walls; other viewpoints may look distorted."
        ),
    }
    return report


def _local_optimality_test(mask: np.ndarray, best: dict, aspect: float, step: int) -> dict:
    integ = cv2.integral((mask > 0).astype(np.uint8))
    h, w = mask.shape[:2]

    def full(x0, y0, ww, hh) -> bool:
        x1, y1 = x0 + ww - 1, y0 + hh - 1
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return False
        s = int(integ[y1 + 1, x1 + 1] - integ[y0, x1 + 1] - integ[y1 + 1, x0] + integ[y0, x0])
        return s >= ww * hh

    ww, hh = best["width"], best["height"]
    x0, y0 = best["x0"], best["y0"]
    # translations
    for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step), (-step, -step), (step, step)]:
        if full(x0 + dx, y0 + dy, ww, hh):
            # same size elsewhere still ok — not larger
            pass
    # scale up
    hh2 = hh + step
    ww2 = int(round(hh2 * aspect))
    scale_up_ok = False
    for dx in range(-2 * step, 2 * step + 1, step):
        for dy in range(-2 * step, 2 * step + 1, step):
            if full(x0 + dx, y0 + dy, ww2, hh2):
                scale_up_ok = True
                break
        if scale_up_ok:
            break
    return {
        "larger_scale_fits": scale_up_ok,
        "locally_maximal_scale": not scale_up_ok,
        "note": "If larger_scale_fits, search step may have missed a better candidate",
    }


def apply_video_fit(
    src_w: int,
    src_h: int,
    dest_rect: dict,
    video_fit: str = "contain",
) -> dict:
    """Map source video into the desired camera rectangle per fit mode."""
    poly = np.array(dest_rect["camera_polygon"], dtype=np.float64)
    dw = float(np.linalg.norm(poly[1] - poly[0]))
    dh = float(np.linalg.norm(poly[3] - poly[0]))
    src_aspect = src_w / max(src_h, 1)
    dest_aspect = dw / max(dh, 1)
    if video_fit == "stretch":
        return {"mode": "stretch", "source_quad": [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], "dest_quad": poly.tolist()}
    if video_fit == "cover":
        # scale so source covers dest
        if src_aspect > dest_aspect:
            # source wider — crop width
            new_w = src_h * dest_aspect
            x0 = (src_w - new_w) / 2
            src_q = [[x0, 0], [x0 + new_w, 0], [x0 + new_w, src_h - 1], [x0, src_h - 1]]
        else:
            new_h = src_w / dest_aspect
            y0 = (src_h - new_h) / 2
            src_q = [[0, y0], [src_w - 1, y0], [src_w - 1, y0 + new_h], [0, y0 + new_h]]
        return {"mode": "cover", "source_quad": src_q, "dest_quad": poly.tolist()}
    # contain (default): letterbox inside dest
    if src_aspect > dest_aspect:
        # fit width
        new_h = dw / src_aspect
        y0 = (dh - new_h) / 2
        # build dest inset
        top = poly[0] + (poly[3] - poly[0]) * (y0 / dh)
        bot = poly[0] + (poly[3] - poly[0]) * ((y0 + new_h) / dh)
        right_top = poly[1] + (poly[2] - poly[1]) * (y0 / dh)
        right_bot = poly[1] + (poly[2] - poly[1]) * ((y0 + new_h) / dh)
        dest_q = [top.tolist(), right_top.tolist(), right_bot.tolist(), bot.tolist()]
    else:
        new_w = dh * src_aspect
        x0 = (dw - new_w) / 2
        left_top = poly[0] + (poly[1] - poly[0]) * (x0 / dw)
        right_top = poly[0] + (poly[1] - poly[0]) * ((x0 + new_w) / dw)
        left_bot = poly[3] + (poly[2] - poly[3]) * (x0 / dw)
        right_bot = poly[3] + (poly[2] - poly[3]) * ((x0 + new_w) / dw)
        dest_q = [left_top.tolist(), right_top.tolist(), right_bot.tolist(), left_bot.tolist()]
    return {
        "mode": "contain",
        "source_quad": [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        "dest_quad": dest_q,
    }


def save_video_region(out_dir: Path, region: dict, fit: Optional[dict] = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "maximum_video_region.json").write_text(json.dumps(region, indent=2))
    if fit is not None:
        (out_dir / "video_fit_mapping.json").write_text(json.dumps(fit, indent=2))
