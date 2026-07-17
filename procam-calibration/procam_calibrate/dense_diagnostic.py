"""Dense diagnostic targets and spatial-grid measurement for region validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .charuco import apply_H


def generate_dense_diagnostic_frame(
    w: int,
    h: int,
    t: float = 0.0,
    cols: int = 6,
    rows: int = 4,
    show_seam: bool = True,
    seam_x_frac: float = 0.5,
) -> np.ndarray:
    """High-contrast diagnostic in source/desired space with numbered cells."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)
    # Outer / inner borders
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (255, 255, 255), 3)
    cv2.rectangle(img, (24, 24), (w - 25, h - 25), (180, 180, 180), 2)
    # Fine grid
    for x in range(0, w, max(20, w // 48)):
        cv2.line(img, (x, 0), (x, h - 1), (40, 40, 50), 1)
    for y in range(0, h, max(20, h // 36)):
        cv2.line(img, (0, y), (w - 1, y), (40, 40, 50), 1)
    # Numbered cells + local crosshairs + circles
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            x0, y0 = int(c * cw), int(r * ch)
            x1, y1 = int((c + 1) * cw) - 1, int((r + 1) * ch) - 1
            # Plane colour hint: leftish cyan, rightish orange
            col = (200, 200, 40) if c < cols * seam_x_frac else (40, 140, 255)
            cv2.rectangle(img, (x0 + 4, y0 + 4), (x1 - 4, y1 - 4), col, 2)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            rad = max(10, int(min(cw, ch) * 0.18))
            cv2.circle(img, (cx, cy), rad, col, 2)
            label = f"{r}{c}"
            cv2.putText(img, label, (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Extra LR cluster
            if r == rows - 1 and c == cols - 1:
                for i, (dx, dy) in enumerate([(-40, -30), (0, -35), (40, -30), (-20, 10), (20, 10)]):
                    cv2.drawMarker(img, (cx + dx, cy + dy), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
                    cv2.putText(
                        img,
                        f"LR{i}",
                        (cx + dx - 10, cy + dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1,
                    )
    # Large centre circle
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 5, (255, 255, 255), 2)
    # Travelling lines
    y_line = int((0.1 + 0.8 * ((t * 0.7) % 1.0)) * h)
    x_line = int((0.1 + 0.8 * ((t * 0.55 + 0.2) % 1.0)) * w)
    cv2.line(img, (0, y_line), (w - 1, y_line), (0, 255, 0), 2)
    cv2.line(img, (x_line, 0), (x_line, h - 1), (0, 128, 255), 2)
    # Moving square across seam
    sx = int((0.15 + 0.7 * ((t) % 1.0)) * w)
    sy = h // 2
    cv2.rectangle(img, (sx - 30, sy - 30), (sx + 30, sy + 30), (255, 0, 255), 2)
    if show_seam:
        sx0 = int(seam_x_frac * w)
        cv2.line(img, (sx0, 0), (sx0, h - 1), (0, 255, 255), 1)
    # Checker pulse in corners
    phase = int(t * 8) % 2
    for (px, py) in [(40, 40), (w - 80, 40), (40, h - 80), (w - 80, h - 80)]:
        cv2.rectangle(img, (px, py), (px + 36, py + 36), (255, 255, 255) if phase else (0, 0, 0), -1)
        cv2.rectangle(img, (px, py), (px + 36, py + 36), (255, 0, 0), 1)
    cv2.putText(img, f"t={t:.2f}", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def measure_spatial_grid(
    pts_cam: np.ndarray,
    errs: np.ndarray,
    rect_bbox: dict,
    cols: int = 6,
    rows: int = 4,
    labels: Optional[np.ndarray] = None,
) -> dict:
    """Per-cell support and residual stats inside a camera rectangle."""
    x0, y0, x1, y1 = rect_bbox["x0"], rect_bbox["y0"], rect_bbox["x1"], rect_bbox["y1"]
    cw = (x1 - x0 + 1) / cols
    ch = (y1 - y0 + 1) / rows
    cells = {}
    for r in range(rows):
        for c in range(cols):
            b = [x0 + c * cw, y0 + r * ch, x0 + (c + 1) * cw, y0 + (r + 1) * ch]
            m = (
                (pts_cam[:, 0] >= b[0])
                & (pts_cam[:, 0] < b[2])
                & (pts_cam[:, 1] >= b[1])
                & (pts_cam[:, 1] < b[3])
            )
            n = int(m.sum())
            cell = {
                "bbox": b,
                "n": n,
                "supported": n >= 3,
                "median_err": float(np.median(errs[m])) if n else None,
                "p95_err": float(np.percentile(errs[m], 95)) if n else None,
            }
            if labels is not None and n:
                if np.issubdtype(np.asarray(labels).dtype, np.number):
                    cell["n_A"] = int(np.sum(m & (labels == 0)))
                    cell["n_B"] = int(np.sum(m & (labels == 1)))
                else:
                    cell["n_A"] = int(np.sum(m & (labels == "A")))
                    cell["n_B"] = int(np.sum(m & (labels == "B")))
            # local gate
            if n < 3:
                cell["pass"] = False
                cell["fail_reason"] = "insufficient_support"
            elif cell["median_err"] > 5 or cell["p95_err"] > 12:
                cell["pass"] = False
                cell["fail_reason"] = "local_error"
            else:
                cell["pass"] = True
                cell["fail_reason"] = None
            cells[f"r{r}c{c}"] = cell
    corners = {
        "ul": [x0, y0],
        "ur": [x1, y0],
        "lr": [x1, y1],
        "ll": [x0, y1],
    }
    corner_support = {}
    for name, cxy in corners.items():
        d = np.linalg.norm(pts_cam - np.array(cxy), axis=1)
        j = int(np.argmin(d))
        corner_support[name] = {
            "nearest_dist_px": float(d[j]),
            "nearest_err_px": float(errs[j]),
            "supported": bool(d[j] <= 60.0),
        }
    n_pass = sum(1 for c in cells.values() if c["pass"])
    return {
        "cols": cols,
        "rows": rows,
        "cells": cells,
        "corner_support": corner_support,
        "n_cells_pass": n_pass,
        "n_cells": cols * rows,
        "all_supported_cells_pass": all(c["pass"] or c["fail_reason"] == "insufficient_support" for c in cells.values())
        and all(c["supported"] for c in cells.values()),
        "lower_right_pass": cells[f"r{rows-1}c{cols-1}"]["pass"],
        "lower_right_supported": cells[f"r{rows-1}c{cols-1}"]["supported"],
    }


def calib_residuals_for_run(run_dir: Path) -> dict:
    """Load calib correspondences and per-point dual-H residuals."""
    run_dir = Path(run_dir)
    agg = json.loads((run_dir / "correspondences" / "aggregated_obs.json").read_text())
    assign = json.loads((run_dir / "model_fit" / "plane_assignment_by_observation.json").read_text())
    Ha = np.load(run_dir / "model_fit" / "H_cam_from_proj_plane_A.npy")
    Hb = np.load(run_dir / "model_fit" / "H_cam_from_proj_plane_B.npy")
    keys = list(agg.keys())
    pts_p = np.array([agg[k]["proj"] for k in keys], dtype=np.float64)
    pts_c = np.array([agg[k]["cam"] for k in keys], dtype=np.float64)
    labs = np.array([0 if assign.get(k) == "A" else 1 for k in keys], dtype=np.int32)
    err_a = np.linalg.norm(apply_H(Ha, pts_p) - pts_c, axis=1)
    err_b = np.linalg.norm(apply_H(Hb, pts_p) - pts_c, axis=1)
    err = np.where(labs == 0, err_a, err_b)
    return {"keys": keys, "pts_proj": pts_p, "pts_cam": pts_c, "labels": labs, "err": err}
