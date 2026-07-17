"""Projector framebuffer pattern generation (ProjFB space)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _blank(h: int, w: int, value: int = 0) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def make_black(h: int, w: int) -> np.ndarray:
    return _blank(h, w, 0)


def make_white(h: int, w: int) -> np.ndarray:
    return _blank(h, w, 255)


def make_border(h: int, w: int, thickness: int = 40) -> np.ndarray:
    img = _blank(h, w, 0)
    img[:thickness, :] = 255
    img[-thickness:, :] = 255
    img[:, :thickness] = 255
    img[:, -thickness:] = 255
    return img


def make_corner_markers(h: int, w: int, size: int = 120) -> np.ndarray:
    img = _blank(h, w, 32)
    # Four high-contrast L markers near corners (ProjFB pixels)
    for (y0, x0, dy, dx) in [
        (40, 40, 1, 1),
        (40, w - 40 - size, 1, -1),
        (h - 40 - size, 40, -1, 1),
        (h - 40 - size, w - 40 - size, -1, -1),
    ]:
        cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 2)
        cv2.circle(img, (x0 + size // 2, y0 + size // 2), 12, (0, 0, 255), -1)
    return img


def make_dense_grid(h: int, w: int, step: int = 80) -> np.ndarray:
    img = _blank(h, w, 0)
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h - 1), (255, 255, 255), 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w - 1, y), (255, 255, 255), 1)
    # outer rectangle
    cv2.rectangle(img, (20, 20), (w - 21, h - 21), (0, 255, 0), 3)
    return img


def make_validation_target(h: int, w: int) -> np.ndarray:
    """Outer rectangle, corners, H/V lines, grid intersections — ProjFB."""
    img = _blank(h, w, 0)
    margin = 60
    cv2.rectangle(img, (margin, margin), (w - margin - 1, h - margin - 1), (255, 255, 255), 4)
    for (x, y) in [
        (margin, margin),
        (w - margin - 1, margin),
        (margin, h - margin - 1),
        (w - margin - 1, h - margin - 1),
    ]:
        cv2.drawMarker(img, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=3)
    for x in range(margin, w - margin, 100):
        cv2.line(img, (x, margin), (x, h - margin - 1), (200, 200, 200), 1)
    for y in range(margin, h - margin, 100):
        cv2.line(img, (margin, y), (w - margin - 1, y), (200, 200, 200), 1)
    # mid H/V
    cv2.line(img, (margin, h // 2), (w - margin - 1, h // 2), (0, 255, 255), 2)
    cv2.line(img, (w // 2, margin), (w // 2, h - margin - 1), (0, 255, 255), 2)
    return img


def make_color_grid(h: int, w: int, seed: int = 1) -> np.ndarray:
    """CSPR-style colorful irregular grid for training pairs (ProjFB)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # vertical bands with random colors
    x = 0
    while x < w:
        bw = int(rng.integers(40, 120))
        color = rng.integers(0, 256, size=3).tolist()
        img[:, x : x + bw] = color
        x += bw
    # overlay horizontal lines
    y = 0
    while y < h:
        bh = int(rng.integers(30, 100))
        color = rng.integers(0, 256, size=3).tolist()
        img[y : y + 4, :] = color
        y += bh
    return img


def make_solid_red(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # BGR for OpenCV write; store as BGR consistently
    return img


def generate_capture_pattern_set(out_dir: Path, proj_w: int, proj_h: int) -> dict:
    """
    Write all patterns needed for CSPR train + validation.
    Images saved as BGR PNG in ProjFB resolution.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []

    def save(name: str, img: np.ndarray, role: str):
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), img)
        items.append(
            {
                "id": name,
                "file": path.name,
                "role": role,
                "space": "ProjFB",
                "width": proj_w,
                "height": proj_h,
                "dtype": "uint8",
                "channels": 3,
                "color_order": "BGR",
            }
        )

    save("00_black", make_black(proj_h, proj_w), "mask_black")
    save("01_white", make_white(proj_h, proj_w), "mask_white")
    save("02_border", make_border(proj_h, proj_w), "geometry_border")
    save("03_corners", make_corner_markers(proj_h, proj_w), "geometry_corners")
    save("04_dense_grid", make_dense_grid(proj_h, proj_w), "geometry_grid")
    # CSPR training patterns (3 colorful + red)
    save("05_cspr_grid_1", make_color_grid(proj_h, proj_w, seed=1), "cspr_train_1")
    save("06_cspr_grid_2", make_color_grid(proj_h, proj_w, seed=2), "cspr_train_2")
    save("07_cspr_grid_3", make_color_grid(proj_h, proj_w, seed=3), "cspr_train_3")
    save("08_cspr_red", make_solid_red(proj_h, proj_w), "cspr_mask_red")
    save("09_validation_uncorrected", make_validation_target(proj_h, proj_w), "validation_uncorrected")

    manifest = {
        "projector_resolution": [proj_w, proj_h],
        "coordinate_space": "ProjFB",
        "patterns": items,
    }
    (out_dir / "pattern_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
