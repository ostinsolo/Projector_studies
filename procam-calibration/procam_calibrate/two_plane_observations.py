"""Multi-frame ChArUco observation keys for two-plane calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import generate_charuco_image, detect_charuco, MIN_CORNERS_CALIB


def observation_key(pattern_index: int, corner_id: int) -> str:
    return f"{int(pattern_index)}:{int(corner_id)}"


def parse_observation_key(key: str) -> tuple[int, int]:
    a, b = key.split(":", 1)
    return int(a), int(b)


@dataclass
class PatternSpec:
    index: int
    name: str
    roi: Optional[tuple[int, int, int, int]]  # inclusive box or None for full


def default_calibration_sequence(proj_w: int, proj_h: int) -> list[PatternSpec]:
    """Shifted / biased boards to populate both planes with unique observations."""
    mx, my = int(proj_w * 0.08), int(proj_h * 0.08)
    mid_x0, mid_x1 = int(proj_w * 0.15), int(proj_w * 0.85)
    mid_y0, mid_y1 = int(proj_h * 0.15), int(proj_h * 0.85)
    return [
        PatternSpec(0, "full_frame", None),
        PatternSpec(1, "shift_right", (mx, my, proj_w - 1, proj_h - 1 - my)),
        PatternSpec(2, "shift_left", (0, my, proj_w - 1 - mx, proj_h - 1 - my)),
        PatternSpec(3, "left_biased", (0, mid_y0, mid_x1, mid_y1)),
        PatternSpec(4, "right_biased", (mid_x0, mid_y0, proj_w - 1, mid_y1)),
        PatternSpec(5, "shift_up", (mx, 0, proj_w - 1 - mx, proj_h - 1 - my)),
        PatternSpec(6, "higher_margin", (mx * 2, my * 2, proj_w - 1 - mx * 2, proj_h - 1 - my * 2)),
    ]


def generate_pattern_set(proj_w: int, proj_h: int, out_dir: Path) -> list[dict]:
    """Write projector patterns + known projector coords keyed by observation_key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = default_calibration_sequence(proj_w, proj_h)
    catalog = []
    for spec in specs:
        img, proj_ids, board, board_spec, meta = generate_charuco_image(
            proj_w, proj_h, roi=spec.roi
        )
        path = out_dir / f"{spec.index:02d}_{spec.name}.png"
        cv2.imwrite(str(path), img)
        # Known projector coords under unique observation keys
        known = {
            observation_key(spec.index, cid): [float(xy[0]), float(xy[1])]
            for cid, xy in proj_ids.items()
        }
        ids_path = out_dir / f"{spec.index:02d}_{spec.name}_proj_ids.json"
        ids_path.write_text(json.dumps({"pattern": asdict(spec), "known_proj_by_key": known}, indent=2))
        catalog.append(
            {
                "pattern_index": spec.index,
                "name": spec.name,
                "path": str(path),
                "ids_path": str(ids_path),
                "roi": list(spec.roi) if spec.roi else None,
                "n_known_corners": len(known),
                "meta": meta,
            }
        )
    (out_dir / "pattern_catalog.json").write_text(json.dumps(catalog, indent=2, default=str))
    return catalog


def extract_observations_from_capture(
    pattern_index: int,
    known_proj_by_key: dict[str, list[float]],
    capture_bgr: np.ndarray,
    min_corners: int = MIN_CORNERS_CALIB,
) -> tuple[dict[str, dict], dict]:
    """Detect camera corners; pair with known projector coords via pattern:id keys."""
    cam_ids, det = detect_charuco(capture_bgr)
    obs: dict[str, dict] = {}
    for cid, cam_xy in cam_ids.items():
        key = observation_key(pattern_index, cid)
        if key not in known_proj_by_key:
            continue
        obs[key] = {
            "pattern_index": pattern_index,
            "corner_id": int(cid),
            "proj": known_proj_by_key[key],
            "cam": [float(cam_xy[0]), float(cam_xy[1])],
        }
    meta = {
        "detection": det,
        "n_detected_raw": len(cam_ids),
        "n_matched_observations": len(obs),
        "valid": len(obs) >= min_corners,
    }
    return obs, meta


def aggregate_observations(obs_list: list[dict[str, dict]]) -> dict[str, dict]:
    """Merge observation dicts; keys remain unique across patterns."""
    out: dict[str, dict] = {}
    for block in obs_list:
        for k, v in block.items():
            if k in out:
                raise ValueError(f"Duplicate observation key {k}")
            out[k] = v
    return out


def observations_to_arrays(obs: dict[str, dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    keys = sorted(obs.keys(), key=lambda k: (parse_observation_key(k)[0], parse_observation_key(k)[1]))
    pts_p = np.array([obs[k]["proj"] for k in keys], dtype=np.float64)
    pts_c = np.array([obs[k]["cam"] for k in keys], dtype=np.float64)
    return keys, pts_p, pts_c


def canonicalize_plane_labels(
    pts_proj: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """Ensure label 0 = left/top (smaller projector centroid axis), label 1 = other."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    if len(np.unique(labels)) < 2:
        return labels
    ca = pts_proj[labels == 0].mean(0)
    cb = pts_proj[labels == 1].mean(0)
    # Prefer left (smaller x); if nearly same x, prefer top (smaller y)
    if abs(ca[0] - cb[0]) >= abs(ca[1] - cb[1]):
        if ca[0] > cb[0]:
            labels = 1 - labels
    else:
        if ca[1] > cb[1]:
            labels = 1 - labels
    return labels
