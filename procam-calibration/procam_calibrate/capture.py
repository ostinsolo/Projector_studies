"""Capture package preparation and validation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .patterns import generate_capture_pattern_set


REQUIRED_CAPTURE_STEMS = [
    "00_black",
    "01_white",
    "02_border",
    "03_corners",
    "04_dense_grid",
    "05_cspr_grid_1",
    "06_cspr_grid_2",
    "07_cspr_grid_3",
    "08_cspr_red",
    "09_validation_uncorrected",
]


def prepare_capture_package(
    run_dir: Path,
    proj_w: int = 1920,
    proj_h: int = 1080,
) -> dict:
    patterns_dir = run_dir / "projector_patterns"
    captures_dir = run_dir / "camera_captures_raw"
    captures_dir.mkdir(parents=True, exist_ok=True)
    manifest_patterns = generate_capture_pattern_set(patterns_dir, proj_w, proj_h)

    capture_manifest = {
        "run_id": run_dir.name,
        "status": "AWAITING_CAPTURE",
        "destination_directory": str(captures_dir.resolve()),
        "projector_resolution": [proj_w, proj_h],
        "coordinate_spaces": {
            "patterns": "ProjFB",
            "captures": "CamImg",
        },
        "required_captures": [
            {
                "stem": stem,
                "pattern_file": f"{stem}.png",
                "accepted_extensions": [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"],
                "notes": "One still matching the fullscreen projected pattern; no crop/mirror/resize",
            }
            for stem in REQUIRED_CAPTURE_STEMS
        ],
        "iphone_protocol": {
            "fixed_mount": True,
            "camera_independent_of_projector": True,
            "must_see_full_projected_area": True,
            "projected_area_substantial_in_frame": True,
            "near_projector_axis_optional_not_required": True,
            "single_viewpoint_optimization": True,
            "viewpoint_independent_correction_not_claimed": True,
            "moderate_viewing_angle_recommended": True,
            "lock_focus": True,
            "lock_exposure": True,
            "lock_white_balance": True,
            "fixed_lens_and_zoom": True,
            "disable_stabilization": True,
            "no_portrait_mode": True,
            "no_mirroring": True,
            "no_auto_crop": True,
            "preserve_native_resolution": True,
            "unchanged_orientation": True,
        },
        "validate_command": f"procam-calibrate validate-capture --run-dir {run_dir}",
    }
    (run_dir / "capture_manifest.json").write_text(json.dumps(capture_manifest, indent=2))

    request = f"""# Capture request — flat-wall calibration

## Status: USER_ACTION_REQUIRED

Physical actions only. Software preparation is complete.

**Viewpoint rule:** the projector and iPhone may be placed independently.
The camera field of view must fully contain the projected area (all four boundaries).
The iPhone position is the **target observer viewpoint** for which the pre-warp is optimized.
A single CSPR-Net fit is **not** viewpoint-independent. Placing the phone near the projector is optional, not required. Prefer a moderate viewing angle for the first experiment.

### Destination directory

```
{captures_dir.resolve()}
```

Put **one still per pattern**, using the pattern stem as the filename
(example: `05_cspr_grid_1.jpg`).

### Projector resolution

`{proj_w} x {proj_h}` (fullscreen, no OS chrome)

### Numbered physical actions

1. Mount the projector on a fixed stand aimed at a flat wall. Mount the iPhone on a separate fixed stand at any convenient position that (a) sees the **entire** projected rectangle including all four edges, (b) fills a **substantial** fraction of the camera frame with that projected area, and (c) uses a **moderate** viewing angle for this first experiment. Record this pose as the target observer viewpoint in `meta.json`. Do **not** move the phone or projector afterward.
2. Lock lens selection, zoom, focus, exposure, and white balance. Disable digital stabilization and portrait mode. Do not mirror or auto-crop. Keep resolution and orientation unchanged for the whole session (calibration patterns, corrected projection, and validation).
3. Darken the room as much as practical.
4. For each pattern file in `projector_patterns/` (in numeric order):
   1. Display the PNG **fullscreen** on the projector.
   2. Wait at least 0.5 seconds.
   3. Capture one still with the iPhone from the **same fixed viewpoint**.
   4. Transfer the still into the destination directory and rename it to the pattern stem (keep the image extension).
5. Patterns to capture in order:
"""
    for i, stem in enumerate(REQUIRED_CAPTURE_STEMS, 1):
        request += f"   {i}. `{stem}.png` → `{stem}.<ext>`\n"
    request += f"""
6. Fill `meta.json` in the run directory: iPhone model, lens, zoom, orientation, camera resolution, projector–wall and camera–wall distances, approximate viewing angle, and a short description of the observer viewpoint. Note that correction is for this viewpoint only.
7. When all files are present, reply in chat (or run):

```bash
procam-calibrate validate-capture --run-dir {run_dir}
```

Do not resize, crop, rotate, or mirror the originals.
"""
    (run_dir / "capture_request.md").write_text(request)

    meta_template = {
        "session_id": run_dir.name,
        "projector_resolution": [proj_w, proj_h],
        "camera_resolution": None,
        "device_orientation": None,
        "iphone_model": None,
        "lens": None,
        "zoom_factor": 1.0,
        "target_observer_viewpoint": {
            "description": None,
            "approximate_camera_wall_distance_m": None,
            "approximate_projector_wall_distance_m": None,
            "approximate_viewing_angle_deg": None,
            "approximate_camera_projector_separation_m": None,
            "notes": "CSPR-Net pre-warp is optimized for this fixed camera pose only; not viewpoint-independent.",
        },
        "full_projected_area_visible": True,
        "projected_area_substantial_in_frame": True,
        "moderate_viewing_angle": True,
        "camera_near_projector_optional": True,
        "focus_locked": True,
        "exposure_locked": True,
        "white_balance_locked": True,
        "lens_and_zoom_fixed": True,
        "orientation_fixed": True,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta_template, indent=2))

    return {
        "run_dir": str(run_dir),
        "patterns": manifest_patterns,
        "capture_manifest": str(run_dir / "capture_manifest.json"),
        "capture_request": str(run_dir / "capture_request.md"),
        "n_patterns": len(REQUIRED_CAPTURE_STEMS),
    }


def find_capture(captures_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = captures_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def validate_capture(run_dir: Path) -> dict:
    captures_dir = run_dir / "camera_captures_raw"
    validated_dir = run_dir / "camera_captures_validated"
    validated_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    found = []
    sizes = []
    issues = []

    for stem in REQUIRED_CAPTURE_STEMS:
        path = find_capture(captures_dir, stem)
        if path is None:
            missing.append(stem)
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            issues.append(f"{stem}: unreadable")
            continue
        h, w = img.shape[:2]
        sizes.append([w, h])
        # basic quality checks
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if float(gray.std()) < 2.0:
            issues.append(f"{stem}: near-zero contrast")
        if stem == "01_white" and float(gray.mean()) < 40:
            issues.append(f"{stem}: unexpectedly dark for white pattern")
        # Continuity AE often floors black captures around mean 70–90 on walls with
        # ambient texture; only flag truly crushed-bright black fields.
        if stem == "00_black" and float(gray.mean()) > 140:
            issues.append(f"{stem}: unexpectedly bright for black pattern")
        # copy unmodified bytes into validated (same pixels; record path)
        dest = validated_dir / path.name
        dest.write_bytes(path.read_bytes())
        found.append({"stem": stem, "file": path.name, "width": w, "height": h})

    size_ok = len({tuple(s) for s in sizes}) <= 1
    if not size_ok:
        issues.append("inconsistent capture resolutions across frames")

    result = {
        "pass": len(missing) == 0 and len(issues) == 0,
        "n_required": len(REQUIRED_CAPTURE_STEMS),
        "n_found": len(found),
        "missing": missing,
        "issues": issues,
        "captures": found,
        "consistent_resolution": size_ok,
        "camera_resolution": sizes[0] if sizes else None,
    }
    (run_dir / "calibration" / "capture_validation.json").write_text(json.dumps(result, indent=2))
    return result
