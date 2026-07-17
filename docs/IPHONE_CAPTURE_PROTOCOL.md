# iPhone Capture Protocol

Physical protocol for the **production flat-wall** pipeline (ChArUco + planar
homography). Deterministic recorded stills before any wireless live stream.

---

## Viewpoint rules (production)

The projector and camera may be placed independently provided the camera field
of view **fully captures the projection area**.

| Rule | Requirement |
|------|-------------|
| Independence | The iPhone may be placed independently of the projector |
| Coverage | Camera must see the **complete** projected area, including all four boundaries |
| Framing | Projected area should occupy a **substantial** fraction of camera resolution |
| Fixed pose | iPhone must remain fixed for all calibration and validation captures |
| Fixed optics/settings | Lens, zoom, focus, exposure, white balance, resolution, and orientation unchanged |
| Observer role | Chosen camera position is the **target observer viewpoint** for which the pre-warp is optimized |
| First experiment | Prefer a **moderate** viewing angle |
| Near-projector placement | **Optional**, not mandatory |
| Scope of correction | Do **not** claim viewpoint-independent correction from a single camera pose |

---

## Hardware configuration

| Item | Requirement |
|------|-------------|
| Projector | One unit; fixed mount; preferably wired 1920×1080 extended display |
| Camera | One iPhone; fixed mount at the chosen observer viewpoint |
| Surface | Flat wall |
| Relative pose | Independent of projector; full projected area visible |
| Lens | Fixed lens selection; fixed zoom |
| Focus | **Locked** |
| Exposure | **Locked** |
| White balance | **Locked** where possible |
| Stabilization | Digital stabilization **disabled** where possible |
| Crop / mirror | **No** automatic cropping; **no** mirroring |
| Resolution | Preserve original frame resolution |
| Motion | No movement of phone, projector, or wall during a session |

---

## Metadata to record (`meta.json`)

```json
{
  "session_id": "YYYYMMDD_HHMMSS_flatwall",
  "projector_resolution": [W, H],
  "camera_resolution": [W, H],
  "device_orientation": "landscape|portrait",
  "iphone_model": "",
  "capture_app": "",
  "lens": "",
  "zoom_factor": 1.0,
  "focus_locked": true,
  "exposure_locked": true,
  "white_balance_locked": true,
  "stabilization_disabled": true,
  "lens_and_zoom_fixed": true,
  "orientation_fixed": true,
  "full_projected_area_visible": true,
  "projected_area_substantial_in_frame": true,
  "moderate_viewing_angle": true,
  "camera_near_projector_optional": true,
  "target_observer_viewpoint": {
    "description": "",
    "approximate_camera_wall_distance_m": null,
    "approximate_projector_wall_distance_m": null,
    "approximate_viewing_angle_deg": null,
    "approximate_camera_projector_separation_m": null,
    "notes": "Homography pre-warp is optimized for this fixed camera pose only; not viewpoint-independent."
  },
  "surface": "flat_wall",
  "pattern_set_id": "charuco_DICT_4X4_50_10x7",
  "notes": ""
}
```

---

## Capture mode order

1. **Still photographs** (primary) — automatic via `procam-calibrate auto-wall`  
2. Recorded video with timed pattern advances (backup)  
3. Continuous/wireless stream — only after deterministic recorded capture succeeds  

---

## Session folder layout

```text
procam-test-data/runs/<run_id>/
  homography_baseline/patterns/
  homography_baseline/captures/
  meta.json
  capture_manifest.json
```

Canonical successful run (immutable reference):
`procam-test-data/runs/flatwall_20260717_141610`

---

## Pattern display rules

- Fullscreen projector framebuffer; no OS chrome.  
- Settle time ≥ 0.5 s before shutter.  
- Production pattern: ChArUco `DICT_4X4_50`, 10×7 squares (shared `charuco.py`).  
- Include black settle frames between displays.  

---

## Checklist (per session)

- [ ] Projector and iPhone fixed; phone pose is the recorded observer viewpoint  
- [ ] Full projected rectangle (all four edges) visible in camera  
- [ ] Projected area occupies substantial camera resolution  
- [ ] Moderate viewing angle (first experiment)  
- [ ] Focus / exposure / WB / lens / zoom / orientation locked  
- [ ] Stabilization off; no mirror / auto-crop  
- [ ] Projector and camera resolutions recorded  
- [ ] `target_observer_viewpoint` filled in `meta.json`  

---

## Relationship to research repos

| Repo | Role for flat wall |
|------|--------------------|
| CSPR-Net | Experimental non-planar research only — **not** production |
| GS-ProCams | `REUSABLE_COMPONENTS_ONLY` — **not** a flat-wall dependency |

---

## Exit for capture usability

A session is usable when another developer can, from `meta.json` + folders alone,
know the observer viewpoint and align each capture to its pattern without guessing.
