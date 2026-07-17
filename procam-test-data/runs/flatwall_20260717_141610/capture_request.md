# Capture request — flat-wall calibration

## Status: USER_ACTION_REQUIRED

Physical actions only. Software preparation is complete.

**Run ID (unchanged):** `flatwall_20260717_141610`  
**Patterns:** preserved — do not regenerate; content does not depend on camera placement.

**CSPR-Net viewpoint rule:** the projector and iPhone may be placed independently.
The camera field of view must fully contain the projected area (all four boundaries).
The iPhone position is the **target observer viewpoint** for which the pre-warp is optimized.
A single CSPR-Net fit is **not** viewpoint-independent. Placing the phone near the projector is optional, not required. Prefer a moderate viewing angle for the first experiment.

### Destination directory

```
/Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610/camera_captures_raw
```

Put **one still per pattern**, using the pattern stem as the filename
(example: `05_cspr_grid_1.jpg`).

### Projector resolution

`1920 x 1080` (fullscreen, no OS chrome)

### Numbered physical actions

1. Mount the projector on a fixed stand aimed at a flat wall. Mount the iPhone on a separate fixed stand at any convenient position that (a) sees the **entire** projected rectangle including all four edges, (b) fills a **substantial** fraction of the camera frame with that projected area, and (c) uses a **moderate** viewing angle for this first experiment. Record this pose as the target observer viewpoint in `meta.json`. Do **not** move the phone or projector afterward.
2. Lock lens selection, zoom, focus, exposure, and white balance. Disable digital stabilization and portrait mode. Do not mirror or auto-crop. Keep resolution and orientation unchanged for the whole session (calibration patterns, training, corrected projection, and validation).
3. Darken the room as much as practical.
4. For each pattern file in `projector_patterns/` (in numeric order):
   1. Display the PNG **fullscreen** on the projector.
   2. Wait at least 0.5 seconds.
   3. Capture one still with the iPhone from the **same fixed viewpoint**.
   4. Transfer the still into the destination directory and rename it to the pattern stem (keep the image extension).
5. Patterns to capture in order:
   1. `00_black.png` → `00_black.<ext>`
   2. `01_white.png` → `01_white.<ext>`
   3. `02_border.png` → `02_border.<ext>`
   4. `03_corners.png` → `03_corners.<ext>`
   5. `04_dense_grid.png` → `04_dense_grid.<ext>`
   6. `05_cspr_grid_1.png` → `05_cspr_grid_1.<ext>`
   7. `06_cspr_grid_2.png` → `06_cspr_grid_2.<ext>`
   8. `07_cspr_grid_3.png` → `07_cspr_grid_3.<ext>`
   9. `08_cspr_red.png` → `08_cspr_red.<ext>`
   10. `09_validation_uncorrected.png` → `09_validation_uncorrected.<ext>`

6. Fill `meta.json` in the run directory: iPhone model, lens, zoom, orientation, camera resolution, projector–wall and camera–wall distances, approximate viewing angle, and a short description of the observer viewpoint. Note that correction is for this viewpoint only.
7. When all files are present, reply in chat (or run):

```bash
procam-calibrate validate-capture --run-dir /Users/ostino/Documents/Code/Projector_studies/procam-test-data/runs/flatwall_20260717_141610
```

Do not resize, crop, rotate, or mirror the originals.
