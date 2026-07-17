# Video Playback

## Purpose

Reuse a saved two-plane + ceiling-exclusion calibration to play diagnostic or user video on the projector without re-detecting features every frame.

## Commands

Diagnostic loop:

```bash
procam-calibrate play-video \
  --calibration-run <PHYSICAL_RUN_DIRECTORY> \
  --diagnostic \
  --loop
```

User video:

```bash
procam-calibrate play-video \
  --calibration-run <PHYSICAL_RUN_DIRECTORY> \
  --input <VIDEO_FILE> \
  --loop \
  --video-fit contain \
  --alignment camera
```

## Requirements for the run directory

Must contain `video_playback/piecewise_remap.npz` (written by the oblique / two-plane pipeline after usable-mask + max-rectangle stages).

## Behaviour

- Output framebuffer: **1920×1080** (or calibrated projector size).
- Precomputed piecewise remaps for wall A and wall B.
- Ceiling / usable exclusion masks applied every frame.
- Source aspect preserved under `contain` (default).
- Does **not** modify calibration artifacts.
- Reports frames shown and dropped-frame counter (when available).
- Stops cleanly on interrupt / end of non-looped input.

## Codecs / pixel formats

OpenCV `VideoCapture` backends on macOS typically accept MP4/MOV/AVI with H.264 or MPEG. Frames are processed as BGR `uint8`. Prefer progressive 8-bit video close to the calibration source resolution (e.g. 1280×720) for predictable fit behaviour.

## Reuse without recalibration

As long as projector, camera, and walls remain fixed, reuse:

- `model_fit/H_cam_from_proj_plane_{A,B}.npy`
- `seam/` masks and seam model
- `excluded_geometry/` usable + ceiling masks
- `video_region/` maximum rectangle
- `video_playback/` remap bundle

Recalibrate if devices move or the wall–ceiling geometry changes materially.
