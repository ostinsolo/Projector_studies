# iPhone Capture Plan

Deterministic capture for the first geometric MVP. Prefer still photographs or USB/recorded transfer over wireless real-time streaming.

---

## Goal

Produce a fixed set of camera images of known projector patterns so structured-light decoding can build a cam↔proj correspondence map on a **flat wall**.

---

## Capture mode (first MVP)

| Mode | Use for MVP? | Why |
|------|--------------|-----|
| Still photographs | **Yes — primary** | Deterministic, easy EXIF, no sync bugs |
| USB file transfer / AirDrop | **Yes — delivery** | Reproducible batch import |
| Recorded video then extract frames | Backup | Only if still sync is impractical |
| Local USB webcam-style stream | Later | Needs sync + frame timing |
| Wireless real-time stream | **Not for first MVP** | Least reproducible |

Default workflow: project pattern → capture one still → advance pattern → repeat → import folder to Mac.

---

## Hardware setup

1. Mount projector and iPhone on **fixed** stands (no handheld).
2. Aim both at a matte **flat wall**; fill most of the camera frame with the projected rectangle.
3. Disable auto-brightness / True Tone if possible; use a fixed exposure lock (AE/AF lock) once set.
4. Turn off flash; prefer a dark room so the projection dominates.
5. Record: projector native resolution, throw distance, iPhone model, camera app used, image pixel size.

---

## Pattern sequence (structured light)

Project in this order (full screen, no OS UI chrome):

1. `black.png` — all zeros
2. `white.png` — all ones
3. Gray-code (or binary) bitplanes: for each bit `b`, project `pos` then `neg` (inverse)
4. Optional: solid red or checker for visual sanity (not required for decode)

Name files so capture filenames match projection order, e.g. `00_black`, `01_white`, `02_gray_b00_pos`, …

Bit count: enough to cover projector width/height uniquely (e.g. ≥11 bits for 1920, ≥11 for 1080 — exact set depends on implementation).

---

## Capture checklist (per pattern)

- [ ] Projector shows the intended pattern fullscreen
- [ ] Wait ≥0.5 s for display settle (avoid phone shutter during transition)
- [ ] Capture still; verify no motion blur
- [ ] Confirm frame index matches pattern index (written log)
- [ ] Do not move camera, projector, or wall between frames

---

## Import and naming

```text
session_YYYYMMDD_flatwall/
├── meta.json          # resolutions, device IDs, notes
├── patterns/          # exact PNGs that were projected
└── captures/          # iPhone stills, same stem as patterns
```

`meta.json` should include at least:

- `projector_resolution`: `[W, H]`
- `camera_image_size`: `[W, H]` (actual pixels after import)
- `iphone_model`, `capture_app`
- `surface`: `"flat_wall"`
- `pattern_type`: `"graycode"` (or similar)
- `locked_exposure`: true/false

---

## Sync strategy

| Approach | Procedure |
|----------|-----------|
| Manual (MVP) | Operator advances pattern on laptop; taps shutter on phone; logs index |
| Semi-auto (later) | Laptop shows pattern N; beep/countdown; phone continuous shutter; drop unmatched frames |
| Video (backup) | Record while patterns advance on a timer (e.g. 1 s each); extract mid-interval frames |

Do not begin with wireless streaming until still-based correspondence succeeds.

---

## Quality gates before decoding

1. White−black difference clearly shows illuminated region.
2. No saturated bloom that merges adjacent bit boundaries (reduce projector brightness if needed).
3. All pattern files and captures have identical count and matching stems.
4. Camera crop (if any) is recorded and applied consistently.

---

## Relationship to existing repos

| Repo | Capture lesson |
|------|----------------|
| CSPR-Net | Uses 3 colorful grids + red mask; fixed 3072×1728 after crop — **do not inherit those sizes**; keep iPhone native or a documented crop |
| GS-ProCams | Multi-view + COLMAP calib — **overkill for flat-wall MVP**; single fixed viewpoint is enough |

---

## Success criterion for this plan

A folder of stills that, after classical Gray-code decode, yields a dense `cam_to_proj` map with a clear valid mask covering the projected rectangle on the wall.
