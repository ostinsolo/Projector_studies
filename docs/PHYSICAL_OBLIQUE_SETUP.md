# Physical Oblique Setup

## Intent

The projector is placed **obliquely** on purpose. Coverage may include:

- two connected wall planes;
- part of the ceiling (often upper-right);
- other unusable regions.

The camera is at the **intended audience / viewing position**.

Do **not** centre or straighten the projector unless calibration becomes mathematically impossible. The software compensates for obliqueness from the camera viewpoint.

## Fixed after start

Once the run begins:

- projector pose fixed;
- camera pose fixed;
- walls unchanged.

## What “success” means

From the audience camera:

- largest practical straight rectangular video on the usable walls;
- correct aspect (default contain 16:9);
- seam continuity;
- **no meaningful video on the ceiling**.

Distortion from other viewpoints is expected.

## Checklist

1. Two wall planes with a visible fold in the projector image  
2. Ceiling may receive probe light; final video must not  
3. Camera at audience position; full projection + fold visible  
4. Extended display, 1920×1080 projector framebuffer  
5. Room dark enough for ChArUco  
6. Confirm setup, then run the physical command (see runbook)

## After calibration

```bash
procam-calibrate play-video \
  --calibration-run <RUN_DIR> \
  --diagnostic \
  --loop
```

Then play a user video with `--input` once diagnostic gates look good.
