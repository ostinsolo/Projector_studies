# Maximum Straight Video Region

## Goal

From the **fixed camera / audience viewpoint**, find the largest axis-aligned rectangle that:

- fits inside the usable projection mask;
- preserves the requested aspect ratio;
- does not intersect the ceiling or other excluded regions;
- may cross the wall fold (piecewise wall homographies compensate).

## Straightness definition

Straight means rectangular in **camera / viewer coordinates**:

- outer border rectangular;
- top/bottom horizontal, sides vertical;
- corners ~90°;
- circles circular; squares square;
- travelling H/V lines remain H/V across the fold;
- content continuous at the seam.

The same output may look distorted from other positions. That is expected for single-viewpoint anamorphic correction.

## Search method

`procam_calibrate.video_region.maximum_inscribed_rectangle` performs a constrained inscribed-rectangle search (integral-image fullness checks over scales and positions). It does **not** use the bounding box of the usable mask as the answer.

Claim language: **maximum found under current constraints** (local optimality probes are saved; global optimality is not always guaranteed).

## Aspect ratios and fit

Supported aspects: `16:9`, `4:3`, `1:1`, source-video aspect, custom `W:H`.

Video fit modes (playback):

| Flag | Behaviour |
|------|-----------|
| `--video-fit contain` (default) | Preserve aspect; letterbox inside the target rectangle |
| `--video-fit cover` | Fill rectangle; may crop source |
| `--video-fit stretch` | Force-fill (only when explicitly selected) |

## Alignment

| Flag | Meaning |
|------|---------|
| `--alignment camera` (default for oblique physical test) | Rectangle aligned to camera image axes |
| `--alignment architecture` | Rotate search toward dominant architectural axes |
| `--alignment custom-angle` | User angle |

## Artifacts

Under `run_dir/video_region/`:

- selected rectangle JSON / polygon;
- rejected larger candidates and reasons;
- area and `% of usable mask occupied`.
