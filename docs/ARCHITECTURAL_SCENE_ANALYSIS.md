# Architectural Scene Analysis

**Module:** `procam_calibrate/architecture_analysis.py`  
**Authority:** prior only — projected correspondences remain final.

## What it does

On an empty-scene camera frame (projector black):

1. Contrast normalize (CLAHE)
2. Detect line segments (OpenCV LSD when available, else Probabilistic Hough)
3. Classify horizontal / vertical / oblique
4. Estimate vanishing points by robust intersection clustering within H/V groups
5. Propose seam candidates from long near-vertical lines (centre preference)
6. Propose a camera-space target quadrilateral with margins
7. Heuristic occlusion blobs (large dark lower-half components)
8. Write overlays + JSON under `run_dir/architecture/`

## What it does not do

- Does not treat a dark line as a verified wall fold
- Does not assign plane homographies
- Does not replace ChArUco / Gray-code evidence
- Does not require learned segmentation (classical path is mandatory)

## Outputs

| File | Content |
|------|---------|
| `architectural_overlay.png` | Lines, seam prior, target polygon |
| `detected_lines.json` | Segment list |
| `vanishing_points.json` | H/V VP estimates |
| `seam_candidates.json` | Scored priors + rejections |
| `target_region.json` / `proposed_target_region.json` | Camera polygon |
| `architecture_report.json` | Full report |

## Verification

`verify_seam_with_architecture_prior` compares correspondence seam fraction to the best prior. Disagreement is logged; **correspondence seam is kept**.
