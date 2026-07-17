# Two-Plane Synthetic Results

## Status

**PASS** — 14/14 cases

Artifacts: `procam-test-data/runs/two_plane_synthetic_20260717/suite/`

## Cases

| Case | Pass | Notes |
|------|------|-------|
| `one_plane_noise_0.0` | PASS | model=one_plane; acc=None; seam_med=None |
| `one_plane_noise_0.5` | PASS | model=one_plane; acc=None; seam_med=None |
| `two_plane_a30_v_n0.0` | PASS | model=two_plane; acc=1.0; seam_med=0.11468689216997895 |
| `two_plane_a30_v_n0.4` | PASS | model=two_plane; acc=0.9259259259259259; seam_med=0.46417287588312484 |
| `two_plane_a30_h_n0.0` | PASS | model=two_plane; acc=1.0; seam_med=1.258093562403704e-05 |
| `two_plane_a30_h_n0.4` | PASS | model=two_plane; acc=0.9814814814814815; seam_med=0.27519133854007816 |
| `two_plane_a60_v_n0.0` | PASS | model=two_plane; acc=1.0; seam_med=0.04499716292150983 |
| `two_plane_a60_v_n0.4` | PASS | model=two_plane; acc=0.9565217391304348; seam_med=0.44108193577438637 |
| `two_plane_a60_h_n0.0` | PASS | model=two_plane; acc=1.0; seam_med=0.2079992593344382 |
| `two_plane_a60_h_n0.4` | PASS | model=two_plane; acc=0.9777777777777777; seam_med=0.2474506773788286 |
| `two_plane_a90_v_n0.0` | PASS | model=two_plane; acc=1.0; seam_med=0.1408665026293668 |
| `two_plane_a90_v_n0.4` | PASS | model=two_plane; acc=0.96; seam_med=0.4292830927078736 |
| `two_plane_a90_h_n0.0` | PASS | model=two_plane; acc=1.0; seam_med=0.16792512811969174 |
| `two_plane_a90_h_n0.4` | PASS | model=two_plane; acc=1.0; seam_med=0.5850286636976123 |

## Gates covered

- one-plane vs two-plane classification
- assignment accuracy
- per-plane reprojection
- seam mismatch
- single-H baseline fails on genuine two-plane scenes
- no regression intended in flat-wall suite (run separately)

## Physical action

**Not requested yet.** See `docs/TWO_PLANE_PHYSICAL_SETUP_REQUEST.md` (blocked until this suite stays green).
