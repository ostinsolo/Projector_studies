# Two-Plane Failure Modes

| Failure | Recovery | Terminal if stuck |
|---------|----------|-------------------|
| Blurry empty scene | Retry / flush; USER_ACTION if persistent | USER_ACTION_REQUIRED |
| False architectural seam (shadow) | Prior only; correspondence overrides | — |
| Low-contrast fold | Arch prior weak; correspondence seam | — |
| Insufficient points one plane | Alternate ChArUco patterns / retries | ACCEPTANCE_FAILED |
| One-H selected on real fold | Alternate patterns; then stronger fold request | USER_ACTION_REQUIRED |
| Unstable seam | Bootstrap + exclusion band | ACCEPTANCE_FAILED |
| Plane label swap | Centroid canonicalize | — |
| Mask overlap/gap | Assert exclusive masks | ACCEPTANCE_FAILED |
| Wrong matrix direction | Convention tests + composition check | — |
| Mirrored pre-warp | `detect_mirrored_or_inverted` | ACCEPTANCE_FAILED |
| Stale Continuity frame | Flush + reject | retry |
| Root state pollution | `project_state_path=None` by default | — |
| Software exception | ACCEPTANCE_FAILED + traceback (not BLOCKED_EXTERNAL) | ACCEPTANCE_FAILED |
