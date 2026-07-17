# Two-Plane Test Matrix

| Suite | Cases | Command / entry |
|-------|-------|-----------------|
| Full package | 71 | `pytest tests/` |
| Core two-plane synthetic | 14 | `run_synthetic_suite` |
| Extended synthetic | 25 | `run_extended_synthetic_suite` |
| CLI synthetic | core+extended | `procam-calibrate auto-two-plane --mode synthetic` |
| Architecture detection | fold / shadow / margins | `tests/test_architecture_analysis.py` |
| Coordinate guards | composition / mirror / wrong-H | same |
| Physical SM offline | empty→architecture→propose | offline fixture |
| State isolation | no root PROJECT_STATE write | isolation tests |
| Flat-wall regression | existing auto-wall / homography tests | full `tests/` |

Immutable physical flat-wall runs are never overwritten by these tests.
