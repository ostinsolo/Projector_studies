# Two-Plane Failure Modes

| Failure | Automatic recovery | When to ask user |
|---------|--------------------|------------------|
| Insufficient points on one plane | Alternate shifted/scaled ChArUco patterns; retry capture | Only if alternate sequence still fails |
| Stale Continuity frame | Increase flush; timestamp/diff check; recapture | — |
| Blur / exposure | Retry with longer settle; keep rejected frames | Room lighting if persistent |
| Plane-label swap | Canonicalize by projector-space centroid | — |
| Unstable seam | Bootstrap; exclusion band; more observations | Stronger fold if still unstable |
| One-H selected | Confirm distribution; alternate patterns | Stronger physical fold / wider coverage |
| Software exception | `ACCEPTANCE_FAILED` with traceback | Never classify as `BLOCKED_EXTERNAL` |
| No projector / iPhone | Setup request | `USER_ACTION_REQUIRED` |
| Genuine platform/hardware limit | Document alternatives | `BLOCKED_EXTERNAL` only with evidence |

## Terminal states

- `DONE` — absolute physical gate pass
- `USER_ACTION_REQUIRED` — hardware placement pending
- `ACCEPTANCE_FAILED` / measured physical limit — best matrices retained
- `BLOCKED_EXTERNAL` — external only
