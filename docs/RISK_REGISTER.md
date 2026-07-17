# Risk Register

Calibration-project risks. Severity: High / Medium / Low.

| ID | Risk | Sev | Impact | Mitigation / stop condition |
|----|------|-----|--------|------------------------------|
| R1 | GS-ProCams needs NVIDIA CUDA 11.8 + extensions | High | M2 cannot succeed on Apple Silicon Mac | Document blocker; defer GS; do not invent CPU port |
| R2 | GS datasets / checkpoints missing | High | No train/compensate output | Official SharePoint/Nepmap only; stop if unavailable |
| R3 | GS non-commercial license | High | Product integration blocked | Keep GS as research reference; own code in INTEGRATION_ROOT |
| R4 | CSPR no LICENSE | High | Unclear reuse of code/weights | Reproduce in place; do not copy into product without approval |
| R5 | Neither repo exports geometric LUT | High | Integration friction | INTEGRATION_ROOT must write lossless maps |
| R6 | Neither repo implements Gray-code SL | High | Automatic dense corr not provided | Implement SL in INTEGRATION_ROOT after M1/M2; OpenCV only as glue |
| R7 | CSPR hardcoded 3072×1728 / 1920×1080 | High | iPhone mismatch | Configurable sizes in integration; never silent resize |
| R8 | CSPR retrain per physical setup | Medium | Slow for new scenes | Classical SL preferred for MVP |
| R9 | GS multi-view requirement | Medium | Single iPhone MVP mismatch | Exclude GS from flat single-view path |
| R10 | Coordinate ambiguity (`H`, `warp`) | High | Silent wrong warps | Enforce COORDINATE_SYSTEMS.md naming |
| R11 | CSPR vs GS `align_corners` mismatch | Medium | Wrong if mixing grids | Do not mix NDC conventions |
| R12 | `compensate.py` desire/desired path typo | Medium | M2 compensate fails | Pass `--desired`; document |
| R13 | Shell script `-r`/`-s` bugs | Medium | Pipeline scripts fail | Use README Python commands |
| R14 | iPhone AE/AF/WB drift | High | Decode failure | Lock settings per protocol |
| R15 | Silent EXIF orientation / crop | High | Correspondence offset | Protocol forbids; validate metadata |
| R16 | Lens distortion ignored | Medium | Sub-pixel floor violated | Measure; document if uncorrected |
| R17 | Measurement variance > improvement | Medium | False “success” | Stop per stop-conditions |
| R18 | Temptation to add external ML | High | Scope violation | Hard ban in PROJECT_STATE |
| R19 | PyVista blocks CSPR ray-trace | Low | Headless repro annoyance | Prefer neural inference for M1 |
| R20 | Manual point picking treated as automatic | High | False MVP claim | Label diagnostics only |

## Active blockers for next milestones

| Milestone | Blocker |
|-----------|---------|
| M1 | None expected (data+weights local) |
| M2 | CUDA + datasets (external) |
| M3+ | Physical projector + iPhone session not yet captured |
