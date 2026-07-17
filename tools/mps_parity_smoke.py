#!/usr/bin/env python3
"""Small MPS compatibility + numerical-parity smoke test.

Run AFTER the active CPU CSPR fit completes. Does not touch an in-flight trainer.
Does not launch a full wall fit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("procam-test-data/mps_parity_smoke.json"))
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    report: dict = {
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "torch_version": torch.__version__,
    }
    if not report["mps_available"]:
        report["pass"] = False
        report["reason"] = "MPS not available"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2

    torch.manual_seed(0)
    x_cpu = torch.randn(2, 3, 64, 64)
    w_cpu = torch.randn(8, 3, 3, 3)
    y_cpu = F.conv2d(x_cpu, w_cpu, padding=1)
    y_cpu = torch.relu(y_cpu)
    y_cpu = y_cpu.mean()

    x_mps = x_cpu.to("mps")
    w_mps = w_cpu.to("mps")
    t0 = time.time()
    y_mps = F.conv2d(x_mps, w_mps, padding=1)
    y_mps = torch.relu(y_mps)
    y_mps = y_mps.mean()
    torch.mps.synchronize()
    mps_s = time.time() - t0

    diff = float((y_mps.cpu() - y_cpu).abs().item())
    report.update(
        {
            "cpu_value": float(y_cpu.item()),
            "mps_value": float(y_mps.cpu().item()),
            "abs_diff": diff,
            "mps_elapsed_s": mps_s,
            "pass": diff < 1e-4,
            "note": "Smoke only — not a full CSPR fit. Run a full MPS fit only after pass.",
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
