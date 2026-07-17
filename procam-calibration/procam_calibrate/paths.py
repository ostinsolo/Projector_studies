from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import (
    DEFAULT_CSPR_NET_ROOT,
    DEFAULT_GS_PROCAMS_ROOT,
    DEFAULT_INTEGRATION_ROOT,
    DEFAULT_TEST_DATA_ROOT,
)


@dataclass(frozen=True)
class Roots:
    cspr: Path
    gs: Path
    integration: Path
    test_data: Path

    @classmethod
    def resolve(cls) -> "Roots":
        return cls(
            cspr=Path(os.environ.get("CSPR_NET_ROOT", DEFAULT_CSPR_NET_ROOT)).resolve(),
            gs=Path(os.environ.get("GS_PROCAMS_ROOT", DEFAULT_GS_PROCAMS_ROOT)).resolve(),
            integration=Path(os.environ.get("INTEGRATION_ROOT", DEFAULT_INTEGRATION_ROOT)).resolve(),
            test_data=Path(os.environ.get("TEST_DATA_ROOT", DEFAULT_TEST_DATA_ROOT)).resolve(),
        )


def new_run_dir(test_data: Path, prefix: str) -> Path:
    from datetime import datetime

    run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = test_data / "runs" / run_id
    for sub in (
        "logs",
        "projector_patterns",
        "camera_captures_raw",
        "camera_captures_validated",
        "calibration",
        "correspondences",
        "prewarps",
        "projected_validation",
        "captured_validation",
        "error_visualizations",
        "synthetic",
    ):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir
