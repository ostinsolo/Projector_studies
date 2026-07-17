"""Compare our two-H pipeline vs single-H / naive stereo-style one-H on synthetic data.

Does not call external repos; uses the same synthetic generator as production tests.
External full-intrinsic methods cannot consume sparse ChArUco two-plane points without
their Gray-code capture stack — recorded as N/A with reason.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Use production package (repo env), not the contrib venv
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "procam-calibration"))

from procam_calibrate.synthetic_two_plane import (  # noqa: E402
    generate_one_plane_control,
    generate_two_plane_correspondences,
    run_synthetic_suite,
)
from procam_calibrate.two_plane import (  # noqa: E402
    fit_single_homography,
    fit_two_homographies,
    estimate_straight_seam,
    build_plane_masks,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "two_plane_vs_baselines.json"


def _reproj(H, pts_p, pts_c):
    from procam_calibrate.charuco import apply_H

    err = np.linalg.norm(apply_H(H, pts_p) - pts_c, axis=1)
    return float(np.median(err)), float(np.percentile(err, 95))


def main() -> int:
    rows = []
    # One-plane control
    sc = generate_one_plane_control(0.0)
    t0 = time.perf_counter()
    fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
    t1 = time.perf_counter()
    H1, err1, _ = fit_single_homography(sc["pts_proj"], sc["pts_cam"])
    rows.append(
        {
            "case": "one_plane_control",
            "our_model": fit["model"],
            "our_valid": fit["valid"],
            "single_H_median": float(np.median(err1)),
            "single_H_p95": float(np.percentile(err1, 95)),
            "runtime_s": t1 - t0,
            "external_full_intrinsic_applicable": False,
            "reason": "synthetic sparse correspondences only; no Gray-code stack",
        }
    )

    for angle in (60.0, 90.0):
        for vertical in (True, False):
            for noise in (0.0, 0.4):
                sc = generate_two_plane_correspondences(
                    angle, vertical, noise, rng=np.random.default_rng(0)
                )
                t0 = time.perf_counter()
                fit = fit_two_homographies(sc["pts_proj"], sc["pts_cam"], ids=sc["ids"])
                t1 = time.perf_counter()
                H1, err1, _ = fit_single_homography(sc["pts_proj"], sc["pts_cam"])
                row = {
                    "case": f"two_plane_a{angle}_{'v' if vertical else 'h'}_n{noise}",
                    "our_model": fit.get("model"),
                    "our_valid": fit.get("valid"),
                    "assignment_acc": None,
                    "single_H_median": float(np.median(err1)),
                    "single_H_p95": float(np.percentile(err1, 95)),
                    "two_H_median": fit.get("comparison", {}).get("two_H_median_reproj_px"),
                    "two_H_p95": fit.get("comparison", {}).get("two_H_p95_reproj_px"),
                    "runtime_s": t1 - t0,
                    "mask_exclusive": None,
                }
                if fit.get("model") == "two_plane" and fit.get("valid"):
                    labels = np.array(fit["labels"])
                    true = sc["true_labels"]
                    acc = max(
                        float((labels == true).mean()),
                        float((labels == 1 - true).mean()),
                    )
                    row["assignment_acc"] = acc
                    Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
                    seam = estimate_straight_seam(sc["pts_proj"], labels, 1920, 1080, Ha, Hb)
                    ma, mb, _ = build_plane_masks(seam, 1920, 1080)
                    row["mask_exclusive"] = int(np.sum((ma > 0) & (mb > 0))) == 0
                    row["mask_full_coverage"] = int(np.sum((ma == 0) & (mb == 0))) == 0
                    row["seam_midpoint"] = seam.get("midpoint")
                rows.append(row)

    suite_dir = Path(__file__).resolve().parents[1] / "results" / "synthetic_suite_replay"
    t0 = time.perf_counter()
    summary = run_synthetic_suite(suite_dir)
    t1 = time.perf_counter()

    report = {
        "benchmark": "two_plane_vs_single_H_baselines",
        "cases": rows,
        "full_suite": {
            "all_pass": summary["all_pass"],
            "n_pass": summary["n_pass"],
            "n_cases": summary["n_cases"],
            "runtime_s": t1 - t0,
        },
        "external_notes": {
            "kamino410": "Gray-code + chessboard stereo; requires multi-pose planar board; not two unknown walls",
            "cho_kim": "no public code; requires custom multi-plane target + phase shift",
            "jin_seo_han": "no public code; semi-automated reference alignment + surface markers",
            "benjumea": "MATLAB; screen+mirror+aux camera; not wall fold mapping",
            "roomalive": "Kinect+Windows; depth mandatory",
            "gs_procams": "CUDA + noncommercial; out of milestone",
            "cspr": "experimental nonplanar neural; out of milestone",
        },
        "decision_hint": "CONTINUE_CURRENT_ARCHITECTURE unless Cho/Kim or Jin code becomes available",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"all_pass": summary["all_pass"], "n_cases": len(rows), "out": str(OUT)}, indent=2))
    return 0 if summary.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
