"""procam-calibrate CLI entry point."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .capture import prepare_capture_package, validate_capture
from .cspr_runner import run_cspr_inference
from .metrics import compare_uncorrected_corrected, measure_validation_image
from .paths import Roots, new_run_dir
from .patterns import generate_capture_pattern_set, make_validation_target
from .refine import refine_prewarp
from .synthetic import run_synthetic_gate
import cv2


def _write_env(run_dir: Path, roots: Roots) -> None:
    lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"platform={platform.platform()}",
        f"python={sys.version}",
        f"cspr={roots.cspr}",
        f"gs={roots.gs}",
        f"integration={roots.integration}",
        f"test_data={roots.test_data}",
    ]
    try:
        import torch

        lines.append(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as e:
        lines.append(f"torch_error={e}")
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n")


def cmd_doctor(args: argparse.Namespace) -> int:
    roots = Roots.resolve()
    report = {
        "version": __version__,
        "roots": {
            "cspr_exists": roots.cspr.is_dir(),
            "gs_exists": roots.gs.is_dir(),
            "integration_exists": roots.integration.is_dir(),
            "test_data_exists": roots.test_data.is_dir(),
            "paths": {k: str(v) for k, v in roots.__dict__.items()},
        },
        "cspr_weights": {
            "sim": (roots.cspr / "neural_results_simulation" / "net_c2p.pth").exists(),
            "exp": (roots.cspr / "neural_results_exp" / "net_c2p.pth").exists(),
        },
        "cspr_venv": (roots.cspr / ".venv" / "bin" / "python").exists(),
        "gs_data": (roots.gs / "data").exists(),
        "gs_cuda_host": False,
    }
    try:
        import torch

        report["torch"] = {"version": torch.__version__, "cuda": torch.cuda.is_available()}
        report["gs_cuda_host"] = bool(torch.cuda.is_available())
    except Exception as e:
        report["torch"] = {"error": str(e)}

    print(json.dumps(report, indent=2))
    ok = report["roots"]["cspr_exists"] and report["cspr_weights"]["exp"]
    return 0 if ok else 1


def cmd_generate_patterns(args: argparse.Namespace) -> int:
    roots = Roots.resolve()
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(roots.test_data, "patterns")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_env(run_dir, roots)
    manifest = generate_capture_pattern_set(
        run_dir / "projector_patterns", args.proj_w, args.proj_h
    )
    print(json.dumps({"run_dir": str(run_dir), "n": len(manifest["patterns"])}, indent=2))
    return 0


def cmd_prepare_capture(args: argparse.Namespace) -> int:
    roots = Roots.resolve()
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(roots.test_data, "flatwall")
    _write_env(run_dir, roots)
    result = prepare_capture_package(run_dir, args.proj_w, args.proj_h)
    # snapshot PROJECT_STATE if present
    state = roots.integration.parent / "PROJECT_STATE.md"
    if state.exists():
        shutil.copy2(state, run_dir / "PROJECT_STATE_snapshot.md")
    print(json.dumps(result, indent=2))
    print(f"\nUSER_ACTION_REQUIRED: see {run_dir / 'capture_request.md'}")
    return 0


def cmd_validate_capture(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    result = validate_capture(run_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 2


def cmd_run_cspr(args: argparse.Namespace) -> int:
    roots = Roots.resolve()
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(roots.test_data, "cspr")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_env(run_dir, roots)
    img = Path(args.image) if args.image else None
    result = run_cspr_inference(roots.cspr, run_dir, mode=args.mode, image_path=img)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


def cmd_generate_prewarp(args: argparse.Namespace) -> int:
    """
    For flat-wall sessions before CSPR retrain: copy CSPR custom inference output
    or generate validation target prewarp placeholder from synthetic H if provided.
    Primary path: run CSPR inference_custom on a content image.
    """
    roots = Roots.resolve()
    run_dir = Path(args.run_dir)
    content = Path(args.content) if args.content else None
    if content is None:
        # default: write validation target at projector res and run CSPR on it
        proj_w, proj_h = args.proj_w, args.proj_h
        content = run_dir / "projected_validation" / "content_validation.png"
        content.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(content), make_validation_target(proj_h, proj_w))
    result = run_cspr_inference(
        roots.cspr, run_dir, mode="inference_custom", image_path=content
    )
    # also store as canonical name
    src = run_dir / "prewarps" / "cspr_pre_warped_test.png"
    if not src.exists():
        src = run_dir / "prewarps" / "cspr_pre_warped.png"
    if src.exists():
        shutil.copy2(src, run_dir / "prewarps" / "prewarp_best.png")
        shutil.copy2(src, run_dir / "projected_validation" / "prewarp_to_project.png")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


def cmd_prepare_validation(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    pre = run_dir / "prewarps" / "prewarp_best.png"
    if not pre.exists():
        print(json.dumps({"pass": False, "error": "missing prewarps/prewarp_best.png"}))
        return 1
    dest = run_dir / "projected_validation" / "prewarp_to_project.png"
    shutil.copy2(pre, dest)
    request = f"""# Validation capture request

## Status: USER_ACTION_REQUIRED

1. Display fullscreen: `{dest.resolve()}`
2. Wait >= 0.5 s.
3. Capture one still with the locked iPhone.
4. Save as: `{(run_dir / 'captured_validation' / 'validation_corrected.jpg').resolve()}`
5. Run: `procam-calibrate measure-validation --run-dir {run_dir}`
"""
    (run_dir / "validation_capture_request.md").write_text(request)
    print(json.dumps({"pass": True, "project": str(dest), "request": str(run_dir / "validation_capture_request.md")}))
    return 0


def cmd_measure_validation(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out = run_dir / "error_visualizations"
    unc = None
    # uncorrected capture if present
    from .capture import find_capture

    unc_path = find_capture(run_dir / "camera_captures_raw", "09_validation_uncorrected")
    results = {}
    if unc_path:
        results["uncorrected"] = measure_validation_image(unc_path, out, "uncorrected")
        unc = results["uncorrected"]
    cor_path = Path(args.corrected) if args.corrected else run_dir / "captured_validation" / "validation_corrected.jpg"
    if not cor_path.exists():
        # try other extensions
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            cand = run_dir / "captured_validation" / f"validation_corrected{ext}"
            if cand.exists():
                cor_path = cand
                break
    if not cor_path.exists():
        print(json.dumps({"pass": False, "error": f"missing corrected capture: {cor_path}"}))
        return 2
    results["corrected"] = measure_validation_image(cor_path, out, "corrected")
    if unc and results.get("corrected"):
        results["comparison"] = compare_uncorrected_corrected(unc, results["corrected"])
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if results.get("comparison", {}).get("acceptance_pass") else 1


def cmd_refine(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    pre = run_dir / "prewarps" / "prewarp_best.png"
    cor = run_dir / "captured_validation" / "validation_corrected.jpg"
    for ext in (".jpg", ".jpeg", ".png"):
        if not cor.exists():
            cand = run_dir / "captured_validation" / f"validation_corrected{ext}"
            if cand.exists():
                cor = cand
    metrics_path = run_dir / "metrics.json"
    best = 1e9
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        best = m.get("corrected", {}).get("median_residual_px", best)
    out = refine_prewarp(run_dir, pre, cor, iteration=args.iteration, best_median=best)
    print(json.dumps(out, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    metrics = {}
    if (run_dir / "metrics.json").exists():
        metrics = json.loads((run_dir / "metrics.json").read_text())
    report = f"""# Run report — {run_dir.name}

## Metrics

```json
{json.dumps(metrics, indent=2)}
```

## Artifacts

See directories under `{run_dir}`.
"""
    (run_dir / "report.md").write_text(report)
    print(report)
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    roots = Roots.resolve()
    run_dir = Path(args.run_dir) if args.run_dir else new_run_dir(roots.test_data, "synthetic")
    _write_env(run_dir, roots)
    result = run_synthetic_gate(
        run_dir / "synthetic",
        proj_w=args.proj_w,
        proj_h=args.proj_h,
        cam_w=args.cam_w,
        cam_h=args.cam_h,
    )
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (run_dir / "report.md").write_text(
        f"# Synthetic gate\n\npass={result['pass']}\n\n```json\n{json.dumps(result, indent=2)}\n```\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


def cmd_auto_wall(args: argparse.Namespace) -> int:
    from .auto_wall import run_auto_wall

    run_dir = Path(args.run_dir)
    return run_auto_wall(
        run_dir,
        proj_w=args.proj_w,
        proj_h=args.proj_h,
        settle_s=args.settle,
        train_iters=args.train_iters,
        device=args.device,
        allow_main_display_fallback=args.allow_main_display,
        prefer_screen_id=args.screen_id,
        pipeline=args.pipeline,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="procam-calibrate", description="ProCam geometric calibration CLI")
    p.add_argument("--version", action="version", version=f"procam-calibrate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("doctor", help="Check environment and roots")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("generate-patterns", help="Generate projector patterns")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--proj-w", type=int, default=1920)
    s.add_argument("--proj-h", type=int, default=1080)
    s.set_defaults(func=cmd_generate_patterns)

    s = sub.add_parser("prepare-capture", help="Create capture package + USER_ACTION_REQUIRED request")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--proj-w", type=int, default=1920)
    s.add_argument("--proj-h", type=int, default=1080)
    s.set_defaults(func=cmd_prepare_capture)

    s = sub.add_parser("validate-capture", help="Validate capture folder completeness")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=cmd_validate_capture)

    s = sub.add_parser("run-cspr", help="Run CSPR-Net unchanged")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--mode", default="inference_custom", choices=["inference", "inference_custom"])
    s.add_argument("--image", default=None)
    s.set_defaults(func=cmd_run_cspr)

    s = sub.add_parser("generate-prewarp", help="Generate projector-space pre-warp via CSPR")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--content", default=None)
    s.add_argument("--proj-w", type=int, default=1920)
    s.add_argument("--proj-h", type=int, default=1080)
    s.set_defaults(func=cmd_generate_prewarp)

    s = sub.add_parser("prepare-validation", help="Prepare corrected projection capture request")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=cmd_prepare_validation)

    s = sub.add_parser("measure-validation", help="Measure geometric error on validation captures")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--corrected", default=None)
    s.set_defaults(func=cmd_measure_validation)

    s = sub.add_parser("refine", help="Closed-loop residual update (max 8 external iters)")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--iteration", type=int, default=1)
    s.set_defaults(func=cmd_refine)

    s = sub.add_parser("report", help="Write human-readable report.md")
    s.add_argument("--run-dir", required=True)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("synthetic", help="Run synthetic planar verification gate")
    s.add_argument("--run-dir", default=None)
    s.add_argument("--proj-w", type=int, default=1920)
    s.add_argument("--proj-h", type=int, default=1080)
    s.add_argument("--cam-w", type=int, default=1920)
    s.add_argument("--cam-h", type=int, default=1080)
    s.set_defaults(func=cmd_synthetic)

    s = sub.add_parser("auto-wall", help="Autonomous flat-wall calibration loop")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--proj-w", type=int, default=1920)
    s.add_argument("--proj-h", type=int, default=1080)
    s.add_argument("--settle", type=float, default=0.6)
    s.add_argument("--train-iters", type=int, default=6000)
    s.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Torch device for future CSPR fits (auto=cuda if available else cpu; does not auto-select MPS)",
    )
    s.add_argument(
        "--pipeline",
        choices=["homography", "cspr"],
        default="homography",
        help="Production MVP is homography; cspr is experimental non-planar only",
    )
    s.add_argument(
        "--allow-main-display",
        action="store_true",
        help="Allow using the main laptop display as projector (debug only; not exact external ProjFB)",
    )
    s.add_argument("--screen-id", type=int, default=None, help="Preferred NSScreenNumber")
    s.set_defaults(func=cmd_auto_wall)

    s = sub.add_parser(
        "flatwall-homography",
        help="Automatic classical flat-wall ChArUco + planar homography baseline",
    )
    s.add_argument("--run-dir", required=True)
    s.add_argument("--proj-w", type=int, default=1920)
    s.add_argument("--proj-h", type=int, default=1080)
    s.add_argument("--settle", type=float, default=0.8)
    s.add_argument("--screen-id", type=int, default=None)
    s.set_defaults(func=cmd_flatwall_homography)

    return p


def cmd_flatwall_homography(args: argparse.Namespace) -> int:
    from .homography_baseline import run_flatwall_homography_baseline

    result = run_flatwall_homography_baseline(
        Path(args.run_dir),
        proj_w=args.proj_w,
        proj_h=args.proj_h,
        settle_s=args.settle,
        screen_id=args.screen_id,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("pass") else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
