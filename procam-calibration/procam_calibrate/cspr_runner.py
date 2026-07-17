"""Run CSPR-Net from INTEGRATION_ROOT without modifying CSPR sources."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def run_cspr_inference(
    cspr_root: Path,
    run_dir: Path,
    mode: str = "inference_custom",
    image_path: Path | None = None,
    python_bin: str | None = None,
) -> dict:
    """
    Execute CSPR-Net experimental script unchanged via subprocess.
    Copies pre_warped outputs into run_dir/prewarps.
    """
    venv_python = cspr_root / ".venv" / "bin" / "python"
    py = python_bin or (str(venv_python) if venv_python.exists() else "python3")
    script = cspr_root / "deep_learning_exp_for_gradient.py"
    cmd = [py, str(script), mode]
    if mode == "inference_custom":
        cmd.append(str(image_path) if image_path else str(cspr_root / "real_data" / "test.jpg"))

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prewarps").mkdir(parents=True, exist_ok=True)
    (run_dir / "calibration").mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"cspr_{mode}_stdout.txt"
    stderr_path = log_dir / f"cspr_{mode}_stderr.txt"
    (run_dir / "commands.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cspr_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.time() - t0
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)

    out_dir = cspr_root / "neural_results_exp"
    copied = []
    for name in ("pre_warped.png", "pre_warped_test.png", "fake_cam.png"):
        src = out_dir / name
        if src.exists():
            dst = run_dir / "prewarps" / f"cspr_{name}"
            shutil.copy2(src, dst)
            copied.append(str(dst))

    weights_ok = "Models loaded successfully" in proc.stdout
    result = {
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "weights_loaded": weights_ok,
        "copied_outputs": copied,
        "command": cmd,
        "pass": proc.returncode == 0 and weights_ok and len(copied) > 0,
    }
    (run_dir / "calibration" / "cspr_run.json").write_text(json.dumps(result, indent=2))
    return result


def prepare_cspr_train_tree(
    run_dir: Path,
    captures_dir: Path,
    patterns_dir: Path,
    work_dir: Path,
) -> dict:
    """
    Stage capture files into a CSPR-compatible real_data layout for training.
    Does not modify CSPR_NET_ROOT; creates a working copy under run_dir.
    Expected capture stems (from prepare-capture):
      05_cspr_grid_1, 06_cspr_grid_2, 07_cspr_grid_3, 08_cspr_red
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    real = work_dir / "real_data"
    real.mkdir(exist_ok=True)
    mapping = {
        "05_cspr_grid_1": ("1_pro.png", "1_image_distorted_origin.jpeg"),
        "06_cspr_grid_2": ("2_pro.png", "2_image_distorted_origin.jpeg"),
        "07_cspr_grid_3": ("3_pro.png", "3_image_distorted_origin.jpeg"),
    }
    copied = []
    for stem, (pro_name, cam_name) in mapping.items():
        pro_src = patterns_dir / f"{stem}.png"
        # accept .jpg/.jpeg/.png for captures
        cam_src = None
        for ext in (".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG"):
            cand = captures_dir / f"{stem}{ext}"
            if cand.exists():
                cam_src = cand
                break
        if not pro_src.exists() or cam_src is None:
            raise FileNotFoundError(f"Missing pair for {stem}: pattern={pro_src.exists()} capture={cam_src}")
        shutil.copy2(pro_src, real / pro_name)
        # CSPR expects jpeg origin names; copy bytes even if png
        shutil.copy2(cam_src, real / cam_name)
        copied.append({"pattern": str(pro_src), "capture": str(cam_src)})

    red_cap = None
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        cand = captures_dir / f"08_cspr_red{ext}"
        if cand.exists():
            red_cap = cand
            break
    if red_cap is None:
        raise FileNotFoundError("Missing capture for 08_cspr_red")
    shutil.copy2(patterns_dir / "08_cspr_red.png", real / "4_proj.png")
    shutil.copy2(red_cap, real / "red_distorted_origin.jpeg")
    copied.append({"pattern": "08_cspr_red.png", "capture": str(red_cap)})

    meta = {"work_dir": str(work_dir), "real_data": str(real), "pairs": copied}
    (run_dir / "calibration" / "cspr_train_staging.json").write_text(json.dumps(meta, indent=2))
    return meta
