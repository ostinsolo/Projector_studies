"""CSPR-Net scene-specific adaptation for a new physical setup.

Repository-supported procedure (deep_learning_exp_for_gradient.py):
  - Mode `train`: full scene-specific optimization of CoordinateNet weights
    from scratch (or optional PRETRAINED_* fine-tune) using photometric +
    cycle-consistency losses on 3 pattern pairs + red mask.
  - Mode `inference` / `inference_custom`: direct inference with saved .pth

For a new projector/camera/wall/viewpoint, bundled checkpoints are NOT valid.
This module stages captures into a working copy and runs **full model training**
(scene-specific optimization) with Config adapted to the measured camera
resolution. Original CSPR_NET_ROOT sources are not modified.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
import time
from pathlib import Path


TRAIN_STEMS = [
    ("05_cspr_grid_1", "1"),
    ("06_cspr_grid_2", "2"),
    ("07_cspr_grid_3", "3"),
]

ITER_RE = re.compile(
    r"Iter\s+(\d+)\s+\[Img\s+(\d+)\]\s+\|\s+Loss:\s+([0-9.]+)",
    re.IGNORECASE,
)


def find_capture(captures_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = captures_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def resolve_torch_device(device: str = "auto") -> str:
    """Map CLI --device to a torch device string. Default baseline remains CPU-first auto."""
    import torch

    choice = (device or "auto").lower().strip()
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False")
        return "cuda"
    if choice == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("Requested --device mps but MPS is not available")
        return "mps"
    if choice == "auto":
        # Keep historical CSPR baseline: cuda if present else cpu (do not auto-pick MPS).
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unknown device {device!r}; expected auto|cpu|mps|cuda")


def _stream_train_process(
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    progress_path: Path,
    total_iters: int,
    env: dict,
) -> int:
    """Run training with live tee + progress.jsonl. Preserves child exit code."""
    t0 = time.time()
    last_iter_t = t0
    last_iter_n = 0
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as out_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as err_f, open(progress_path, "a", encoding="utf-8") as prog_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None and proc.stderr is not None

        def handle_line(line: str, sink) -> None:
            nonlocal last_iter_t, last_iter_n
            sink.write(line)
            sink.flush()
            m = ITER_RE.search(line)
            if not m:
                return
            it = int(m.group(1))
            loss = float(m.group(3))
            now = time.time()
            elapsed = now - t0
            di = max(1, it - last_iter_n) if it >= last_iter_n else 50
            dt = max(1e-6, now - last_iter_t)
            # When prints are every 50 iters, attribute dt across the gap
            spi = dt / max(it - last_iter_n, 1) if it > last_iter_n else dt / 50.0
            remaining = spi * max(0, total_iters - it)
            rec = {
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "iteration": it,
                "loss": loss,
                "elapsed_s": round(elapsed, 3),
                "seconds_per_iteration": round(spi, 6),
                "eta_remaining_s": round(remaining, 1),
            }
            prog_f.write(json.dumps(rec) + "\n")
            prog_f.flush()
            last_iter_t = now
            last_iter_n = it

        # Interleave by draining stdout then stderr (training logs mostly stdout)
        import selectors

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ, ("out", out_f))
        sel.register(proc.stderr, selectors.EVENT_READ, ("err", err_f))
        while proc.poll() is None or sel.get_map():
            for key, _ in sel.select(timeout=0.5):
                stream_name, sink = key.data
                line = key.fileobj.readline()
                if line == "":
                    try:
                        sel.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue
                if stream_name == "out":
                    handle_line(line, sink)
                else:
                    sink.write(line)
                    sink.flush()
            if proc.poll() is not None and not sel.get_map():
                break
        # Drain leftovers
        for stream, sink, is_out in (
            (proc.stdout, out_f, True),
            (proc.stderr, err_f, False),
        ):
            try:
                for line in stream:
                    if is_out:
                        handle_line(line, sink)
                    else:
                        sink.write(line)
                        sink.flush()
            except Exception:
                pass
        return int(proc.wait())


def stage_and_train(
    cspr_root: Path,
    run_dir: Path,
    patterns_dir: Path,
    captures_dir: Path,
    python_bin: str,
    iters: int = 6000,
    seed: int = 0,
    device: str = "auto",
) -> dict:
    work = run_dir / "calibration" / "cspr_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    real = work / "real_data"
    real.mkdir()
    out_dir = work / "neural_results_exp"
    out_dir.mkdir()

    # Copy script + create adapted Config wrapper
    src_script = cspr_root / "deep_learning_exp_for_gradient.py"
    staged_pairs = []
    cam_w = cam_h = None
    for stem, idx in TRAIN_STEMS:
        pro = patterns_dir / f"{stem}.png"
        cam = find_capture(captures_dir, stem)
        if not pro.exists() or cam is None:
            raise FileNotFoundError(f"Missing train pair {stem}")
        # Keep originals; CSPR expects jpeg names for origin captures
        shutil.copy2(pro, real / f"{idx}_pro.png")
        dest_cam = real / f"{idx}_image_distorted_origin.jpeg"
        # write as jpeg if needed
        import cv2

        img = cv2.imread(str(cam), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Unreadable {cam}")
        cam_h, cam_w = img.shape[:2]
        cv2.imwrite(str(dest_cam), img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        staged_pairs.append({"stem": stem, "pattern": str(pro), "capture": str(cam), "size": [cam_w, cam_h]})

    red = find_capture(captures_dir, "08_cspr_red")
    if red is None:
        raise FileNotFoundError("08_cspr_red")
    import cv2

    red_img = cv2.imread(str(red), cv2.IMREAD_COLOR)
    cv2.imwrite(str(real / "red_distorted_origin.jpeg"), red_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    shutil.copy2(patterns_dir / "08_cspr_red.png", real / "4_proj.png")

    # Read projector size from a pattern
    proj = cv2.imread(str(patterns_dir / "05_cspr_grid_1.png"), cv2.IMREAD_COLOR)
    proj_h, proj_w = proj.shape[:2]

    torch_device = resolve_torch_device(device)

    # Build runner that patches Config after import via exec of modified copy
    script_body = src_script.read_text()
    # Replace hardcoded resolutions and disable destructive crop for non-3072 cameras
    script_body = script_body.replace("H_CAM, W_CAM = 1728, 3072", f"H_CAM, W_CAM = {cam_h}, {cam_w}")
    script_body = script_body.replace("H_PROJ, W_PROJ = 1080, 1920", f"H_PROJ, W_PROJ = {proj_h}, {proj_w}")
    script_body = script_body.replace("ITERS = 6000", f"ITERS = {iters}")
    # Force device selection (future runs). Active in-flight trainers are unaffected.
    script_body = script_body.replace(
        'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
        f'DEVICE = torch.device("{torch_device}")',
    )
    # Disable crop: make preprocess_and_crop a no-op that writes undistorted path
    patch = textwrap.dedent(
        f'''
# --- procam-calibrate adaptation patch ---
import os as _os
_os.chdir(r"{work}")
# Force relative paths to work directory
Config.TRAIN_PAIRS = [
    ("./real_data/1_image_distorted_origin.jpeg", "./real_data/1_pro.png"),
    ("./real_data/2_image_distorted_origin.jpeg", "./real_data/2_pro.png"),
    ("./real_data/3_image_distorted_origin.jpeg", "./real_data/3_pro.png"),
]
Config.MASK_EXTRACT_PATH = "./real_data/red_distorted_origin.jpeg"
Config.OUTPUT_DIR = "./neural_results_exp"
Config.H_CAM, Config.W_CAM = {cam_h}, {cam_w}
Config.H_PROJ, Config.W_PROJ = {proj_h}, {proj_w}
Config.H_TRAIN = int(Config.H_PROJ * Config.TRAIN_SCALE)
Config.W_TRAIN = int(Config.W_PROJ * Config.TRAIN_SCALE)
Config.ITERS = {iters}
Config.DEVICE = __import__("torch").device("{torch_device}")

def preprocess_and_crop(img_path):
    """Adaptation: no CSPR hardcoded crop; use capture as-is."""
    import cv2 as _cv
    img = _cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    out = img_path.replace("_origin.jpeg", ".jpeg").replace("_origin.jpg", ".jpg")
    if out == img_path:
        out = img_path + ".cropped.jpeg"
    _cv.imwrite(out, img)
    print(f"[adapt] using uncropped capture -> {{out}} size={{img.shape[1]}}x{{img.shape[0]}}")
    return out
'''
    )
    # Append patch before main by writing a driver
    adapted = work / "train_adapted.py"
    adapted.write_text(script_body + "\n\n" + patch + "\n")

    driver = work / "run_train.py"
    driver.write_text(
        textwrap.dedent(
            f'''
            import random, numpy as np, torch, sys
            random.seed({seed})
            np.random.seed({seed})
            torch.manual_seed({seed})
            sys.argv = ["train_adapted.py", "train"]
            # Execute adapted script as __main__
            g = {{"__name__": "__main__", "__file__": r"{adapted}"}}
            with open(r"{adapted}") as f:
                code = f.read()
            # Re-apply preprocess_and_crop override after class defs by exec in two stages is hard;
            # instead monkeypatch after import-like exec without running main, then run.
            ns = {{"__name__": "cspr_adapt_mod", "__file__": r"{adapted}"}}
            exec(compile(code.replace('if __name__ == "__main__":', 'if False and __name__ == "__main__":'), r"{adapted}", "exec"), ns)
            # Override preprocess in module ns
            def preprocess_and_crop(img_path):
                import cv2 as _cv
                img = _cv.imread(img_path)
                out = img_path.replace("_origin.jpeg", ".jpeg")
                if out == img_path:
                    out = img_path + ".nocrop.jpeg"
                _cv.imwrite(out, img)
                print(f"[adapt] uncropped {{out}} {{img.shape[1]}}x{{img.shape[0]}}")
                return out
            ns["preprocess_and_crop"] = preprocess_and_crop
            Config = ns["Config"]
            Config.TRAIN_PAIRS = [
                ("./real_data/1_image_distorted_origin.jpeg", "./real_data/1_pro.png"),
                ("./real_data/2_image_distorted_origin.jpeg", "./real_data/2_pro.png"),
                ("./real_data/3_image_distorted_origin.jpeg", "./real_data/3_pro.png"),
            ]
            Config.MASK_EXTRACT_PATH = "./real_data/red_distorted_origin.jpeg"
            Config.OUTPUT_DIR = "./neural_results_exp"
            Config.H_CAM, Config.W_CAM = {cam_h}, {cam_w}
            Config.H_PROJ, Config.W_PROJ = {proj_h}, {proj_w}
            Config.TRAIN_SCALE = 0.25
            Config.H_TRAIN = int(Config.H_PROJ * Config.TRAIN_SCALE)
            Config.W_TRAIN = int(Config.W_PROJ * Config.TRAIN_SCALE)
            Config.ITERS = {iters}
            Config.DEVICE = torch.device("{torch_device}")
            print(f"[adapt] Config.DEVICE={{Config.DEVICE}}")
            import os
            os.chdir(r"{work}")
            warper = ns["NeuralWarper"](MODE="train")
            warper.train()
            '''
        )
    )

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout = log_dir / "cspr_train_stdout.txt"
    stderr = log_dir / "cspr_train_stderr.txt"
    progress_path = log_dir / "cspr_train_progress.jsonl"
    # Unbuffered python for live Iter lines (future runs only; does not affect in-flight trainers)
    cmd = [python_bin, "-u", str(driver)]
    prev = (run_dir / "commands.txt").read_text() if (run_dir / "commands.txt").exists() else ""
    (run_dir / "commands.txt").write_text(prev + "\n" + " ".join(cmd) + "\n")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    exit_code = _stream_train_process(
        cmd=cmd,
        cwd=work,
        stdout_path=stdout,
        stderr_path=stderr,
        progress_path=progress_path,
        total_iters=iters,
        env=env,
    )
    elapsed = time.time() - t0
    ckpt_c2p = out_dir / "net_c2p.pth"
    ckpt_p2c = out_dir / "net_p2c.pth"
    result = {
        "procedure": "full_scene_specific_training",
        "not_mere_finetune": True,
        "repository_entry": "deep_learning_exp_for_gradient.py train",
        "adapted_camera_resolution": [cam_w, cam_h],
        "projector_resolution": [proj_w, proj_h],
        "iters": iters,
        "seed": seed,
        "device_requested": device,
        "device_resolved": torch_device,
        "elapsed_s": elapsed,
        "exit_code": exit_code,
        "streaming_logs": True,
        "progress_jsonl": str(progress_path),
        "checkpoints": {
            "net_c2p": str(ckpt_c2p) if ckpt_c2p.exists() else None,
            "net_p2c": str(ckpt_p2c) if ckpt_p2c.exists() else None,
        },
        "staged_pairs": staged_pairs,
        "work_dir": str(work),
        "pass": exit_code == 0 and ckpt_c2p.exists() and ckpt_p2c.exists(),
    }
    (run_dir / "calibration" / "cspr_adapt.json").write_text(json.dumps(result, indent=2))
    return result


def run_inference_custom(
    work_dir: Path,
    python_bin: str,
    content_path: Path,
    run_dir: Path,
) -> dict:
    """Run inference_custom using adapted working tree checkpoints."""
    driver = work_dir / "run_infer.py"
    # simpler: invoke a small script loading nets from work_dir
    script = textwrap.dedent(
        f'''
        import os, sys
        os.chdir(r"{work_dir}")
        sys.argv = ["train_adapted.py", "inference_custom", r"{content_path}"]
        # Use original adapted file if present
        ns = {{}}
        path = r"{work_dir / "train_adapted.py"}"
        code = open(path).read().replace('if __name__ == "__main__":', 'if False:')
        exec(compile(code, path, "exec"), ns)
        def preprocess_and_crop(img_path):
            import cv2
            img = cv2.imread(img_path)
            out = img_path.replace("_origin.jpeg", ".jpeg")
            if out == img_path:
                out = img_path + ".nocrop.jpeg"
            cv2.imwrite(out, img)
            return out
        ns["preprocess_and_crop"] = preprocess_and_crop
        warper = ns["NeuralWarper"](MODE="inference_custom")
        warper.run_inference(img_path=r"{content_path}")
        '''
    )
    driver.write_text(script)
    proc = subprocess.run([python_bin, str(driver)], cwd=str(work_dir), capture_output=True, text=True)
    (run_dir / "logs" / "cspr_infer_stdout.txt").write_text(proc.stdout)
    (run_dir / "logs" / "cspr_infer_stderr.txt").write_text(proc.stderr)
    pre = work_dir / "neural_results_exp" / "pre_warped_test.png"
    if not pre.exists():
        pre = work_dir / "neural_results_exp" / "pre_warped.png"
    dest = run_dir / "prewarps" / "prewarp_best.png"
    dest.parent.mkdir(exist_ok=True)
    if pre.exists():
        shutil.copy2(pre, dest)
        shutil.copy2(pre, run_dir / "projected_validation" / "prewarp_to_project.png")
    return {
        "exit_code": proc.returncode,
        "prewarp": str(dest) if dest.exists() else None,
        "pass": proc.returncode == 0 and dest.exists(),
    }
