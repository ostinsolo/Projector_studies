"""macOS iPhone / Continuity camera capture via ffmpeg AVFoundation + OpenCV fallback."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraDevice:
    avf_index: int
    name: str
    backend: str = "avfoundation"


def list_avfoundation_video_devices() -> list[CameraDevice]:
    proc = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    text = proc.stderr
    devices: list[CameraDevice] = []
    in_video = False
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            break
        if not in_video:
            continue
        m = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if m:
            devices.append(CameraDevice(avf_index=int(m.group(1)), name=m.group(2).strip()))
    return devices


def select_iphone_camera(devices: list[CameraDevice]) -> Optional[CameraDevice]:
    for d in devices:
        name = d.name.lower()
        if "iphone" in name and "desk view" not in name:
            return d
    for d in devices:
        if "iphone" in d.name.lower():
            return d
    return None


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clipping_percentage(gray: np.ndarray) -> float:
    n = gray.size
    return float(((gray <= 2).sum() + (gray >= 253).sum()) / n)


def contrast_score(gray: np.ndarray) -> float:
    return float(gray.std())


def motion_score(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 1e9
    return float(np.mean(cv2.absdiff(a, b)))


class CameraController:
    def __init__(self, device: CameraDevice, prefer_size: Optional[tuple[int, int]] = None):
        self.device = device
        self.prefer_size = prefer_size
        self.last_frame: Optional[np.ndarray] = None
        self.frame_size: Optional[tuple[int, int]] = None  # (w,h)
        self.lock_status = {
            "api_supports_lock": False,
            "lens_locked": False,
            "zoom_locked": False,
            "focus_locked": False,
            "exposure_locked": False,
            "white_balance_locked": False,
            "note": "Continuity/AVFoundation capture via ffmpeg does not expose confirmed lock APIs; consistency enforced by frame checks.",
        }

    def capture_frame(
        self,
        out_path: Path,
        settle_s: float = 0.35,
        flush_n: int = 12,
        expect_texture: bool = True,
    ) -> dict:
        """
        Capture one high-quality frame after flushing stale buffers.
        Uses ffmpeg AVFoundation without forced video_size (Continuity often
        returns black frames when -video_size is set). OpenCV fallback.

        expect_texture=False: for uniform patterns (black/white/solid color) where
        Laplacian variance is not a sharpness signal; skip blur rejection and
        prefer stable mid-exposure frames over max-blur selection.
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        time.sleep(settle_s)
        # Capture a short burst to a temp pattern then pick sharpest via OpenCV path
        tmp_pat = out_path.parent / f".burst_{out_path.stem}_%03d.jpg"
        # Remove old burst
        for p in out_path.parent.glob(f".burst_{out_path.stem}_*.jpg"):
            p.unlink()

        # Do not force -video_size: Continuity Camera frequently yields clipped/black frames.
        n_frames = max(flush_n, 15)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "avfoundation",
            "-framerate",
            "30",
            "-i",
            f"{self.device.avf_index}:none",
            "-frames:v",
            str(n_frames),
            str(tmp_pat),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        frames = sorted(out_path.parent.glob(f".burst_{out_path.stem}_*.jpg"))
        if not frames:
            # fallback OpenCV
            return self._capture_opencv(out_path, flush_n=flush_n)

        scored = []
        prev = None
        for fp in frames:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_l = float(gray.mean())
            sc = {
                "path": str(fp),
                "blur": blur_score(gray),
                "clip": clipping_percentage(gray),
                "contrast": contrast_score(gray),
                "motion": motion_score(prev, gray),
                "mean": mean_l,
                "shape": list(img.shape),
            }
            # Discard fully clipped / empty frames
            if sc["clip"] > 0.95 or sc["contrast"] < 3.0:
                prev = gray
                continue
            if expect_texture:
                rank = sc["blur"] - 50 * sc["clip"] + 0.1 * sc["contrast"]
            else:
                # Uniform fields: prefer less crushing (not near-black), low motion,
                # and do not reward spurious Laplacian on sensor noise.
                rank = (
                    -abs(mean_l - 70.0)
                    - 0.5 * sc["motion"]
                    - 20.0 * max(0.0, sc["clip"] - 0.5)
                    + 0.05 * sc["contrast"]
                )
            scored.append((rank, sc, img))
            prev = gray
        if not scored:
            # retry via OpenCV before failing
            try:
                return self._capture_opencv(out_path, flush_n=max(flush_n, 20))
            except Exception as e:
                raise RuntimeError(
                    f"No usable frames from camera {self.device}; ffmpeg={proc.returncode}; opencv={e}"
                )

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_meta, best_img = scored[0]
        # Reject if extremely blurry — only meaningful when texture is expected
        if expect_texture and best_meta["blur"] < 8:
            raise RuntimeError(f"Capture too blurry: {best_meta}")
        h, w = best_img.shape[:2]
        self.frame_size = (w, h)
        self.last_frame = best_img
        # Save lossless PNG preferred
        if out_path.suffix.lower() in {".png", ""}:
            dest = out_path if out_path.suffix else out_path.with_suffix(".png")
            cv2.imwrite(str(dest), best_img)
        else:
            dest = out_path
            cv2.imwrite(str(dest), best_img, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        # cleanup burst
        for fp in frames:
            try:
                fp.unlink()
            except OSError:
                pass
        meta = {
            "file": str(dest),
            "device": asdict(self.device),
            "width": w,
            "height": h,
            "metrics": best_meta,
            "selection_score": best_score,
            "lock_status": self.lock_status,
            "ffmpeg_returncode": proc.returncode,
            "timestamp": time.time(),
        }
        return meta

    def _capture_opencv(self, out_path: Path, flush_n: int = 10) -> dict:
        # Map avf index heuristically: OpenCV often lists Continuity as 1 when MacBook is 0
        idx = min(self.device.avf_index, 1)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open camera index {idx}")
        if self.prefer_size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.prefer_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.prefer_size[1])
        best = None
        best_blur = -1
        best_meta = {}
        prev = None
        for i in range(flush_n + 5):
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            if i < flush_n:
                prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            b = blur_score(gray)
            if b > best_blur:
                best_blur = b
                best = frame.copy()
                best_meta = {
                    "blur": b,
                    "clip": clipping_percentage(gray),
                    "contrast": contrast_score(gray),
                    "motion": motion_score(prev, gray),
                }
            prev = gray
        cap.release()
        if best is None:
            raise RuntimeError("OpenCV capture failed")
        h, w = best.shape[:2]
        self.frame_size = (w, h)
        self.last_frame = best
        dest = out_path if out_path.suffix else out_path.with_suffix(".png")
        cv2.imwrite(str(dest), best)
        return {
            "file": str(dest),
            "device": asdict(self.device),
            "width": w,
            "height": h,
            "metrics": best_meta,
            "lock_status": self.lock_status,
            "backend": "opencv_fallback",
            "timestamp": time.time(),
        }


def save_camera_report(path: Path, devices: list[CameraDevice], selected: Optional[CameraDevice]) -> None:
    path.write_text(
        json.dumps(
            {
                "devices": [asdict(d) for d in devices],
                "selected": asdict(selected) if selected else None,
            },
            indent=2,
        )
    )
