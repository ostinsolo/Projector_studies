"""macOS display discovery and fullscreen projector control (AppKit helper)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DisplayInfo:
    index: int
    screen_id: int
    width: int
    height: int
    pixel_width: int
    pixel_height: int
    origin_x: float
    origin_y: float
    scale: float
    is_main: bool
    name: str


def list_displays() -> list[DisplayInfo]:
    from AppKit import NSScreen

    out: list[DisplayInfo] = []
    for i, screen in enumerate(NSScreen.screens()):
        frame = screen.frame()
        scale = float(screen.backingScaleFactor())
        w = int(frame.size.width)
        h = int(frame.size.height)
        name = str(screen.localizedName()) if hasattr(screen, "localizedName") else f"Screen-{i}"
        sid = int(screen.deviceDescription().get("NSScreenNumber", i))
        out.append(
            DisplayInfo(
                index=i,
                screen_id=sid,
                width=w,
                height=h,
                pixel_width=int(round(w * scale)),
                pixel_height=int(round(h * scale)),
                origin_x=float(frame.origin.x),
                origin_y=float(frame.origin.y),
                scale=scale,
                is_main=(i == 0),
                name=name,
            )
        )
    return out


def select_projector_display(
    displays: list[DisplayInfo],
    prefer_resolution: tuple[int, int] = (1920, 1080),
    prefer_screen_id: Optional[int] = None,
    allow_main_fallback: bool = False,
) -> Optional[DisplayInfo]:
    if prefer_screen_id is not None:
        for d in displays:
            if d.screen_id == prefer_screen_id:
                return d
    tw, th = prefer_resolution
    non_main = [d for d in displays if not d.is_main]
    # Prefer wired/extended displays over AirPlay when both exist
    wired = [
        d
        for d in non_main
        if "airplay" not in d.name.lower() and "apple tv" not in d.name.lower()
    ]
    pool = wired or non_main
    for d in pool:
        if (d.width, d.height) == (tw, th) or (d.pixel_width, d.pixel_height) == (tw, th):
            return d
    if pool:
        return pool[0]
    if allow_main_fallback and displays:
        return displays[0]
    return None


class ProjectorController:
    def __init__(self, display: DisplayInfo, all_displays: list[DisplayInfo]):
        self.display = display
        self.all_displays = all_displays
        self._proc: Optional[subprocess.Popen] = None
        self._helper = Path(__file__).with_name("projector_helper_appkit.py")
        self._frame_path = Path(tempfile.gettempdir()) / f"procam_proj_frame_{os.getpid()}.png"

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        if not self._helper.exists():
            raise FileNotFoundError(self._helper)
        self._proc = subprocess.Popen(
            [
                sys.executable,
                str(self._helper),
                str(self.display.index),
                str(self.display.width),
                str(self.display.height),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        t0 = time.time()
        line = ""
        while time.time() - t0 < 25:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    err = self._proc.stderr.read() if self._proc.stderr else ""
                    raise RuntimeError(f"Projector helper exited early: {err}")
                time.sleep(0.05)
                continue
            line = line.strip()
            if line.startswith("READY"):
                return
        err = self._proc.stderr.read() if self._proc.stderr else ""
        raise RuntimeError(f"Projector helper failed to start; last='{line}' stderr='{err[:800]}'")

    def _cmd(self, command: str, timeout: float = 15.0) -> str:
        if not self._proc or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Projector helper not running")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError("Projector helper died")
                continue
            line = line.strip()
            if line.startswith("OK") or line.startswith("ERR"):
                return line
        raise TimeoutError(command)

    def show_path(self, path: Path) -> None:
        resp = self._cmd(f"SHOW {Path(path).resolve()}")
        if not resp.startswith("OK"):
            raise RuntimeError(resp)

    def show_image(self, img: np.ndarray, path: Optional[Path] = None) -> None:
        """Write a BGR image and display it (for in-memory frames / video)."""
        import cv2

        out = Path(path) if path is not None else self._frame_path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), img):
            raise RuntimeError(f"failed to write projector frame: {out}")
        self.show_path(out)

    def show_black(self) -> None:
        resp = self._cmd("BLACK")
        if not resp.startswith("OK"):
            raise RuntimeError(resp)

    def actual_resolution(self) -> dict:
        return {
            "logical": [self.display.width, self.display.height],
            "pixels": [self.display.pixel_width, self.display.pixel_height],
            "scale": self.display.scale,
            "screen_id": self.display.screen_id,
            "name": self.display.name,
            "is_main": self.display.is_main,
            "n_displays": len(self.all_displays),
            "helper": "appkit",
        }

    def shutdown(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._proc = None


def save_display_report(path: Path, displays: list[DisplayInfo], selected: Optional[DisplayInfo]) -> None:
    path.write_text(
        json.dumps(
            {
                "displays": [asdict(d) for d in displays],
                "selected": asdict(selected) if selected else None,
                "extended_display_available": len(displays) >= 2,
                "mirror_or_missing_external": len(displays) < 2,
                "note": (
                    "Exact projector-space output requires a separate extended display. "
                    "Current external may be AirPlay/Apple TV or a wired adapter."
                    if len(displays) >= 2
                    else "No extended display."
                ),
            },
            indent=2,
        )
    )
