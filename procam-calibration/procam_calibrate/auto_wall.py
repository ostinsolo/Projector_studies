"""Autonomous flat-wall calibration state machine: procam-calibrate auto-wall."""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

import cv2
import numpy as np

from .camera_control import (
    CameraController,
    list_avfoundation_video_devices,
    save_camera_report,
    select_iphone_camera,
)
from .capture import REQUIRED_CAPTURE_STEMS, validate_capture
from .cspr_adapt import run_inference_custom, stage_and_train
from .display_control import (
    ProjectorController,
    list_displays,
    save_display_report,
    select_projector_display,
)
from .homography_baseline import run_flatwall_homography_baseline
from .metrics import compare_uncorrected_corrected, measure_validation_image
from .paths import Roots
from .patterns import make_validation_target
from .refine import refine_prewarp

STATES = [
    "PREFLIGHT",
    "DISCOVER_PROJECTOR",
    "DISCOVER_CAMERA",
    "HOMOGRAPHY_CALIBRATE",
    "VERIFY_CAPTURE",
    "GENERATE_PATTERNS",
    "CAPTURE_CALIBRATION",
    "VALIDATE_CALIBRATION_CAPTURES",
    "FIT_CSPR_SCENE",
    "GENERATE_PREWARP",
    "PROJECT_VALIDATION",
    "CAPTURE_VALIDATION",
    "MEASURE_RESIDUAL",
    "MEASURE_CORRECTED_TARGET",
    "DIAGNOSE_PLANAR_RESIDUAL",
    "REFINE_HOMOGRAPHY",
    "PROJECT_REFINED",
    "CAPTURE_REFINED",
    "MEASURE_REFINED",
    "ACCEPT_OR_ROLLBACK",
    "REFINE_PREWARP",
    "FINAL_REPRODUCTION",
    "DONE",
    "BLOCKED_EXTERNAL",
    "ACCEPTANCE_FAILED",
]

# Homography residual-refinement states (not diverted to CSPR / remeasure).
HOMOGRAPHY_REFINE_STATES = {
    "MEASURE_CORRECTED_TARGET",
    "DIAGNOSE_PLANAR_RESIDUAL",
    "REFINE_HOMOGRAPHY",
    "PROJECT_REFINED",
    "CAPTURE_REFINED",
    "MEASURE_REFINED",
    "ACCEPT_OR_ROLLBACK",
}


@dataclass
class AutoWallConfig:
    run_dir: Path
    proj_w: int = 1920
    proj_h: int = 1080
    settle_s: float = 0.6
    black_s: float = 0.25
    max_capture_retries: int = 4
    train_iters: int = 6000
    device: str = "auto"  # unused for homography MVP; kept for experimental CSPR
    max_refine_iters: int = 8
    convergence_median_delta: float = 0.05
    allow_main_display_fallback: bool = False
    prefer_screen_id: Optional[int] = None
    # Production flat-wall path is planar ChArUco homography (not CSPR).
    pipeline: str = "homography"
    # If set, live auto-wall status is written here. None = do not write project state.
    # Tests must always pass a temporary path; never leave this pointing at the repo root.
    project_state_path: Optional[Path] = None


class AutoWall:
    def __init__(self, cfg: AutoWallConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.state_path = self.run_dir / "auto_wall_state.json"
        self.roots = Roots.resolve()
        self.projector: Optional[ProjectorController] = None
        self.camera: Optional[CameraController] = None
        self.state = self._load_state()
        self._lock_fh: Optional[TextIO] = None

    def _acquire_run_lock(self) -> None:
        """Prevent concurrent auto-wall / train processes on the same run_dir."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.run_dir / "auto_wall.lock"
        fh = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            fh.seek(0)
            holder = fh.read().strip() or "(unknown)"
            fh.close()
            raise RuntimeError(
                f"Another auto-wall process already holds {lock_path}. "
                f"Holder info: {holder}. Kill the duplicate, then retry."
            ) from e
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started={datetime.now().isoformat()}\n")
        fh.flush()
        self._lock_fh = fh

        def _release() -> None:
            if self._lock_fh is not None:
                try:
                    fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
                    self._lock_fh.close()
                except OSError:
                    pass
                self._lock_fh = None

        atexit.register(_release)

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {
            "state": "PREFLIGHT",
            "completed": [],
            "history": [],
            "data": {},
            "updated_at": None,
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)
        self._update_project_state_file()

    def _transition(self, new_state: str, note: str = "") -> None:
        prev = self.state.get("state")
        self.state["history"].append(
            {"from": prev, "to": new_state, "note": note, "t": datetime.now().isoformat()}
        )
        if prev and prev not in self.state["completed"] and prev not in ("BLOCKED_EXTERNAL",):
            if new_state != prev:
                # mark previous as completed when leaving successfully
                pass
        self.state["state"] = new_state
        if new_state not in self.state["completed"] and new_state not in ("BLOCKED_EXTERNAL", "PREFLIGHT"):
            # completed list tracks finished stages
            pass
        self._save_state()

    def _mark_completed(self, stage: str) -> None:
        if stage not in self.state["completed"]:
            self.state["completed"].append(stage)
        self._save_state()

    def _update_project_state_file(self) -> None:
        """Write live auto-wall status only to the configured path (never inferred).

        When ``project_state_path`` is None, this is a no-op so tests and offline
        replay cannot mutate the repository-level PROJECT_STATE.md.
        """
        ps = self.cfg.project_state_path
        if ps is None:
            return
        st = self.state["state"]
        if st == "DONE":
            status = "DONE"
        elif st == "ACCEPTANCE_FAILED":
            status = "ACCEPTANCE_FAILED"
        elif st == "BLOCKED_EXTERNAL":
            status = "BLOCKED_EXTERNAL"
        else:
            status = "RUNNING"
        body = f"""# PROJECT_STATE (live auto-wall)

## Status

**{status}**

## Auto-wall state

`{st}`

Run: `{self.run_dir}`

Completed stages: {', '.join(self.state.get('completed', [])) or '(none)'}

Updated: {self.state.get('updated_at')}

See `{self.state_path}` for full machine state.
"""
        if st in ("BLOCKED_EXTERNAL", "ACCEPTANCE_FAILED"):
            body += f"\n## Notes\n\n{self.state.get('data', {}).get('blocker', '')}\n"
        ps = Path(ps)
        ps.parent.mkdir(parents=True, exist_ok=True)
        ps.write_text(body)

    def _log(self, msg: str) -> None:
        log = self.run_dir / "logs" / "auto_wall.log"
        log.parent.mkdir(exist_ok=True)
        line = f"{datetime.now().isoformat()} {msg}\n"
        with log.open("a") as f:
            f.write(line)
        print(msg, flush=True)

    def _bind_devices_if_needed(self) -> None:
        """Re-attach projector/camera on resume without rewinding the state machine."""
        # MEASURE_RESIDUAL remeasures saved captures offline; devices only for ROI recapture.
        need_proj = self.state["state"] in {
            "VERIFY_CAPTURE",
            "CAPTURE_CALIBRATION",
            "HOMOGRAPHY_CALIBRATE",
            "PROJECT_VALIDATION",
            "CAPTURE_VALIDATION",
            "REFINE_HOMOGRAPHY",
            "PROJECT_REFINED",
            "CAPTURE_REFINED",
            "MEASURE_REFINED",
            "ACCEPT_OR_ROLLBACK",
        }
        need_cam = self.state["state"] in {
            "VERIFY_CAPTURE",
            "CAPTURE_CALIBRATION",
            "HOMOGRAPHY_CALIBRATE",
            "CAPTURE_VALIDATION",
            "REFINE_HOMOGRAPHY",
            "PROJECT_REFINED",
            "CAPTURE_REFINED",
            "MEASURE_REFINED",
            "ACCEPT_OR_ROLLBACK",
        }
        if need_cam and self.camera is None:
            devices = list_avfoundation_video_devices()
            selected = select_iphone_camera(devices)
            if selected is None:
                raise RuntimeError("No iPhone camera available on resume")
            self.camera = CameraController(selected)
            self._log(f"Rebound camera on resume: {selected.name}")
        if need_proj and self.projector is None:
            displays = list_displays()
            selected = select_projector_display(
                displays,
                prefer_resolution=(self.cfg.proj_w, self.cfg.proj_h),
                prefer_screen_id=self.cfg.prefer_screen_id,
                allow_main_fallback=self.cfg.allow_main_display_fallback,
            )
            if selected is None:
                raise RuntimeError("No projector display available on resume")
            self.cfg.proj_w = int(selected.width)
            self.cfg.proj_h = int(selected.height)
            self.projector = ProjectorController(selected, displays)
            self.projector.start()
            self.projector.show_black()
            self._log(f"Rebound projector on resume: {selected.name} {selected.width}x{selected.height}")

    def run(self) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        self._acquire_run_lock()
        # Re-attempt discovery when previously blocked on devices
        if self.state.get("state") == "BLOCKED_EXTERNAL":
            blocker = str(self.state.get("data", {}).get("blocker", "")).lower()
            if "display" in blocker or "projector" in blocker or "iphone" in blocker or "camera" in blocker:
                self._log("Retrying after previous BLOCKED_EXTERNAL device issue")
                self.state["state"] = "DISCOVER_PROJECTOR"
                self._save_state()
        # Promote prior acceptance failures into homography finalize/remeasure
        if self.state.get("state") == "ACCEPTANCE_FAILED" and self.cfg.pipeline == "homography":
            self._log("ACCEPTANCE_FAILED → MEASURE_RESIDUAL (homography remeasure / finalize)")
            self.state["data"].pop("blocker", None)
            self.state["state"] = "MEASURE_RESIDUAL"
            self._save_state()
        # Legacy CSPR mid-states: divert to production homography path
        # (do not divert homography residual-refinement states or REFINE_PREWARP→refine)
        if self.cfg.pipeline == "homography" and self.state.get("state") in {
            "VERIFY_CAPTURE",
            "GENERATE_PATTERNS",
            "CAPTURE_CALIBRATION",
            "VALIDATE_CALIBRATION_CAPTURES",
            "FIT_CSPR_SCENE",
            "GENERATE_PREWARP",
            "PROJECT_VALIDATION",
            "CAPTURE_VALIDATION",
        }:
            # If homography artifacts already exist, jump to remeasure; else calibrate
            hb = self.run_dir / "homography_baseline"
            if (hb / "H_cam_from_proj.npy").exists() and (hb / "captures" / "charuco_uncorrected.png").exists():
                self._log(f"Divert {self.state['state']} → MEASURE_RESIDUAL (existing homography artifacts)")
                self.state["state"] = "MEASURE_RESIDUAL"
            else:
                self._log(f"Divert {self.state['state']} → HOMOGRAPHY_CALIBRATE (production MVP)")
                self.state["state"] = "HOMOGRAPHY_CALIBRATE"
            self._save_state()
        try:
            self._bind_devices_if_needed()
        except Exception as e:
            self._log(f"Device bind on resume failed: {e}")
        handlers: dict[str, Callable[[], None]] = {
            "PREFLIGHT": self.stage_preflight_init,
            "DISCOVER_PROJECTOR": self.stage_discover_projector,
            "DISCOVER_CAMERA": self.stage_discover_camera,
            "HOMOGRAPHY_CALIBRATE": self.stage_homography_calibrate,
            "VERIFY_CAPTURE": self.stage_verify_capture,
            "GENERATE_PATTERNS": self.stage_generate_patterns,
            "CAPTURE_CALIBRATION": self.stage_capture_calibration,
            "VALIDATE_CALIBRATION_CAPTURES": self.stage_validate_calibration,
            "FIT_CSPR_SCENE": self.stage_fit_cspr,
            "GENERATE_PREWARP": self.stage_generate_prewarp,
            "PROJECT_VALIDATION": self.stage_project_validation,
            "CAPTURE_VALIDATION": self.stage_capture_validation,
            "MEASURE_RESIDUAL": self.stage_measure_residual,
            "MEASURE_CORRECTED_TARGET": self.stage_measure_corrected_target,
            "DIAGNOSE_PLANAR_RESIDUAL": self.stage_diagnose_planar_residual,
            "REFINE_HOMOGRAPHY": self.stage_refine_homography,
            "PROJECT_REFINED": self.stage_refine_homography,
            "CAPTURE_REFINED": self.stage_refine_homography,
            "MEASURE_REFINED": self.stage_refine_homography,
            "ACCEPT_OR_ROLLBACK": self.stage_refine_homography,
            "REFINE_PREWARP": self.stage_refine,
            "FINAL_REPRODUCTION": self.stage_final_reproduction,
            "DONE": lambda: None,
            "BLOCKED_EXTERNAL": lambda: None,
            "ACCEPTANCE_FAILED": lambda: None,
        }
        guard = 0
        while self.state["state"] not in ("DONE", "BLOCKED_EXTERNAL", "ACCEPTANCE_FAILED") and guard < 40:
            guard += 1
            st = self.state["state"]
            self._log(f"STATE {st}")
            try:
                self._bind_devices_if_needed()
                handlers[st]()
            except Exception as e:
                self._log(f"ERROR in {st}: {e}\n{traceback.format_exc()}")
                self.state["data"]["last_error"] = {"state": st, "error": str(e), "trace": traceback.format_exc()}
                self._save_state()
                # software errors: retry once by re-entering; after repeated, block
                fails = self.state["data"].setdefault("fail_counts", {})
                fails[st] = fails.get(st, 0) + 1
                if fails[st] >= 3:
                    self.state["data"]["blocker"] = f"Repeated failure in {st}: {e}"
                    self._transition("BLOCKED_EXTERNAL", str(e))
                    return 2
                time.sleep(1)
        return 0 if self.state["state"] == "DONE" else 2

    # --- stages ---

    def stage_preflight_init(self) -> None:
        env = {
            "started": datetime.now().isoformat(),
            "roots": {k: str(v) for k, v in self.roots.__dict__.items()},
        }
        (self.run_dir / "environment.txt").write_text(json.dumps(env, indent=2))
        # Discover camera first so evidence exists even if projector is missing
        self._transition("DISCOVER_CAMERA", "init")

    def stage_discover_projector(self) -> None:
        self.run_dir.joinpath("calibration").mkdir(exist_ok=True)
        selected = None
        displays = []
        for attempt in range(6):
            displays = list_displays()
            selected = select_projector_display(
                displays,
                prefer_resolution=(self.cfg.proj_w, self.cfg.proj_h),
                prefer_screen_id=self.cfg.prefer_screen_id,
                allow_main_fallback=self.cfg.allow_main_display_fallback,
            )
            save_display_report(self.run_dir / "calibration" / "displays.json", displays, selected)
            self.state["data"]["displays"] = json.loads((self.run_dir / "calibration" / "displays.json").read_text())
            if selected is not None:
                break
            self._log(f"Projector display not found (attempt {attempt+1}/6); retrying…")
            time.sleep(3)
        if selected is None:
            self.state["data"]["blocker"] = (
                "No extended projector display is available to software. "
                f"Detected {len(displays)} display(s): "
                + ", ".join(f"{d.name} {d.width}x{d.height}" for d in displays)
                + ". The iPhone camera IS available, but exact projector-space fullscreen "
                "control requires the projector as an **extended** (non-mirrored) macOS display "
                f"preferably at {self.cfg.proj_w}x{self.cfg.proj_h}. "
                "Connect/configure that display, then re-run `procam-calibrate auto-wall` "
                f"--run-dir {self.run_dir} (state will resume)."
            )
            self._transition("BLOCKED_EXTERNAL", "no projector display")
            return
        # Warn if resolution mismatch
        res_ok = (selected.width, selected.height) == (self.cfg.proj_w, self.cfg.proj_h) or (
            selected.pixel_width,
            selected.pixel_height,
        ) == (self.cfg.proj_w, self.cfg.proj_h)
        self.state["data"]["projector"] = {
            "selected": selected.__dict__,
            "resolution_match": res_ok,
        }
        # Adapt framebuffer size to the actual selected display (exact ProjFB)
        self.cfg.proj_w = int(selected.width)
        self.cfg.proj_h = int(selected.height)
        if "airplay" in selected.name.lower() or "apple tv" in selected.name.lower():
            self.cfg.settle_s = max(self.cfg.settle_s, 2.0)
            self.cfg.black_s = max(self.cfg.black_s, 0.5)
            self._log(f"AirPlay/Apple TV display detected — using settle_s={self.cfg.settle_s}s")
        self.state["data"]["projector_resolution"] = [self.cfg.proj_w, self.cfg.proj_h]
        self.projector = ProjectorController(selected, displays)
        self.projector.start()
        meta = self.projector.actual_resolution()
        (self.run_dir / "calibration" / "projector_live.json").write_text(json.dumps(meta, indent=2))
        # smoke: show black
        self.projector.show_black()
        self._mark_completed("DISCOVER_PROJECTOR")
        if self.cfg.pipeline == "homography":
            self._transition("HOMOGRAPHY_CALIBRATE", "projector ready — production homography MVP")
        else:
            self._transition("GENERATE_PATTERNS", "projector ready — experimental CSPR path")

    def stage_discover_camera(self) -> None:
        devices = list_avfoundation_video_devices()
        selected = select_iphone_camera(devices)
        self.run_dir.joinpath("calibration").mkdir(exist_ok=True)
        save_camera_report(self.run_dir / "calibration" / "cameras.json", devices, selected)
        if selected is None:
            self.state["data"]["blocker"] = (
                "No iPhone camera device found via AVFoundation. Devices: "
                + ", ".join(d.name for d in devices)
            )
            if self.projector:
                self.projector.shutdown()
            self._transition("BLOCKED_EXTERNAL", "no iphone camera")
            return
        self.camera = CameraController(selected)
        probe = self.run_dir / "camera_captures_raw" / "_probe" / "discover_iphone.png"
        meta = self.camera.capture_frame(probe, settle_s=0.5)
        (self.run_dir / "calibration" / "camera_live.json").write_text(json.dumps(meta, indent=2))
        self.state["data"]["camera"] = meta
        self._mark_completed("DISCOVER_CAMERA")
        self._transition("DISCOVER_PROJECTOR", "camera ready")

    def stage_homography_calibrate(self) -> None:
        """Production flat-wall path: ChArUco → H → pre-warp → corrected capture → metrics."""
        assert self.projector and self.camera
        result = run_flatwall_homography_baseline(
            run_dir=self.run_dir,
            proj_w=self.cfg.proj_w,
            proj_h=self.cfg.proj_h,
            settle_s=max(self.cfg.settle_s, 0.8),
            reuse_devices=(self.projector, self.camera),
        )
        self.state["data"]["homography"] = {
            k: result.get(k)
            for k in (
                "pass",
                "valid",
                "case",
                "error",
                "homography_fit",
                "before_detection",
                "after_detection",
                "comparison",
                "roi_recapture",
                "n_proj_self_detect",
                "pipeline",
            )
        }
        metrics_path = self.run_dir / "metrics.json"
        if metrics_path.exists():
            self.state["data"]["metrics"] = json.loads(metrics_path.read_text())
        else:
            self.state["data"]["metrics"] = {
                "uncorrected": result.get("uncorrected_metrics"),
                "corrected": result.get("corrected_metrics"),
                "comparison": result.get("comparison"),
            }
        acceptance = bool(result.get("pass") and result.get("valid"))
        self.state["data"]["acceptance_pass"] = acceptance
        comp = result.get("comparison") or (self.state["data"].get("metrics") or {}).get("comparison") or {}
        abs_ok = bool(comp.get("absolute_gate_pass"))
        rel_ok = bool(comp.get("relative_gate_pass"))
        self.state["data"]["absolute_gate_pass"] = abs_ok
        self.state["data"]["relative_gate_pass"] = rel_ok
        self._mark_completed("HOMOGRAPHY_CALIBRATE")
        if result.get("error") in ("no_projector_display", "no_iphone_camera"):
            self.state["data"]["blocker"] = str(result.get("error"))
            self._transition("BLOCKED_EXTERNAL", "device missing during homography")
        elif abs_ok:
            self._transition("FINAL_REPRODUCTION", "homography absolute gate passed")
        elif acceptance or rel_ok:
            # Relative MVP ok — enter planar residual refinement for absolute accuracy
            self._transition(
                "MEASURE_CORRECTED_TARGET",
                "relative pass; start planar residual absolute-accuracy loop",
            )
        else:
            # Remeasure / ROI path may still recover from saved or new captures
            self._transition("MEASURE_RESIDUAL", "homography needs remeasure or ROI")

    def stage_verify_capture(self) -> None:
        assert self.projector and self.camera
        patterns = self.run_dir / "projector_patterns"
        preflight_dir = self.run_dir / "error_visualizations" / "preflight"
        preflight_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = {}
        for stem in ("00_black", "01_white", "02_border", "03_corners"):
            path = patterns / f"{stem}.png"
            if not path.exists():
                raise FileNotFoundError(path)
            self.projector.show_black()
            time.sleep(self.cfg.black_s)
            self.projector.show_path(path)
            time.sleep(self.cfg.settle_s)
            cap_path = preflight_dir / f"{stem}.png"
            meta = self.camera.capture_frame(cap_path, settle_s=0.3, flush_n=20)
            img = cv2.imread(str(cap_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            diagnostics[stem] = {
                "mean": float(gray.mean()),
                "std": float(gray.std()),
                "blur": meta["metrics"].get("blur"),
                "clip": meta["metrics"].get("clip"),
                "shape": list(img.shape),
            }
        black = cv2.imread(str(preflight_dir / "00_black.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        white = cv2.imread(str(preflight_dir / "01_white.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        diff = white - black
        global_contrast = float(white.mean() - black.mean())
        # Local projected region: strongest positive difference blob
        diff_clip = np.clip(diff, 0, 255).astype(np.uint8)
        # threshold at max(10, 40% of max diff)
        thr_val = max(10.0, 0.4 * float(diff_clip.max()) if diff_clip.max() > 0 else 10.0)
        _, mask = cv2.threshold(diff_clip, thr_val, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        ys, xs = np.where(mask > 0)
        local_contrast = 0.0
        bbox = None
        bounds_ok = False
        substantial = False
        if len(xs) >= 200:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = [x0, y0, x1, y1]
            roi = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            local_contrast = float(white[roi].mean() - black[roi].mean())
            h, w = mask.shape
            margin = 2
            bounds_ok = x0 > margin and y0 > margin and x1 < w - margin - 1 and y1 < h - margin - 1
            area_frac = (x1 - x0) * (y1 - y0) / float(w * h)
            substantial = area_frac >= 0.05 and local_contrast >= 12.0
        border = cv2.imread(str(preflight_dir / "02_border.png"), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(border, 50, 150)
        edge_frac = float(edges.mean() / 255.0)
        # Prefer local contrast when the display is a sub-region of the Continuity FOV
        contrast = max(global_contrast, local_contrast)
        report = {
            "diagnostics": diagnostics,
            "black_white_contrast_global": global_contrast,
            "black_white_contrast_local": local_contrast,
            "black_white_contrast": contrast,
            "edge_fraction_border": edge_frac,
            "diff_mask_pixels": int(len(xs)),
            "projected_bbox_cam": bbox,
            "all_boundaries_inside_frame": bounds_ok,
            "substantial_coverage": substantial,
            "display_name": self.projector.display.name if self.projector else None,
            "pass": (local_contrast >= 12.0 or global_contrast >= 15.0)
            and bbox is not None
            and substantial,
        }
        if bbox:
            vis = cv2.imread(str(preflight_dir / "01_white.png"))
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
            cv2.imwrite(str(preflight_dir / "coverage_overlay.png"), vis)
            cv2.imwrite(str(preflight_dir / "diff_mask.png"), mask)
        (preflight_dir / "preflight.json").write_text(json.dumps(report, indent=2))
        self.state["data"]["preflight"] = report
        if not report["pass"]:
            self.state["data"]["blocker"] = (
                "Preflight failed: iPhone Continuity camera does not see enough black→white "
                f"change on the active display ({self.projector.display.name}). "
                f"global_contrast={global_contrast:.1f}, local_contrast={local_contrast:.1f}. "
                "Software IS driving the external display correctly. "
                "Aim the physical iPhone (Continuity camera) at the screen/surface that shows "
                "the fullscreen patterns, fill the frame with that rectangle, darken the room, "
                "then re-run auto-wall. If you intended a wired HDMI projector via dongle, "
                "confirm it appears as a separate display in System Settings (currently only "
                "Apple TV AirPlay was detected besides the laptop panel)."
            )
            self._transition("BLOCKED_EXTERNAL", "preflight fail")
            return
        self._mark_completed("VERIFY_CAPTURE")
        self._transition("CAPTURE_CALIBRATION", "preflight ok")

    def stage_generate_patterns(self) -> None:
        from .patterns import generate_capture_pattern_set

        patterns = self.run_dir / "projector_patterns"
        patterns.mkdir(exist_ok=True)
        # Archive previous package if resolution differs (exact ProjFB required)
        sample = patterns / "00_black.png"
        need_regen = False
        reason = ""
        if not sample.exists():
            need_regen = True
            reason = "missing patterns"
        else:
            img = cv2.imread(str(sample))
            if img is None or (img.shape[1], img.shape[0]) != (self.cfg.proj_w, self.cfg.proj_h):
                need_regen = True
                prev = None if img is None else f"{img.shape[1]}x{img.shape[0]}"
                reason = f"resolution change {prev} -> {self.cfg.proj_w}x{self.cfg.proj_h}"
                arch = self.run_dir / "projector_patterns_archive" / f"before_{self.cfg.proj_w}x{self.cfg.proj_h}"
                arch.mkdir(parents=True, exist_ok=True)
                for p in patterns.glob("*.png"):
                    shutil.copy2(p, arch / p.name)
                for p in patterns.glob("*.json"):
                    shutil.copy2(p, arch / p.name)
        if need_regen:
            generate_capture_pattern_set(patterns, self.cfg.proj_w, self.cfg.proj_h)
            self._log(f"Generated patterns at {self.cfg.proj_w}x{self.cfg.proj_h} ({reason})")
        else:
            self._log("Preserving existing pattern package")
        # Clear prior captures if patterns regenerated
        if need_regen:
            self.state["data"]["captured_ok"] = []
        self._mark_completed("GENERATE_PATTERNS")
        self._transition("VERIFY_CAPTURE", "patterns ready")

    def stage_capture_calibration(self) -> None:
        assert self.projector and self.camera
        raw = self.run_dir / "camera_captures_raw"
        rejected = self.run_dir / "camera_captures_raw" / "_rejected"
        rejected.mkdir(exist_ok=True)
        # Solid / near-uniform projector patterns: Laplacian "blur" is not sharpness.
        low_texture_stems = {"00_black", "01_white", "08_cspr_red"}
        results = {}
        for stem in REQUIRED_CAPTURE_STEMS:
            pattern = self.run_dir / "projector_patterns" / f"{stem}.png"
            dest = raw / f"{stem}.png"
            # Skip if already valid
            if dest.exists() and dest.stat().st_size > 1000 and stem in self.state["data"].get("captured_ok", []):
                self._log(f"skip existing {stem}")
                continue
            ok = False
            last_err = None
            expect_texture = stem not in low_texture_stems
            for attempt in range(self.cfg.max_capture_retries):
                try:
                    self.projector.show_black()
                    time.sleep(self.cfg.black_s)
                    self.projector.show_path(pattern)
                    time.sleep(self.cfg.settle_s)
                    meta = self.camera.capture_frame(
                        dest, settle_s=0.35, flush_n=20, expect_texture=expect_texture
                    )
                    img = cv2.imread(str(dest))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mean_l = float(gray.mean())
                    blur = float(meta["metrics"].get("blur", 0))
                    if expect_texture and blur < 10:
                        raise RuntimeError(f"blur={blur:.2f}")
                    if stem == "01_white" and mean_l < 30:
                        raise RuntimeError(f"white too dark mean={mean_l:.1f}")
                    if stem == "00_black" and mean_l > 140:
                        raise RuntimeError(f"black too bright mean={mean_l:.1f}")
                    if stem == "08_cspr_red" and mean_l < 20:
                        raise RuntimeError(f"red too dark mean={mean_l:.1f} (AE/underexposure)")
                    results[stem] = meta
                    ok = True
                    break
                except Exception as e:
                    last_err = str(e)
                    if dest.exists():
                        shutil.move(str(dest), str(rejected / f"{stem}_attempt{attempt}.png"))
                    self._log(f"recapture {stem} attempt {attempt}: {e}")
            if not ok:
                raise RuntimeError(f"Failed to capture {stem}: {last_err}")
            self.state["data"].setdefault("captured_ok", []).append(stem)
            self._save_state()
        (self.run_dir / "calibration" / "capture_loop.json").write_text(json.dumps(results, indent=2))
        self._mark_completed("CAPTURE_CALIBRATION")
        self._transition("VALIDATE_CALIBRATION_CAPTURES", "captures done")

    def stage_validate_calibration(self) -> None:
        result = validate_capture(self.run_dir)
        (self.run_dir / "calibration" / "capture_validation.json").write_text(json.dumps(result, indent=2))
        if not result["pass"]:
            # auto-clear bad stems and recapture
            for stem in result.get("missing", []):
                if stem in self.state["data"].get("captured_ok", []):
                    self.state["data"]["captured_ok"].remove(stem)
            self._transition("CAPTURE_CALIBRATION", "validation failed -> recapture")
            return
        self._mark_completed("VALIDATE_CALIBRATION_CAPTURES")
        self._transition("FIT_CSPR_SCENE", "validation ok")

    def stage_fit_cspr(self) -> None:
        py = str(self.roots.cspr / ".venv" / "bin" / "python")
        result = stage_and_train(
            cspr_root=self.roots.cspr,
            run_dir=self.run_dir,
            patterns_dir=self.run_dir / "projector_patterns",
            captures_dir=self.run_dir / "camera_captures_validated",
            python_bin=py,
            iters=self.cfg.train_iters,
            seed=0,
            device=self.cfg.device,
        )
        self.state["data"]["cspr_adapt"] = result
        if not result["pass"]:
            raise RuntimeError(f"CSPR adaptation failed: {result}")
        self._mark_completed("FIT_CSPR_SCENE")
        self._transition("GENERATE_PREWARP", "trained")

    def stage_generate_prewarp(self) -> None:
        py = str(self.roots.cspr / ".venv" / "bin" / "python")
        work = self.run_dir / "calibration" / "cspr_work"
        content = self.run_dir / "projected_validation" / "content_validation.png"
        content.parent.mkdir(exist_ok=True)
        # desired content at camera resolution for CSPR generate_cam_without_distortion path:
        # inference_custom expects a projector-resolution content image.
        cv2.imwrite(str(content), make_validation_target(self.cfg.proj_h, self.cfg.proj_w))
        result = run_inference_custom(work, py, content, self.run_dir)
        self.state["data"]["prewarp"] = result
        if not result["pass"]:
            raise RuntimeError(result)
        # dimension check
        img = cv2.imread(result["prewarp"])
        h, w = img.shape[:2]
        if (w, h) != (self.cfg.proj_w, self.cfg.proj_h):
            self._log(f"WARN prewarp size {w}x{h} != {self.cfg.proj_w}x{self.cfg.proj_h}")
        self._mark_completed("GENERATE_PREWARP")
        self._transition("PROJECT_VALIDATION", "prewarp ready")

    def stage_project_validation(self) -> None:
        assert self.projector
        path = self.run_dir / "projected_validation" / "prewarp_to_project.png"
        self.projector.show_black()
        time.sleep(self.cfg.black_s)
        self.projector.show_path(path)
        time.sleep(self.cfg.settle_s)
        self._mark_completed("PROJECT_VALIDATION")
        self._transition("CAPTURE_VALIDATION", "displaying corrected")

    def stage_capture_validation(self) -> None:
        assert self.camera and self.projector
        # Also ensure uncorrected baseline exists from calibration set
        out = self.run_dir / "captured_validation" / "validation_corrected.png"
        out.parent.mkdir(exist_ok=True)
        # Re-show prewarp
        self.projector.show_path(self.run_dir / "projected_validation" / "prewarp_to_project.png")
        time.sleep(self.cfg.settle_s)
        meta = self.camera.capture_frame(out, settle_s=0.2)
        (self.run_dir / "calibration" / "validation_capture.json").write_text(json.dumps(meta, indent=2))
        self._mark_completed("CAPTURE_VALIDATION")
        self._transition("MEASURE_RESIDUAL", "captured validation")

    def stage_measure_residual(self) -> None:
        if self.cfg.pipeline == "homography":
            from .homography_baseline import (
                ensure_production_homography_artifacts,
                remeasure_or_recapture,
            )

            results = remeasure_or_recapture(
                self.run_dir,
                proj_w=self.cfg.proj_w,
                proj_h=self.cfg.proj_h,
                settle_s=max(self.cfg.settle_s, 0.8),
                projector=self.projector,
                camera=self.camera,
                bind_devices=self._bind_devices_for_recapture,
            )
            (self.run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
            self.state["data"]["metrics"] = results
            ensure_production_homography_artifacts(self.run_dir)
            comp = results.get("comparison", {})
            acceptance = bool(comp.get("acceptance_pass") and comp.get("valid"))
            self.state["data"]["acceptance_pass"] = acceptance
            self._mark_completed("MEASURE_RESIDUAL")
            self._transition("FINAL_REPRODUCTION", "homography remeasure complete")
            return

        out = self.run_dir / "error_visualizations"
        from .capture import find_capture

        unc = find_capture(self.run_dir / "camera_captures_raw", "09_validation_uncorrected")
        if unc is None:
            unc = find_capture(self.run_dir / "camera_captures_validated", "09_validation_uncorrected")
        results: dict[str, Any] = {}
        if unc:
            results["uncorrected"] = measure_validation_image(unc, out, "uncorrected")
        else:
            results["uncorrected"] = {
                "valid": False,
                "pass": False,
                "failure_reason": "missing_uncorrected_capture",
            }
        cor = self.run_dir / "captured_validation" / "validation_corrected.png"
        results["corrected"] = measure_validation_image(cor, out, "corrected")
        results["comparison"] = compare_uncorrected_corrected(
            results["uncorrected"], results["corrected"]
        )
        (self.run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
        self.state["data"]["metrics"] = results
        comp = results.get("comparison", {})
        acceptance = bool(comp.get("acceptance_pass") and comp.get("valid"))
        self.state["data"]["acceptance_pass"] = acceptance
        self._mark_completed("MEASURE_RESIDUAL")
        if acceptance:
            self._transition("FINAL_REPRODUCTION", "gate passed")
        elif not results["corrected"].get("valid") or not results["uncorrected"].get("valid"):
            self._transition("FINAL_REPRODUCTION", "invalid_validation_metrics")
        else:
            self._transition("REFINE_PREWARP", "need refine")

    def _bind_devices_for_recapture(self) -> tuple[Any, Any]:
        """Bind projector+camera for optional ROI validation recapture."""
        prev = self.state["state"]
        self.state["state"] = "HOMOGRAPHY_CALIBRATE"
        try:
            self._bind_devices_if_needed()
        finally:
            self.state["state"] = prev
        if self.projector is None or self.camera is None:
            raise RuntimeError("Devices required for ROI validation recapture")
        return self.projector, self.camera

    def stage_measure_corrected_target(self) -> None:
        """Homography absolute-accuracy: remeasure corrected capture with evaluation_ids."""
        from .residual_refine import remeasure_run_v2

        self._log("MEASURE_CORRECTED_TARGET")
        v2 = remeasure_run_v2(self.run_dir)
        self.state["data"]["metrics_v2"] = {
            k: (v2.get("corrected") or {}).get(k)
            for k in (
                "target_median_err_px",
                "target_p95_err_px",
                "evaluation_ids",
                "id_sets",
                "valid",
            )
        }
        self.state["data"]["comparison_v2"] = v2.get("comparison")
        self._mark_completed("MEASURE_CORRECTED_TARGET")
        self._transition("DIAGNOSE_PLANAR_RESIDUAL", "corrected target measured (v2 ids)")

    def stage_diagnose_planar_residual(self) -> None:
        """Homography absolute-accuracy: classify residual field before correction."""
        from .charuco import (
            build_evaluation_ids,
            detect_charuco,
            expected_visible_from_pattern,
            load_desired_target,
            load_proj_ids_json,
        )
        from .homography_baseline import homography_dir
        from .residual_refine import diagnose_residuals
        import cv2

        self._log("DIAGNOSE_PLANAR_RESIDUAL")
        out = homography_dir(self.run_dir)
        cor = out / "captures" / "charuco_corrected.png"
        desired = load_desired_target(out / "desired_target.json")
        proj_ids = load_proj_ids_json(out / "proj_corner_ids.json")
        img = cv2.imread(str(cor))
        cam_ids, _ = detect_charuco(img)
        exp = None
        pre = out / "prewarp_charuco.png"
        if pre.exists():
            exp = expected_visible_from_pattern(cv2.imread(str(pre)), proj_ids)
        id_sets = build_evaluation_ids(
            cam_ids,
            desired["desired_cam_by_id"],
            proj_ids,
            exp.get("expected_visible_ids") if exp else None,
        )
        diag = diagnose_residuals(
            cam_ids,
            desired,
            id_sets["evaluation_ids"],
            self.run_dir / "homography_refinement" / "diagnosis",
            image_bgr=img,
        )
        self.state["data"]["residual_diagnosis"] = {
            k: diag.get(k)
            for k in (
                "dominant_residual_class",
                "median_mag_px",
                "p95_mag_px",
                "median_dx",
                "median_dy",
                "constant_translation_hypothesis",
                "remaining_after_translation_median_px",
                "remaining_after_homography_median_px",
            )
        }
        self._mark_completed("DIAGNOSE_PLANAR_RESIDUAL")
        self._transition("REFINE_HOMOGRAPHY", f"residual class={diag.get('dominant_residual_class')}")

    def stage_refine_homography(self) -> None:
        """Homography-only residual loop (not CSPR). Covers PROJECT/CAPTURE/MEASURE/ACCEPT."""
        from .residual_refine import run_homography_refinement

        self._log("REFINE_HOMOGRAPHY → PROJECT_REFINED → CAPTURE_REFINED → MEASURE_REFINED → ACCEPT_OR_ROLLBACK")
        assert self.projector and self.camera
        # Record intermediate state names for observability
        for st in ("PROJECT_REFINED", "CAPTURE_REFINED", "MEASURE_REFINED", "ACCEPT_OR_ROLLBACK"):
            self.state["history"].append(
                {"state": st, "at": datetime.now().isoformat(), "note": "homography residual refine substep"}
            )
        result = run_homography_refinement(
            self.run_dir,
            self.projector,
            self.camera,
            self.cfg.proj_w,
            self.cfg.proj_h,
            settle_s=max(self.cfg.settle_s, 0.8),
        )
        self.state["data"]["homography_refinement"] = {
            k: result.get(k)
            for k in (
                "status",
                "absolute_gate_pass",
                "relative_gate_pass",
                "physical_limit_documented",
                "best_metrics",
                "baseline_metrics",
                "remaining_residual_class",
                "best_iteration",
                "best_strength",
                "best_prewarp",
            )
        }
        self.state["data"]["absolute_gate_pass"] = bool(result.get("absolute_gate_pass"))
        if (self.run_dir / "metrics.json").exists():
            self.state["data"]["metrics"] = json.loads((self.run_dir / "metrics.json").read_text())
        self._mark_completed("REFINE_HOMOGRAPHY")
        self._mark_completed("ACCEPT_OR_ROLLBACK")
        self._transition(
            "FINAL_REPRODUCTION",
            f"homography refine done: {result.get('status')}",
        )

    def stage_refine(self) -> None:
        if self.cfg.pipeline == "homography":
            # Replace CSPR refine skip with planar residual absolute-accuracy loop
            self._transition(
                "MEASURE_CORRECTED_TARGET",
                "homography residual refine (not CSPR REFINE_PREWARP)",
            )
            return
        data = self.state["data"]
        it = int(data.get("refine_iteration", 0)) + 1
        if it > self.cfg.max_refine_iters:
            self.state["data"]["blocker"] = (
                f"Exceeded max refine iterations ({self.cfg.max_refine_iters}) without meeting acceptance gate. "
                f"Last metrics: {json.dumps(data.get('metrics', {}), indent=2)[:1500]}"
            )
            self._transition("FINAL_REPRODUCTION", "max refine")
            return
        data["refine_iteration"] = it
        metrics = data.get("metrics", {})
        best = metrics.get("corrected", {}).get("median_residual_px", 1e9)
        prev_best = data.get("best_median", best)
        data["best_median"] = min(prev_best, best)
        pre = self.run_dir / "prewarps" / "prewarp_best.png"
        cor = self.run_dir / "captured_validation" / "validation_corrected.png"
        out = refine_prewarp(self.run_dir, pre, cor, iteration=it, best_median=prev_best)
        data["last_refine"] = out
        self._save_state()
        self._transition("PROJECT_VALIDATION", f"refine iter {it}")

    def stage_final_reproduction(self) -> None:
        metrics = self.state["data"].get("metrics", {})
        comp = metrics.get("comparison", {})
        acceptance = bool(
            (comp.get("acceptance_pass") and comp.get("valid"))
            or self.state["data"].get("acceptance_pass")
        )
        if self.cfg.pipeline == "homography":
            from .homography_baseline import ensure_production_homography_artifacts, write_homography_final_report

            ensure_production_homography_artifacts(self.run_dir)
            write_homography_final_report(
                self.run_dir,
                metrics=metrics,
                acceptance=acceptance,
                state=self.state,
            )
        else:
            report = self.run_dir / "report.md"
            report.write_text(
                f"""# Auto-wall final report

## Acceptance: {'PASS' if acceptance else 'FAIL'}

## Metrics

```json
{json.dumps(metrics, indent=2)}
```

## CSPR adaptation

```json
{json.dumps(self.state['data'].get('cspr_adapt', {}), indent=2)}
```

## State history

```json
{json.dumps(self.state.get('history', []), indent=2)}
```
"""
            )
        if self.projector:
            self.projector.shutdown()
        if acceptance:
            self._mark_completed("FINAL_REPRODUCTION")
            self._transition("DONE", "accepted")
        else:
            reason = (
                comp.get("failure_reason")
                or metrics.get("corrected", {}).get("failure_reason")
                or "acceptance_gate_failed"
            )
            self.state["data"]["blocker"] = (
                f"ACCEPTANCE_FAILED: pipeline completed but validation is not trustworthy "
                f"({reason}). See metrics.json and forensic_audit/."
            )
            self._transition("ACCEPTANCE_FAILED", "acceptance not met")


def run_auto_wall(run_dir: Path, **kwargs) -> int:
    cfg = AutoWallConfig(run_dir=run_dir, **kwargs)
    return AutoWall(cfg).run()
