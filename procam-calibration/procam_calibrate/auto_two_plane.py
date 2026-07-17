"""Physical two-plane calibration state machine (separate from flat-wall / CSPR)."""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .camera_control import (
    CameraController,
    blur_score,
    contrast_score,
    list_avfoundation_video_devices,
    save_camera_report,
    select_iphone_camera,
)
from .display_control import (
    ProjectorController,
    list_displays,
    save_display_report,
    select_projector_display,
)
from .homography_baseline import forward_H_proj_from_source, prewarp_from_forward_H
from .paths import Roots
from .patterns import (
    make_black,
    make_border,
    make_corner_markers,
    make_validation_target,
    make_white,
)
from .two_plane import (
    MIN_POINTS_PER_PLANE,
    bootstrap_seam_stability,
    build_plane_masks,
    define_desired_target_two_plane,
    estimate_straight_seam,
    fit_single_homography,
    fit_two_homographies,
    measure_two_plane_against_desired,
    seam_exclusion_mask,
    seam_mismatch_in_camera,
    two_plane_acceptance,
)
from .two_plane_observations import (
    aggregate_observations,
    extract_observations_from_capture,
    generate_pattern_set,
    observation_key,
    observations_to_arrays,
)
from .two_plane_refine import (
    MAX_PHYSICAL_ITERS,
    build_piecewise_from_forwards,
    coupled_accept,
    preserve_best,
    propose_coupled_candidates,
)
from .architecture_analysis import (
    TargetRegionConfig,
    analyse_architecture,
    render_synthetic_two_wall_scene,
    verify_seam_with_architecture_prior,
)
from .charuco import apply_H, detect_charuco, generate_charuco_image
from .excluded_geometry import (
    build_usable_and_exclusion_masks,
    classify_active_and_excluded_planes,
    estimate_wall_ceiling_boundary,
    save_exclusion_artifacts,
)
from .video_playback import prepare_playback_calibration, write_diagnostic_video
from .video_region import maximum_inscribed_rectangle, parse_aspect, save_video_region


def evaluate_projection_preflight(
    gray_black: np.ndarray,
    gray_white: np.ndarray,
    *,
    diff_threshold: float = 25.0,
    min_contrast: float = 35.0,
    min_area_frac: float = 0.04,
    min_side_brightness: float = 20.0,
) -> dict:
    """Score projector visibility inside the illuminated region (camera FOV-safe)."""
    gb = np.asarray(gray_black, dtype=np.float32)
    gw = np.asarray(gray_white, dtype=np.float32)
    if gb.shape != gw.shape:
        raise ValueError("black/white captures must match shape")
    diff = gw - gb
    mask = (diff > diff_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = mask > 0
    frac = float(m.mean())
    global_contrast = float(gw.mean() - gb.mean())
    if m.sum() < 500:
        return {
            "pass": False,
            "checks": {
                "black_white_contrast": global_contrast,
                "projected_contrast": 0.0,
                "contrast_ok": False,
                "bright_fraction": frac,
                "substantial_area": False,
                "left_brightness": 0.0,
                "right_brightness": 0.0,
                "both_sides_lit": False,
                "metric": "projected_region_diff",
            },
        }
    contrast = float(diff[m].mean())
    ys, xs = np.where(m)
    mid = int(np.median(xs))
    xx = np.arange(gw.shape[1])[None, :]
    left = m & (xx < mid)
    right = m & (xx >= mid)
    left_b = float(gw[left].mean()) if left.any() else 0.0
    right_b = float(gw[right].mean()) if right.any() else 0.0
    checks = {
        "black_white_contrast": global_contrast,
        "projected_contrast": contrast,
        "contrast_ok": contrast >= min_contrast,
        "bright_fraction": frac,
        "substantial_area": frac >= min_area_frac,
        "left_brightness": left_b,
        "right_brightness": right_b,
        "both_sides_lit": left_b >= min_side_brightness and right_b >= min_side_brightness,
        "metric": "projected_region_diff",
        "proj_bbox_cam": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }
    return {
        "pass": bool(checks["contrast_ok"] and checks["substantial_area"] and checks["both_sides_lit"]),
        "checks": checks,
    }


STATES = [
    "PREFLIGHT",
    "DISCOVER_PROJECTOR",
    "DISCOVER_CAMERA",
    "CAPTURE_EMPTY_SCENE",
    "ANALYSE_ARCHITECTURE",
    "PROPOSE_TARGET_REGION",
    "PROJECT_GEOMETRY_PROBE",
    "CAPTURE_GEOMETRY_PROBE",
    "VERIFY_TWO_PLANE_SETUP",
    "GENERATE_CALIBRATION_SEQUENCE",
    "PROJECT_CALIBRATION_SEQUENCE",
    "CAPTURE_CALIBRATION_SEQUENCE",
    "DETECT_MULTI_FRAME_CHARUCO",
    "EXTRACT_CORRESPONDENCES",
    "DETECT_ACTIVE_AND_EXCLUDED_PLANES",
    "FIT_SINGLE_H",
    "TEST_MULTI_PLANE",
    "FIT_TWO_H",
    "ASSIGN_SURFACES",
    "VALIDATE_MODEL_SELECTION",
    "ESTIMATE_SEAM",
    "VERIFY_SEAM",
    "ESTIMATE_WALL_CEILING_BOUNDARY",
    "BUILD_USABLE_PROJECTION_MASK",
    "OPTIMISE_MAXIMUM_VIDEO_REGION",
    "FIT_ACTIVE_WALL_HOMOGRAPHIES",
    "BUILD_MASKS",
    "DEFINE_DESIRED_TARGET",
    "BUILD_PIECEWISE_VIDEO_WARP",
    "BUILD_FINAL_EXCLUSION_MASK",
    "GENERATE_PIECEWISE_PREWARP",
    "PLAY_DIAGNOSTIC_VIDEO",
    "CAPTURE_DIAGNOSTIC_VIDEO",
    "MEASURE_VIDEO_GEOMETRY",
    "REFINE_VIDEO_REGION_AND_WARP",
    "PROJECT_UNCORRECTED_VALIDATION",
    "CAPTURE_UNCORRECTED_VALIDATION",
    "PROJECT_CORRECTED_VALIDATION",
    "CAPTURE_CORRECTED_VALIDATION",
    "MEASURE_TWO_PLANE_RESULT",
    "DIAGNOSE_TWO_PLANE_RESIDUAL",
    "REFINE_TWO_PLANE",
    "ACCEPT_OR_ROLLBACK",
    "SAVE_VIDEO_PLAYBACK_CALIBRATION",
    "FINAL_REPRODUCTION",
    "DONE",
    "USER_ACTION_REQUIRED",
    "ACCEPTANCE_FAILED",
    "BLOCKED_EXTERNAL",
]


@dataclass
class AutoTwoPlaneConfig:
    run_dir: Path
    mode: str = "physical"  # synthetic | physical
    proj_w: int = 1920
    proj_h: int = 1080
    settle_s: float = 0.8
    max_capture_retries: int = 4
    max_refine_iters: int = MAX_PHYSICAL_ITERS
    allow_main_display_fallback: bool = False
    prefer_screen_id: Optional[int] = None
    # If set, write live two-plane status here. None = never touch repo PROJECT_STATE.md.
    two_plane_state_path: Optional[Path] = None
    project_state_path: Optional[Path] = None  # unused alias guard; never auto-write root
    setup_confirmed: bool = False
    # Offline fixture: skip devices; use pre-built observations / images under run_dir
    offline: bool = False


class AutoTwoPlane:
    def __init__(self, cfg: AutoTwoPlaneConfig):
        self.cfg = cfg
        self.run_dir = cfg.run_dir
        self.state_path = self.run_dir / "two_plane_state.json"
        self.roots = Roots.resolve()
        self.projector: Optional[ProjectorController] = None
        self.camera: Optional[CameraController] = None
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {
            "state": "PREFLIGHT",
            "mode": self.cfg.mode,
            "completed": [],
            "history": [],
            "data": {},
            "updated_at": None,
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        self.state["mode"] = self.cfg.mode
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        tmp.replace(self.state_path)
        self._update_status_docs()

    def _transition(self, new_state: str, note: str = "") -> None:
        prev = self.state.get("state")
        self.state.setdefault("history", []).append(
            {"from": prev, "to": new_state, "note": note, "t": datetime.now().isoformat()}
        )
        self.state["state"] = new_state
        self._save_state()

    def _mark_completed(self, stage: str) -> None:
        if stage not in self.state.setdefault("completed", []):
            self.state["completed"].append(stage)
        self._save_state()

    def _log(self, msg: str) -> None:
        log = self.run_dir / "logs" / "auto_two_plane.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        line = f"{datetime.now().isoformat()} {msg}\n"
        with log.open("a") as f:
            f.write(line)
        print(msg, flush=True)

    def _update_status_docs(self) -> None:
        """Write run-local / configured two-plane status only — never pollute root PROJECT_STATE."""
        st = self.state["state"]
        if st == "DONE":
            status = "DONE"
        elif st == "USER_ACTION_REQUIRED":
            status = "USER_ACTION_REQUIRED"
        elif st == "ACCEPTANCE_FAILED":
            status = "ACCEPTANCE_FAILED"
        elif st == "BLOCKED_EXTERNAL":
            status = "BLOCKED_EXTERNAL"
        else:
            status = "RUNNING"
        body = f"""# TWO_PLANE_STATE (live)

## Flat-wall (unchanged)

**DONE / VALIDATED / ABSOLUTE PASS**

## Two-plane

**{status}**

Machine state: `{st}`

Run: `{self.run_dir}`

Mode: `{self.cfg.mode}`

Updated: {self.state.get('updated_at')}
"""
        # Always write run-local copy
        (self.run_dir / "TWO_PLANE_STATE_LIVE.md").write_text(body)
        # Optional configured path (tests pass tempfile; production may pass docs path)
        if self.cfg.two_plane_state_path is not None:
            p = Path(self.cfg.two_plane_state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        # Explicitly never write project_state_path unless set (and CLI sets run-local only)
        if self.cfg.project_state_path is not None:
            Path(self.cfg.project_state_path).write_text(
                f"# PROJECT_STATE_LIVE (two-plane)\n\nTwo-plane: **{status}** (`{st}`)\n"
                f"Flat-wall: DONE / VALIDATED / ABSOLUTE PASS\n"
            )

    def _ensure_dirs(self) -> None:
        for name in (
            "projector_patterns",
            "camera_captures_raw",
            "camera_captures_rejected",
            "correspondences",
            "model_fit",
            "seam",
            "desired_target",
            "prewarps",
            "projected_validation",
            "captured_validation",
            "burst_variability",
            "residual_diagnosis",
            "refinement",
            "error_visualizations",
            "logs",
            "preflight",
            "architecture",
            "excluded_geometry",
            "video_region",
            "video_playback",
            "diagnostic_video",
        ):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)

    def _write_env(self) -> None:
        lines = [
            f"timestamp={datetime.now().isoformat()}",
            f"platform={platform.platform()}",
            f"python={sys.version}",
            f"mode={self.cfg.mode}",
            f"proj={self.cfg.proj_w}x{self.cfg.proj_h}",
        ]
        (self.run_dir / "environment.txt").write_text("\n".join(lines) + "\n")

    def _append_command(self, cmd: str) -> None:
        with (self.run_dir / "commands.txt").open("a") as f:
            f.write(f"{datetime.now().isoformat()} {cmd}\n")

    def write_physical_setup_request(self) -> Path:
        """User-facing physical actions only."""
        md = self.run_dir / "capture_request.md"
        body = """# Two-plane physical setup required

## Status: USER_ACTION_REQUIRED

Complete these **physical** actions only. Do not capture images, edit JSON,
select points, or run commands — the agent will do all software steps after you
confirm readiness.

1. Use two rigid, matte, light-coloured flat panels or two walls meeting in a
   clear vertical corner.
2. Prefer an interior angle between approximately 60° and 100° for the first
   physical test.
3. Place the fold approximately near the centre third of the projector image,
   not directly at an extreme edge.
4. Ensure meaningful projected area exists on both planes.
5. Keep the projector fixed.
6. Keep the iPhone fixed.
7. Ensure the complete projection and the fold are visible in the iPhone frame
   (camera may be above, below, left, or right of the projector).
8. Darken the room enough for reliable ChArUco detection.
9. Do not move either device after confirming readiness.

Reply that the setup is ready. The agent will then run:

```
procam-calibrate auto-two-plane --run-dir <THIS_RUN_DIR> --mode physical --setup-confirmed
```
"""
        md.write_text(body)
        # Never write tracked docs/ from the state machine (tests and runs must
        # not pollute repository documentation). Canonical checklist lives in
        # docs/TWO_PLANE_PHYSICAL_SETUP_REQUEST.md and is maintained separately.
        manifest = {
            "status": "USER_ACTION_REQUIRED",
            "run_dir": str(self.run_dir),
            "user_actions_only": True,
            "software_complete": True,
            "next_flag": "--setup-confirmed",
        }
        (self.run_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
        return md

    # ---- state handlers ----

    def step_preflight(self) -> None:
        self._ensure_dirs()
        self._write_env()
        meta = {
            "run_id": self.run_dir.name,
            "created": datetime.now().isoformat(),
            "mode": self.cfg.mode,
            "proj_w": self.cfg.proj_w,
            "proj_h": self.cfg.proj_h,
            "pipeline": "two_plane_piecewise_homography",
            "cspr": False,
        }
        (self.run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
        self._append_command(f"auto-two-plane --mode {self.cfg.mode}")
        if self.cfg.mode == "physical" and not self.cfg.setup_confirmed and not self.cfg.offline:
            self.write_physical_setup_request()
            self._transition("USER_ACTION_REQUIRED", "awaiting physical fold setup")
            return
        self._mark_completed("PREFLIGHT")
        self._transition("DISCOVER_PROJECTOR")

    def step_discover_projector(self) -> None:
        if self.cfg.offline:
            self.state["data"]["display"] = {"offline": True, "width": self.cfg.proj_w, "height": self.cfg.proj_h}
            self._mark_completed("DISCOVER_PROJECTOR")
            self._transition("DISCOVER_CAMERA")
            return
        displays = list_displays()
        selected = select_projector_display(
            displays,
            prefer_resolution=(self.cfg.proj_w, self.cfg.proj_h),
            prefer_screen_id=self.cfg.prefer_screen_id,
            allow_main_fallback=self.cfg.allow_main_display_fallback,
        )
        save_display_report(self.run_dir / "display_report.json", displays, selected)
        if selected is None:
            self.state["data"]["blocker"] = "no_external_projector_display"
            self.write_physical_setup_request()
            self._transition("USER_ACTION_REQUIRED", "projector not found — confirm hardware setup")
            return
        self.projector = ProjectorController(selected, displays)
        self.projector.start()
        self.cfg.proj_w = int(selected.width)
        self.cfg.proj_h = int(selected.height)
        self.state["data"]["display"] = {
            "screen_id": selected.screen_id,
            "width": selected.width,
            "height": selected.height,
            "name": getattr(selected, "name", ""),
        }
        self._mark_completed("DISCOVER_PROJECTOR")
        self._transition("DISCOVER_CAMERA")

    def step_discover_camera(self) -> None:
        if self.cfg.offline:
            self.state["data"]["camera"] = {"offline": True}
            self._mark_completed("DISCOVER_CAMERA")
            self._transition("CAPTURE_EMPTY_SCENE")
            return
        devices = list_avfoundation_video_devices()
        cam = select_iphone_camera(devices)
        save_camera_report(self.run_dir / "camera_report.json", devices, cam)
        if cam is None:
            self.state["data"]["blocker"] = "no_iphone_camera"
            if self.projector:
                self.projector.shutdown()
            self.write_physical_setup_request()
            self._transition("USER_ACTION_REQUIRED", "iPhone camera not found")
            return
        self.camera = CameraController(cam)
        self.state["data"]["camera"] = {"name": cam.name, "avf_index": cam.avf_index}
        self._mark_completed("DISCOVER_CAMERA")
        self._transition("CAPTURE_EMPTY_SCENE")

    def step_capture_empty_scene(self) -> None:
        empty_dir = self.run_dir / "empty_scene"
        empty_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.offline:
            img, gt = render_synthetic_two_wall_scene(
                cam_w=1280, cam_h=720, seam_x_frac=0.5, rng=np.random.default_rng(0)
            )
            path = empty_dir / "empty_burst_00.png"
            cv2.imwrite(str(path), img)
            meta = {
                "offline": True,
                "timestamp": datetime.now().isoformat(),
                "resolution": [1280, 720],
                "gt_synthetic": gt,
                "blur": float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()),
            }
            (empty_dir / "empty_burst_00.json").write_text(json.dumps(meta, indent=2))
            self.state["data"]["empty_scene"] = {"path": str(path), "meta": meta}
            self._mark_completed("CAPTURE_EMPTY_SCENE")
            self._transition("ANALYSE_ARCHITECTURE")
            return
        assert self.camera
        # Ensure projector is black so architecture is natural scene
        if self.projector is not None:
            self.projector.show_black()
            time.sleep(self.cfg.settle_s)
        frames = []
        for i in range(3):
            path = empty_dir / f"empty_burst_{i:02d}.png"
            meta = self.camera.capture_frame(path, settle_s=0.35, flush_n=24)
            img = cv2.imread(str(path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = blur_score(gray)
            rec = {
                "path": str(path),
                "timestamp": datetime.now().isoformat(),
                "capture_meta": meta,
                "resolution": [int(img.shape[1]), int(img.shape[0])],
                "blur": blur,
                "accepted": blur >= 15.0,
            }
            (empty_dir / f"empty_burst_{i:02d}.json").write_text(json.dumps(rec, indent=2, default=str))
            if rec["accepted"]:
                frames.append(rec)
            else:
                rej = self.run_dir / "camera_captures_rejected" / path.name
                cv2.imwrite(str(rej), img)
        if not frames:
            self.write_physical_setup_request()
            self._transition("USER_ACTION_REQUIRED", "empty scene frames unusable")
            return
        # Stability: max pairwise mean abs diff among accepted
        imgs = [cv2.imread(f["path"]) for f in frames]
        stab = []
        for a, b in zip(imgs, imgs[1:]):
            stab.append(float(np.mean(cv2.absdiff(a, b))))
        self.state["data"]["empty_scene"] = {
            "frames": frames,
            "path": frames[0]["path"],
            "stability_mad": stab,
            "resolution": frames[0]["resolution"],
        }
        self._mark_completed("CAPTURE_EMPTY_SCENE")
        self._transition("ANALYSE_ARCHITECTURE")

    def step_analyse_architecture(self) -> None:
        path = Path(self.state["data"]["empty_scene"]["path"])
        img = cv2.imread(str(path))
        out = self.run_dir / "architecture"
        cfg = TargetRegionConfig(
            prefer_two_wall=True,
            prefer_seam_near_centre=True,
        )
        report = analyse_architecture(img, out, cfg)
        self.state["data"]["architecture"] = {
            "valid": report.get("valid"),
            "n_lines": report.get("n_lines"),
            "n_seam_candidates": len(
                [c for c in report.get("seam_candidates", []) if c.get("status") == "candidate"]
            ),
            "target_region": report.get("target_region"),
        }
        self._mark_completed("ANALYSE_ARCHITECTURE")
        self._transition("PROPOSE_TARGET_REGION")

    def step_propose_target_region(self) -> None:
        # Region already written by analyse_architecture; re-emit for stage clarity
        arch_path = self.run_dir / "architecture" / "target_region.json"
        if arch_path.exists():
            region = json.loads(arch_path.read_text())
        else:
            region = self.state["data"].get("architecture", {}).get("target_region") or {}
        (self.run_dir / "architecture" / "proposed_target_region.json").write_text(
            json.dumps(region, indent=2)
        )
        self.state["data"]["proposed_target_region"] = region
        self._mark_completed("PROPOSE_TARGET_REGION")
        self._transition("VERIFY_TWO_PLANE_SETUP")

    def _project_and_capture(self, img: np.ndarray, stem: str, flush_n: int = 24) -> tuple[np.ndarray, dict]:
        assert self.projector and self.camera
        path_p = self.run_dir / "preflight" / f"{stem}_proj.png"
        cv2.imwrite(str(path_p), img)
        self.projector.show_path(path_p)
        time.sleep(self.cfg.settle_s)
        cap_path = self.run_dir / "preflight" / f"{stem}_cam.png"
        meta = self.camera.capture_frame(cap_path, settle_s=0.35, flush_n=flush_n)
        frame = cv2.imread(str(cap_path))
        return frame, meta

    def step_verify_two_plane_setup(self) -> None:
        if self.cfg.offline:
            self.state["data"]["preflight"] = {"offline": True, "pass": True}
            self._mark_completed("VERIFY_TWO_PLANE_SETUP")
            self._transition("GENERATE_CALIBRATION_SEQUENCE")
            return
        w, h = self.cfg.proj_w, self.cfg.proj_h
        report: dict[str, Any] = {"checks": {}}
        black, _ = self._project_and_capture(make_black(h, w), "black")
        white, _ = self._project_and_capture(make_white(h, w), "white")
        border, _ = self._project_and_capture(make_border(h, w), "border")
        corners, _ = self._project_and_capture(make_corner_markers(h, w), "corners")
        # stripes
        vstripe = np.zeros((h, w, 3), dtype=np.uint8)
        vstripe[:, ::40] = 255
        hstripe = np.zeros((h, w, 3), dtype=np.uint8)
        hstripe[::40, :] = 255
        self._project_and_capture(vstripe, "vstripe")
        self._project_and_capture(hstripe, "hstripe")

        gb = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
        gw = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
        # Oblique / partial-FOV: score the *projected* region (white−black), not the
        # full camera frame mean (dark surround would falsely fail contrast).
        pre = evaluate_projection_preflight(gb, gw)
        report["checks"].update(pre["checks"])
        report["checks"]["blur_white"] = blur_score(gw)
        report["checks"]["contrast_score_white"] = contrast_score(gw)
        report["checks"]["framebuffer"] = [w, h]
        report["checks"]["external_projector"] = True
        # Do not reject on camera height/offset
        report["pass"] = bool(pre["pass"])
        (self.run_dir / "preflight" / "preflight_report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        self.state["data"]["preflight"] = report
        if not report["pass"]:
            self.write_physical_setup_request()
            self._transition(
                "USER_ACTION_REQUIRED",
                f"preflight failed: {json.dumps(report['checks'])}",
            )
            return
        self._mark_completed("VERIFY_TWO_PLANE_SETUP")
        self._transition("GENERATE_CALIBRATION_SEQUENCE")

    def step_generate_calibration_sequence(self) -> None:
        catalog = generate_pattern_set(
            self.cfg.proj_w, self.cfg.proj_h, self.run_dir / "projector_patterns"
        )
        self.state["data"]["pattern_catalog"] = catalog
        self._mark_completed("GENERATE_CALIBRATION_SEQUENCE")
        self._transition("PROJECT_CALIBRATION_SEQUENCE")

    def step_project_and_capture_calibration(self) -> None:
        """Combined project+capture loop with retries (covers PROJECT_* and CAPTURE_*)."""
        if self.cfg.offline:
            # Expect observations already under correspondences/aggregated_obs.json
            agg_path = self.run_dir / "correspondences" / "aggregated_obs.json"
            if not agg_path.exists():
                raise RuntimeError("offline mode requires correspondences/aggregated_obs.json")
            self._mark_completed("PROJECT_CALIBRATION_SEQUENCE")
            self._mark_completed("CAPTURE_CALIBRATION_SEQUENCE")
            self._transition("DETECT_MULTI_FRAME_CHARUCO")
            return

        catalog = self.state["data"]["pattern_catalog"]
        obs_blocks = []
        for entry in catalog:
            img = cv2.imread(entry["path"])
            ids_doc = json.loads(Path(entry["ids_path"]).read_text())
            known = ids_doc["known_proj_by_key"]
            pidx = int(entry["pattern_index"])
            ok = False
            last_meta = {}
            for attempt in range(self.cfg.max_capture_retries):
                assert self.projector and self.camera
                self.projector.show_image(img)
                time.sleep(self.cfg.settle_s * (1.0 + 0.25 * attempt))
                cap = self.run_dir / "camera_captures_raw" / f"calib_{pidx:02d}_try{attempt}.png"
                meta = self.camera.capture_frame(
                    cap, settle_s=0.35, flush_n=20 + 5 * attempt, expect_texture=True
                )
                frame = cv2.imread(str(cap))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if blur_score(gray) < 20 or contrast_score(gray) < 15:
                    rej = self.run_dir / "camera_captures_rejected" / cap.name
                    cv2.imwrite(str(rej), frame)
                    last_meta = {"reject": "blur_or_contrast", "metrics": meta}
                    continue
                obs, det_meta = extract_observations_from_capture(pidx, known, frame)
                last_meta = det_meta
                if det_meta.get("valid"):
                    obs_blocks.append(obs)
                    (self.run_dir / "correspondences" / f"obs_pattern_{pidx:02d}.json").write_text(
                        json.dumps(obs, indent=2)
                    )
                    ok = True
                    break
                rej = self.run_dir / "camera_captures_rejected" / cap.name
                cv2.imwrite(str(rej), frame)
            if not ok:
                self._log(f"pattern {pidx} failed after retries: {last_meta}")
        if not obs_blocks:
            self.state["data"]["blocker"] = "no_valid_calibration_captures"
            self._transition("ACCEPTANCE_FAILED", "no valid multi-frame captures")
            return
        agg = aggregate_observations(obs_blocks)
        (self.run_dir / "correspondences" / "aggregated_obs.json").write_text(
            json.dumps(agg, indent=2)
        )
        self.state["data"]["n_observations"] = len(agg)
        self._mark_completed("PROJECT_CALIBRATION_SEQUENCE")
        self._mark_completed("CAPTURE_CALIBRATION_SEQUENCE")
        self._transition("DETECT_MULTI_FRAME_CHARUCO")

    def step_detect_and_fit(self) -> None:
        agg = json.loads((self.run_dir / "correspondences" / "aggregated_obs.json").read_text())
        keys, pts_p, pts_c = observations_to_arrays(agg)
        (self.run_dir / "correspondences" / "observation_keys.json").write_text(
            json.dumps(keys, indent=2)
        )
        # FIT_SINGLE_H
        H1, err1, stats1 = fit_single_homography(pts_p, pts_c)
        single_path = self.run_dir / "model_fit"
        np.save(single_path / "H_cam_from_proj_single.npy", H1)
        (single_path / "single_H_stats.json").write_text(
            json.dumps({"stats": stats1, "median": float(np.median(err1)), "p95": float(np.percentile(err1, 95))}, indent=2)
        )
        self._mark_completed("DETECT_MULTI_FRAME_CHARUCO")
        self._mark_completed("FIT_SINGLE_H")
        self._transition("FIT_TWO_H")

        # Active/excluded plane preflight: peel ceiling before wall A/B fit
        self._transition("DETECT_ACTIVE_AND_EXCLUDED_PLANES")
        cls = classify_active_and_excluded_planes(pts_p, pts_c, ids=keys)
        (single_path / "plane_classification_preflight.json").write_text(
            json.dumps(
                {
                    k: (v if not isinstance(v, np.ndarray) else v.tolist())
                    for k, v in cls.items()
                    if k
                    not in (
                        "H_cam_from_proj_plane_A",
                        "H_cam_from_proj_plane_B",
                        "H_cam_from_proj_ceiling",
                    )
                },
                indent=2,
                default=str,
            )
        )
        self.state["data"]["plane_classification"] = {
            "model": cls.get("model"),
            "ceiling_detected": cls.get("ceiling_detected"),
            "valid_active": cls.get("valid_active"),
            "reason": cls.get("reason"),
            "n_ceiling": cls.get("n_ceiling"),
        }
        self._mark_completed("DETECT_ACTIVE_AND_EXCLUDED_PLANES")

        if (
            cls.get("valid_active")
            and cls.get("H_cam_from_proj_plane_A") is not None
            and cls.get("H_cam_from_proj_plane_B") is not None
            and cls.get("active_walls", 0) >= 2
        ):
            # Prefer ceiling-aware wall solution; strip ceiling from seam labels
            labels_full = np.array(cls["labels"], dtype=np.int32)
            wall = (labels_full == 0) | (labels_full == 1)
            Ha, Hb = cls["H_cam_from_proj_plane_A"], cls["H_cam_from_proj_plane_B"]
            fit = {
                "model": "two_plane",
                "valid": True,
                "reason": cls.get("reason") or "ok_ceiling_aware",
                "labels": np.where(wall, labels_full, -1).tolist(),
                "assignment": {
                    keys[i]: ("A" if labels_full[i] == 0 else "B" if labels_full[i] == 1 else "outlier")
                    for i in range(len(keys))
                    if labels_full[i] in (0, 1)
                },
                "H_cam_from_proj_plane_A": Ha,
                "H_cam_from_proj_plane_B": Hb,
                "comparison": {
                    "ceiling_aware": True,
                    "ceiling_detected": cls.get("ceiling_detected"),
                    "n_ceiling": cls.get("n_ceiling"),
                },
                "ceiling_labels": labels_full.tolist(),
            }
            # Persist only wall-support points for subsequent seam (filter outliers/ceiling)
            wall_idx = np.where(wall)[0]
            if wall_idx.size >= 2 * MIN_POINTS_PER_PLANE:
                pts_p_w, pts_c_w = pts_p[wall_idx], pts_c[wall_idx]
                keys_w = [keys[i] for i in wall_idx]
                labels_w = labels_full[wall_idx]
                seam_tmp = estimate_straight_seam(
                    pts_p_w, labels_w, self.cfg.proj_w, self.cfg.proj_h, Ha, Hb
                )
                excl = seam_exclusion_mask(seam_tmp, keys_w, pts_p_w)
                keep_w = np.array(excl["keep_mask"], dtype=bool)
                ma_k = keep_w & (labels_w == 0)
                mb_k = keep_w & (labels_w == 1)
                if ma_k.sum() >= MIN_POINTS_PER_PLANE and mb_k.sum() >= MIN_POINTS_PER_PLANE:
                    Ha2, _, _ = fit_single_homography(pts_p_w[ma_k], pts_c_w[ma_k])
                    Hb2, _, _ = fit_single_homography(pts_p_w[mb_k], pts_c_w[mb_k])
                    fit["H_cam_from_proj_plane_A"] = Ha2
                    fit["H_cam_from_proj_plane_B"] = Hb2
                (single_path / "seam_exclusion_band.json").write_text(json.dumps(excl, indent=2))
            # Store full classification labels (incl. ceiling) for exclusion stage
            self.state["data"]["ceiling_labels"] = labels_full.tolist()
        else:
            fit = fit_two_homographies(pts_p, pts_c, ids=keys)
            # Seam exclusion band: refit with band excluded if two-plane
            if fit.get("model") == "two_plane" and fit.get("valid"):
                labels = np.array(fit["labels"], dtype=np.int32)
                Ha, Hb = fit["H_cam_from_proj_plane_A"], fit["H_cam_from_proj_plane_B"]
                seam_tmp = estimate_straight_seam(pts_p, labels, self.cfg.proj_w, self.cfg.proj_h, Ha, Hb)
                excl = seam_exclusion_mask(seam_tmp, keys, pts_p)
                keep = np.array(excl["keep_mask"], dtype=bool)
                if keep.sum() >= 2 * MIN_POINTS_PER_PLANE:
                    fit2 = fit_two_homographies(
                        pts_p[keep], pts_c[keep], ids=[keys[i] for i in np.where(keep)[0]]
                    )
                    if fit2.get("valid") and fit2.get("model") == "two_plane":
                        Ha, Hb = fit2["H_cam_from_proj_plane_A"], fit2["H_cam_from_proj_plane_B"]
                        err_a = np.linalg.norm(apply_H(Ha, pts_p) - pts_c, axis=1)
                        err_b = np.linalg.norm(apply_H(Hb, pts_p) - pts_c, axis=1)
                        labels = (err_b < err_a).astype(np.int32)
                        from .two_plane_observations import canonicalize_plane_labels

                        labels = canonicalize_plane_labels(pts_p, labels)
                        if (labels == 0).sum() >= MIN_POINTS_PER_PLANE and (
                            labels == 1
                        ).sum() >= MIN_POINTS_PER_PLANE:
                            fit = fit2
                            fit["labels"] = labels.tolist()
                            fit["assignment"] = {
                                keys[i]: ("A" if labels[i] == 0 else "B") for i in range(len(keys))
                            }
                            Ha, _, _ = fit_single_homography(pts_p[labels == 0], pts_c[labels == 0])
                            Hb, _, _ = fit_single_homography(pts_p[labels == 1], pts_c[labels == 1])
                            fit["H_cam_from_proj_plane_A"] = Ha
                            fit["H_cam_from_proj_plane_B"] = Hb
                (single_path / "seam_exclusion_band.json").write_text(json.dumps(excl, indent=2))
            self.state["data"]["ceiling_labels"] = None

        # Persist
        (single_path / "single_vs_two_model_comparison.json").write_text(
            json.dumps(fit.get("comparison", {}), indent=2, default=str)
        )
        (single_path / "plane_assignment_by_observation.json").write_text(
            json.dumps(fit.get("assignment", {}), indent=2)
        )
        assign_pat = {}
        for k, lab in (fit.get("assignment") or {}).items():
            assign_pat[k] = lab
        (single_path / "plane_assignment_by_pattern_and_id.json").write_text(
            json.dumps(assign_pat, indent=2)
        )
        if fit.get("H_cam_from_proj_plane_A") is not None:
            np.save(single_path / "H_cam_from_proj_plane_A.npy", fit["H_cam_from_proj_plane_A"])
            np.save(single_path / "H_cam_from_proj_plane_B.npy", fit["H_cam_from_proj_plane_B"])
        report_md = [
            "# Model selection report",
            "",
            f"- model: `{fit.get('model')}`",
            f"- valid: `{fit.get('valid')}`",
            f"- reason: {fit.get('reason')}",
            f"- n observations: {len(keys)}",
            "",
            "```json",
            json.dumps(fit.get("comparison", {}), indent=2, default=str),
            "```",
        ]
        (single_path / "model_selection_report.md").write_text("\n".join(report_md) + "\n")
        # residual overlays
        self._save_residual_overlay(pts_p, pts_c, H1, None, single_path / "residual_vector_overlay_single_H.png")
        if fit.get("H_cam_from_proj_plane_A") is not None:
            labels = np.array(fit["labels"], dtype=np.int32)
            pred = np.where(
                labels[:, None] == 0,
                apply_H(fit["H_cam_from_proj_plane_A"], pts_p),
                apply_H(fit["H_cam_from_proj_plane_B"], pts_p),
            )
            self._save_residual_vectors(pts_c, pred, single_path / "residual_vector_overlay_two_H.png")

        self.state["data"]["fit"] = {
            "model": fit.get("model"),
            "valid": fit.get("valid"),
            "reason": fit.get("reason"),
            "n": len(keys),
        }
        self._mark_completed("FIT_TWO_H")
        self._transition("VALIDATE_MODEL_SELECTION")

        if fit.get("model") != "two_plane" or not fit.get("valid"):
            # Recovery: if offline/synthetic fixture forces one plane, fail; physical may need stronger fold
            self.state["data"]["blocker"] = f"model_selection:{fit.get('model')}:{fit.get('reason')}"
            if self.cfg.offline:
                self._transition("ACCEPTANCE_FAILED", "two-plane model not selected")
            else:
                self.write_physical_setup_request()
                self._transition(
                    "USER_ACTION_REQUIRED",
                    "one-H selected or two-H rejected — strengthen fold / coverage",
                )
            return
        self._mark_completed("VALIDATE_MODEL_SELECTION")
        self.state["data"]["keys"] = keys
        self.state["data"]["labels"] = fit["labels"]
        self._transition("ESTIMATE_SEAM")

    def _save_residual_overlay(self, pts_p, pts_c, H, labels, path: Path) -> None:
        pred = apply_H(H, pts_p)
        self._save_residual_vectors(pts_c, pred, path)

    def _save_residual_vectors(self, pts_c, pred, path: Path) -> None:
        # Simple scatter on blank canvas
        xs = np.concatenate([pts_c[:, 0], pred[:, 0]])
        ys = np.concatenate([pts_c[:, 1], pred[:, 1]])
        w = int(max(640, xs.max() + 40))
        h = int(max(480, ys.max() + 40))
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for a, b in zip(pts_c, pred):
            cv2.arrowedLine(
                canvas,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                (0, 255, 255),
                1,
                tipLength=0.2,
            )
        cv2.imwrite(str(path), canvas)

    def step_estimate_seam(self) -> None:
        keys = self.state["data"]["keys"]
        agg = json.loads((self.run_dir / "correspondences" / "aggregated_obs.json").read_text())
        _, pts_p, pts_c = observations_to_arrays(agg)
        labels = np.array(self.state["data"]["labels"], dtype=np.int32)
        # Exclude ceiling/outlier labels from seam geometry
        wall = (labels == 0) | (labels == 1)
        if wall.sum() < 2 * MIN_POINTS_PER_PLANE:
            self._transition("ACCEPTANCE_FAILED", "insufficient_wall_labels_for_seam")
            return
        pts_p_w, pts_c_w = pts_p[wall], pts_c[wall]
        labels_w = labels[wall]
        keys_w = [keys[i] for i in np.where(wall)[0]]
        Ha = np.load(self.run_dir / "model_fit" / "H_cam_from_proj_plane_A.npy")
        Hb = np.load(self.run_dir / "model_fit" / "H_cam_from_proj_plane_B.npy")
        seam = estimate_straight_seam(pts_p_w, labels_w, self.cfg.proj_w, self.cfg.proj_h, Ha, Hb)
        boot = bootstrap_seam_stability(
            pts_p_w, labels_w, Ha, Hb, self.cfg.proj_w, self.cfg.proj_h
        )
        excl = seam_exclusion_mask(seam, keys_w, pts_p_w)
        mask_a, mask_b, viz = build_plane_masks(seam, self.cfg.proj_w, self.cfg.proj_h)
        seam_dir = self.run_dir / "seam"
        (seam_dir / "seam_model.json").write_text(json.dumps(seam, indent=2))
        (seam_dir / "seam_bootstrap_stability.json").write_text(json.dumps(boot, indent=2))
        (seam_dir / "seam_exclusion_band.json").write_text(json.dumps(excl, indent=2))
        cv2.imwrite(str(seam_dir / "plane_mask_A.png"), mask_a)
        cv2.imwrite(str(seam_dir / "plane_mask_B.png"), mask_b)
        union = np.maximum(mask_a, mask_b)
        overlap = ((mask_a > 0) & (mask_b > 0)).astype(np.uint8) * 255
        cv2.imwrite(str(seam_dir / "mask_union.png"), union)
        cv2.imwrite(str(seam_dir / "mask_overlap.png"), overlap)
        cv2.imwrite(str(seam_dir / "seam_overlay_projector.png"), viz)
        # camera overlay
        cam_viz = np.zeros((max(480, int(pts_c[:, 1].max()) + 40), max(640, int(pts_c[:, 0].max()) + 40), 3), dtype=np.uint8)
        for p, lab in zip(pts_c, labels):
            if lab == 0:
                color = (0, 255, 0)
            elif lab == 1:
                color = (0, 128, 255)
            elif lab == 2:
                color = (0, 0, 255)  # ceiling / excluded
            else:
                color = (80, 80, 80)
            cv2.circle(cam_viz, (int(p[0]), int(p[1])), 4, color, -1)
        cv2.imwrite(str(seam_dir / "seam_overlay_camera.png"), cam_viz)
        seam_m = seam_mismatch_in_camera(pts_p_w, labels_w, Ha, Hb, seam)
        self.state["data"]["seam_metrics"] = seam_m
        if not boot.get("stable", False):
            self._log("seam bootstrap unstable — continuing with best estimate")
        self._mark_completed("ESTIMATE_SEAM")
        self._transition("VERIFY_SEAM")

    def step_verify_seam(self) -> None:
        seam = json.loads((self.run_dir / "seam" / "seam_model.json").read_text())
        seam_cands = []
        sc_path = self.run_dir / "architecture" / "seam_candidates.json"
        if sc_path.exists():
            seam_cands = json.loads(sc_path.read_text())
        cam_w = None
        es = self.state["data"].get("empty_scene") or {}
        if es.get("resolution"):
            cam_w = es["resolution"][0]
        elif (es.get("meta") or {}).get("resolution"):
            cam_w = es["meta"]["resolution"][0]
        ver = verify_seam_with_architecture_prior(seam, seam_cands, self.cfg.proj_w, cam_w=cam_w)
        (self.run_dir / "seam" / "seam_architecture_verification.json").write_text(
            json.dumps(ver, indent=2)
        )
        self.state["data"]["seam_verification"] = ver
        # Correspondence seam remains authoritative even if prior disagrees
        self._mark_completed("VERIFY_SEAM")
        self._mark_completed("BUILD_MASKS")
        self._transition("ESTIMATE_WALL_CEILING_BOUNDARY")

    def step_oblique_masks_and_video_region(self) -> None:
        """Ceiling boundary, usable mask, max video rect, playback remap bundle."""
        Ha = np.load(self.run_dir / "model_fit" / "H_cam_from_proj_plane_A.npy")
        Hb = np.load(self.run_dir / "model_fit" / "H_cam_from_proj_plane_B.npy")
        seam = json.loads((self.run_dir / "seam" / "seam_model.json").read_text())
        agg = json.loads((self.run_dir / "correspondences" / "aggregated_obs.json").read_text())
        keys = self.state["data"]["keys"]
        _, pts_p, pts_c = observations_to_arrays(agg)
        cam_w = int(max(1280, pts_c[:, 0].max() + 80))
        cam_h = int(max(720, pts_c[:, 1].max() + 80))
        es = self.state["data"].get("empty_scene") or {}
        if es.get("resolution"):
            cam_w, cam_h = int(es["resolution"][0]), int(es["resolution"][1])
        elif (es.get("meta") or {}).get("resolution"):
            cam_w, cam_h = [int(x) for x in es["meta"]["resolution"][:2]]

        labels = np.array(self.state["data"]["labels"], dtype=np.int32)
        ceil_labels = self.state["data"].get("ceiling_labels")
        if ceil_labels is not None:
            labels_full = np.array(ceil_labels, dtype=np.int32)
        else:
            labels_full = labels.copy()

        arch_lines = []
        arch_path = self.run_dir / "architecture" / "architecture_report.json"
        if arch_path.exists():
            arch = json.loads(arch_path.read_text())
            arch_lines = arch.get("horizontal_lines") or arch.get("lines_horizontal") or []

        boundary = estimate_wall_ceiling_boundary(
            pts_c, labels_full, cam_w, cam_h, arch_horizontal_lines=arch_lines
        )
        (self.run_dir / "excluded_geometry" / "wall_ceiling_boundary.json").write_text(
            json.dumps(boundary, indent=2)
        )
        self._mark_completed("ESTIMATE_WALL_CEILING_BOUNDARY")
        self._transition("BUILD_USABLE_PROJECTION_MASK")

        target_poly = None
        tr_path = self.run_dir / "architecture" / "target_region.json"
        if tr_path.exists():
            tr = json.loads(tr_path.read_text())
            target_poly = tr.get("polygon") or tr.get("camera_polygon")

        masks = build_usable_and_exclusion_masks(
            seam,
            boundary,
            Ha,
            Hb,
            self.cfg.proj_w,
            self.cfg.proj_h,
            cam_w,
            cam_h,
            margin_px=8.0,
            target_region_cam=target_poly,
        )
        cls_meta = self.state["data"].get("plane_classification") or {}
        save_exclusion_artifacts(
            self.run_dir / "excluded_geometry",
            masks,
            boundary,
            {
                **cls_meta,
                "labels": labels_full.tolist(),
                "H_cam_from_proj_plane_A": Ha,
                "H_cam_from_proj_plane_B": Hb,
                "note": "Ceiling is excluded from content; not a third artistic plane",
            },
        )
        # Keep seam wall masks authoritative for piecewise compositing
        mask_a = cv2.imread(str(self.run_dir / "seam" / "plane_mask_A.png"), cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(str(self.run_dir / "seam" / "plane_mask_B.png"), cv2.IMREAD_GRAYSCALE)
        if mask_a is not None:
            masks["mask_plane_A"] = mask_a
        if mask_b is not None:
            masks["mask_plane_B"] = mask_b
        self._mark_completed("BUILD_USABLE_PROJECTION_MASK")
        self._mark_completed("BUILD_FINAL_EXCLUSION_MASK")
        self._transition("OPTIMISE_MAXIMUM_VIDEO_REGION")

        aspect = parse_aspect("16:9")
        region = maximum_inscribed_rectangle(
            masks["usable_mask_camera"],
            aspect=aspect,
            alignment="camera",
            step=6,
            safety_erode_px=2,
        )
        save_video_region(self.run_dir / "video_region", region)
        self.state["data"]["video_region"] = {
            "valid": region.get("valid"),
            "area_px": region.get("area_px"),
            "pct_usable": region.get("pct_usable_mask_occupied"),
            "claim": region.get("claim"),
        }
        self._mark_completed("OPTIMISE_MAXIMUM_VIDEO_REGION")
        self._mark_completed("FIT_ACTIVE_WALL_HOMOGRAPHIES")
        self._transition("BUILD_PIECEWISE_VIDEO_WARP")

        if region.get("valid"):
            meta = prepare_playback_calibration(
                self.run_dir,
                Ha,
                Hb,
                seam,
                masks,
                region,
                source_w=1280,
                source_h=720,
                proj_w=self.cfg.proj_w,
                proj_h=self.cfg.proj_h,
                video_fit="contain",
                aspect=aspect,
            )
            diag_path = self.run_dir / "diagnostic_video" / "diagnostic_source.mp4"
            write_diagnostic_video(diag_path, w=1280, h=720, n_frames=60, fps=30)
            (self.run_dir / "diagnostic_video" / "playback_meta.json").write_text(
                json.dumps(meta, indent=2, default=str)
            )
            self.state["data"]["video_playback"] = {
                "ready": True,
                "ceiling_leakage": meta.get("ceiling_leakage_selfcheck"),
            }
        else:
            self._log("maximum video region invalid — skipping playback bundle")
            self.state["data"]["video_playback"] = {"ready": False, "reason": region.get("failure_reason")}

        self._mark_completed("BUILD_PIECEWISE_VIDEO_WARP")
        self._mark_completed("SAVE_VIDEO_PLAYBACK_CALIBRATION")
        self._mark_completed("PLAY_DIAGNOSTIC_VIDEO")  # artifact generated; physical play via CLI
        self._transition("GENERATE_PIECEWISE_PREWARP")

    def step_generate_prewarp(self) -> None:
        Ha = np.load(self.run_dir / "model_fit" / "H_cam_from_proj_plane_A.npy")
        Hb = np.load(self.run_dir / "model_fit" / "H_cam_from_proj_plane_B.npy")
        mask_a = cv2.imread(str(self.run_dir / "seam" / "plane_mask_A.png"), cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(str(self.run_dir / "seam" / "plane_mask_B.png"), cv2.IMREAD_GRAYSCALE)
        seam = json.loads((self.run_dir / "seam" / "seam_model.json").read_text())
        agg = json.loads((self.run_dir / "correspondences" / "aggregated_obs.json").read_text())
        keys = self.state["data"]["keys"]
        labels = np.array(self.state["data"]["labels"], dtype=np.int32)
        labels_by_key = {keys[i]: ("A" if labels[i] == 0 else "B") for i in range(len(keys))}
        source_by_key = {k: agg[k]["proj"] for k in keys}
        # Camera size from observations
        _, _, pts_c = observations_to_arrays(agg)
        cam_w = int(max(1280, pts_c[:, 0].max() + 80))
        cam_h = int(max(720, pts_c[:, 1].max() + 80))
        desired = define_desired_target_two_plane(
            Ha,
            Hb,
            mask_a,
            mask_b,
            seam,
            self.cfg.proj_w,
            self.cfg.proj_h,
            cam_w,
            cam_h,
            source_by_key,
            labels_by_key,
        )
        dt_dir = self.run_dir / "desired_target"
        (dt_dir / "desired_target_two_plane.json").write_text(json.dumps(desired, indent=2))
        H_des = np.array(desired["H_desired_cam_from_source"], dtype=np.float64)
        np.save(dt_dir / "H_desired_cam_from_source.npy", H_des)

        # Validation content: charuco + seam-crossing lines
        content, proj_ids, *_ = generate_charuco_image(self.cfg.proj_w, self.cfg.proj_h)
        val = make_validation_target(self.cfg.proj_h, self.cfg.proj_w)
        # blend
        content = cv2.addWeighted(content, 0.75, val, 0.35, 0)
        # seam crossing lines in source
        p0 = np.array(seam["endpoint_0"], dtype=np.int32)
        p1 = np.array(seam["endpoint_1"], dtype=np.int32)
        for t in np.linspace(0.1, 0.9, 5):
            mid = ((1 - t) * p0 + t * p1).astype(int)
            cv2.line(content, (0, int(mid[1])), (self.cfg.proj_w - 1, int(mid[1])), (0, 255, 255), 2)
            cv2.line(content, (int(mid[0]), 0), (int(mid[0]), self.cfg.proj_h - 1), (255, 255, 0), 2)

        H_fwd_a = forward_H_proj_from_source(Ha, H_des)
        H_fwd_b = forward_H_proj_from_source(Hb, H_des)
        pre_a = prewarp_from_forward_H(content, H_fwd_a, self.cfg.proj_w, self.cfg.proj_h)
        pre_b = prewarp_from_forward_H(content, H_fwd_b, self.cfg.proj_w, self.cfg.proj_h)
        pre = build_piecewise_from_forwards(
            content, H_fwd_a, H_fwd_b, mask_a, mask_b, self.cfg.proj_w, self.cfg.proj_h
        )
        pw = self.run_dir / "prewarps"
        np.save(pw / "H_proj_from_source_plane_A.npy", H_fwd_a)
        np.save(pw / "H_proj_from_source_plane_B.npy", H_fwd_b)
        (pw / "forward_matrix_metadata_A.json").write_text(
            json.dumps(
                {
                    "matrix": "H_proj_from_source_plane_A",
                    "projector_resolution": [self.cfg.proj_w, self.cfg.proj_h],
                    "matches_framebuffer": True,
                },
                indent=2,
            )
        )
        (pw / "forward_matrix_metadata_B.json").write_text(
            json.dumps(
                {
                    "matrix": "H_proj_from_source_plane_B",
                    "projector_resolution": [self.cfg.proj_w, self.cfg.proj_h],
                    "matches_framebuffer": True,
                },
                indent=2,
            )
        )
        cv2.imwrite(str(pw / "prewarp_plane_A.png"), pre_a)
        cv2.imwrite(str(pw / "prewarp_plane_B.png"), pre_b)
        cv2.imwrite(str(pw / "prewarp_piecewise.png"), pre)
        cv2.imwrite(str(self.run_dir / "projected_validation" / "source_validation.png"), content)
        cv2.imwrite(str(self.run_dir / "projected_validation" / "uncorrected_source.png"), content)
        # Save proj ids for validation under observation keys pattern 0
        known = {observation_key(0, cid): [float(xy[0]), float(xy[1])] for cid, xy in proj_ids.items()}
        (self.run_dir / "projected_validation" / "validation_proj_ids.json").write_text(
            json.dumps(known, indent=2)
        )
        preserve_best(pw, H_fwd_a, H_fwd_b, pre, {"stage": "initial"})
        self.state["data"]["desired"] = {"cam_w": cam_w, "cam_h": cam_h}
        self._mark_completed("GENERATE_PIECEWISE_PREWARP")
        self._transition("PROJECT_UNCORRECTED_VALIDATION")

    def _burst_obs(self, stem: str, known: dict) -> tuple[dict[str, list[float]], dict]:
        out_dir = self.run_dir / "burst_variability"
        frames = []
        for i in range(3):
            if self.cfg.offline:
                path = self.run_dir / "captured_validation" / f"{stem}_burst_{i:02d}.png"
                if not path.exists():
                    path = self.run_dir / "captured_validation" / f"{stem}.png"
                frame = cv2.imread(str(path))
                meta = {"offline": True}
            else:
                assert self.camera
                path = out_dir / f"{stem}_burst_{i:02d}.png"
                meta = self.camera.capture_frame(path, settle_s=0.35, flush_n=20, expect_texture=True)
                frame = cv2.imread(str(path))
                cv2.imwrite(str(self.run_dir / "captured_validation" / path.name), frame)
            obs, _ = extract_observations_from_capture(0, known, frame, min_corners=8)
            frames.append(obs)
        # median per key
        all_keys = sorted(set().union(*[set(f) for f in frames]))
        med = {}
        jitter = []
        for k in all_keys:
            pts = [f[k]["cam"] for f in frames if k in f]
            if not pts:
                continue
            arr = np.array(pts, dtype=np.float64)
            m = np.median(arr, axis=0)
            med[k] = [float(m[0]), float(m[1])]
            if len(arr) >= 2:
                jitter.append(float(np.linalg.norm(arr - m, axis=1).max()))
        report = {
            "aggregate_median_jitter_px": float(np.median(jitter)) if jitter else 0.0,
            "aggregate_p95_jitter_px": float(np.percentile(jitter, 95)) if jitter else 0.0,
            "n_keys": len(med),
        }
        (out_dir / f"{stem}_burst_variability.json").write_text(json.dumps(report, indent=2))
        return med, report

    def step_validation_capture_measure(self) -> None:
        known = json.loads(
            (self.run_dir / "projected_validation" / "validation_proj_ids.json").read_text()
        )
        content = cv2.imread(str(self.run_dir / "projected_validation" / "uncorrected_source.png"))
        pre = cv2.imread(str(self.run_dir / "prewarps" / "prewarp_piecewise.png"))

        if not self.cfg.offline:
            assert self.projector
            self.projector.show_image(content)
            time.sleep(self.cfg.settle_s)
        self._mark_completed("PROJECT_UNCORRECTED_VALIDATION")
        self._transition("CAPTURE_UNCORRECTED_VALIDATION")
        unc_obs, unc_jit = self._burst_obs("uncorrected", known)
        (self.run_dir / "captured_validation" / "uncorrected_median_obs.json").write_text(
            json.dumps(unc_obs, indent=2)
        )
        self._mark_completed("CAPTURE_UNCORRECTED_VALIDATION")

        if not self.cfg.offline:
            assert self.projector
            self.projector.show_image(pre)
            time.sleep(self.cfg.settle_s)
        self._transition("PROJECT_CORRECTED_VALIDATION")
        self._mark_completed("PROJECT_CORRECTED_VALIDATION")
        self._transition("CAPTURE_CORRECTED_VALIDATION")
        cor_obs, cor_jit = self._burst_obs("corrected", known)
        (self.run_dir / "captured_validation" / "corrected_median_obs.json").write_text(
            json.dumps(cor_obs, indent=2)
        )
        self._mark_completed("CAPTURE_CORRECTED_VALIDATION")
        self._transition("MEASURE_TWO_PLANE_RESULT")

        desired = json.loads(
            (self.run_dir / "desired_target" / "desired_target_two_plane.json").read_text()
        )
        labels_by_key = desired["labels_by_key"]
        # Remap validation keys (pattern 0) — labels from nearest calib proj point plane
        keys_cal = self.state["data"]["keys"]
        labs = np.array(self.state["data"]["labels"])
        agg = json.loads((self.run_dir / "correspondences" / "aggregated_obs.json").read_text())
        pts_cal = np.array([agg[k]["proj"] for k in keys_cal], dtype=np.float64)

        def _label_for_key(k: str) -> str:
            if k in labels_by_key:
                return labels_by_key[k]
            # validation pattern-0 keys: assign by projector coordinate nearest calib label
            if k not in known:
                return "A"
            p = np.array(known[k], dtype=np.float64)
            d = np.linalg.norm(pts_cal - p, axis=1)
            return "A" if labs[int(d.argmin())] == 0 else "B"

        lab_unc = {k: _label_for_key(k) for k in unc_obs}
        lab_cor = {k: _label_for_key(k) for k in cor_obs}
        # Extend desired points for validation keys via H_desired
        H_des = np.array(desired["H_desired_cam_from_source"], dtype=np.float64)
        for block in (unc_obs, cor_obs):
            for k in block:
                if k not in desired["desired_point_by_key"] and k in known:
                    p = apply_H(H_des, np.array([known[k]], dtype=np.float64))[0]
                    desired["desired_point_by_key"][k] = [float(p[0]), float(p[1])]

        expected = sorted(set(known.keys()) & set(desired["desired_point_by_key"].keys()))
        unc_m = measure_two_plane_against_desired(unc_obs, desired, lab_unc, expected_keys=expected)
        cor_m = measure_two_plane_against_desired(cor_obs, desired, lab_cor, expected_keys=expected)
        seam_geom = self.state["data"].get("seam_metrics")
        gate = two_plane_acceptance(unc_m, cor_m, seam_geom)
        metrics = {
            "uncorrected": unc_m,
            "corrected": cor_m,
            "acceptance": gate,
            "jitter_uncorrected": unc_jit,
            "jitter_corrected": cor_jit,
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        self.state["data"]["metrics"] = {
            "acceptance_pass": gate.get("acceptance_pass"),
            "cor_median": cor_m.get("target_median_err_px"),
            "cor_p95": cor_m.get("target_p95_err_px"),
        }
        self._mark_completed("MEASURE_TWO_PLANE_RESULT")

        if gate.get("absolute_gate_pass"):
            self._transition("FINAL_REPRODUCTION")
            return
        self._transition("DIAGNOSE_TWO_PLANE_RESIDUAL")

    def step_refine(self) -> None:
        self._mark_completed("DIAGNOSE_TWO_PLANE_RESIDUAL")
        self._transition("REFINE_TWO_PLANE")
        if self.cfg.offline and not (self.run_dir / "captured_validation" / "corrected.png").exists():
            # Offline unit tests may skip physical refine loop
            self._transition("ACCEPT_OR_ROLLBACK", "offline_skip_refine")
            self._finalize_failure_or_done()
            return

        desired = json.loads(
            (self.run_dir / "desired_target" / "desired_target_two_plane.json").read_text()
        )
        H_des = np.array(desired["H_desired_cam_from_source"], dtype=np.float64)
        H_fwd_a = np.load(self.run_dir / "prewarps" / "H_proj_from_source_plane_A.npy")
        H_fwd_b = np.load(self.run_dir / "prewarps" / "H_proj_from_source_plane_B.npy")
        mask_a = cv2.imread(str(self.run_dir / "seam" / "plane_mask_A.png"), cv2.IMREAD_GRAYSCALE)
        mask_b = cv2.imread(str(self.run_dir / "seam" / "plane_mask_B.png"), cv2.IMREAD_GRAYSCALE)
        content = cv2.imread(str(self.run_dir / "projected_validation" / "source_validation.png"))
        known = json.loads(
            (self.run_dir / "projected_validation" / "validation_proj_ids.json").read_text()
        )
        cor_obs = json.loads(
            (self.run_dir / "captured_validation" / "corrected_median_obs.json").read_text()
        )
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        best_metrics = metrics["corrected"]
        best_Ha, best_Hb = H_fwd_a.copy(), H_fwd_b.copy()
        best_pre = cv2.imread(str(self.run_dir / "prewarps" / "prewarp_piecewise.png"))
        jitter = float((metrics.get("jitter_corrected") or {}).get("aggregate_median_jitter_px") or 0.2)

        labels_by_key = {}
        keys_cal = self.state["data"]["keys"]
        labs = np.array(self.state["data"]["labels"])
        agg = json.loads((self.run_dir / "correspondences" / "aggregated_obs.json").read_text())
        pts_cal = np.array([agg[k]["proj"] for k in keys_cal], dtype=np.float64)

        def _lab(k: str) -> str:
            if k not in known:
                return "A"
            p = np.array(known[k], dtype=np.float64)
            d = np.linalg.norm(pts_cal - p, axis=1)
            return "A" if labs[int(d.argmin())] == 0 else "B"

        for k in cor_obs:
            labels_by_key[k] = _lab(k)
            if k not in desired["desired_point_by_key"] and k in known:
                p = apply_H(H_des, np.array([known[k]], dtype=np.float64))[0]
                desired["desired_point_by_key"][k] = [float(p[0]), float(p[1])]

        keys_a = [k for k in cor_obs if labels_by_key[k] == "A"]
        keys_b = [k for k in cor_obs if labels_by_key[k] == "B"]
        if len(keys_a) < 4 or len(keys_b) < 4:
            self._transition("ACCEPT_OR_ROLLBACK", "insufficient per-plane keys for refine")
            self._finalize_failure_or_done()
            return

        control = np.array(
            [[0, 0], [self.cfg.proj_w - 1, 0], [self.cfg.proj_w - 1, self.cfg.proj_h - 1], [0, self.cfg.proj_h - 1]],
            dtype=np.float64,
        )
        # denser control grid
        xs = np.linspace(0, self.cfg.proj_w - 1, 8)
        ys = np.linspace(0, self.cfg.proj_h - 1, 6)
        control = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

        for it in range(self.cfg.max_refine_iters):
            cands = propose_coupled_candidates(
                best_Ha,
                best_Hb,
                keys_a,
                keys_b,
                known,
                cor_obs,
                H_des,
                control,
            )
            accepted_any = False
            for cand in cands:
                pre = build_piecewise_from_forwards(
                    content, cand["H_fwd_a"], cand["H_fwd_b"], mask_a, mask_b, self.cfg.proj_w, self.cfg.proj_h
                )
                if not self.cfg.offline:
                    assert self.projector
                    self.projector.show_image(pre)
                    time.sleep(self.cfg.settle_s)
                new_obs, new_jit = self._burst_obs(f"refine_it{it}_s{cand['strength']}", known)
                lab = {k: _lab(k) for k in new_obs}
                for k in new_obs:
                    if k not in desired["desired_point_by_key"] and k in known:
                        p = apply_H(H_des, np.array([known[k]], dtype=np.float64))[0]
                        desired["desired_point_by_key"][k] = [float(p[0]), float(p[1])]
                expected = sorted(set(known.keys()) & set(desired["desired_point_by_key"].keys()))
                new_m = measure_two_plane_against_desired(new_obs, desired, lab, expected_keys=expected)
                ok, reason = coupled_accept(best_metrics, new_m, jitter)
                (self.run_dir / "refinement" / f"iter{it}_s{cand['strength']}.json").write_text(
                    json.dumps({"accept": ok, "reason": reason, "metrics": new_m}, indent=2, default=str)
                )
                if ok:
                    best_metrics = new_m
                    best_Ha, best_Hb = cand["H_fwd_a"], cand["H_fwd_b"]
                    best_pre = pre
                    cor_obs = new_obs
                    jitter = float(new_jit.get("aggregate_median_jitter_px") or jitter)
                    accepted_any = True
                    preserve_best(self.run_dir / "prewarps", best_Ha, best_Hb, best_pre, best_metrics)
                    break
            if not accepted_any:
                break
            gate = two_plane_acceptance(metrics["uncorrected"], best_metrics, self.state["data"].get("seam_metrics"))
            if gate.get("absolute_gate_pass"):
                break

        # Write best back
        np.save(self.run_dir / "prewarps" / "H_proj_from_source_plane_A.npy", best_Ha)
        np.save(self.run_dir / "prewarps" / "H_proj_from_source_plane_B.npy", best_Hb)
        cv2.imwrite(str(self.run_dir / "prewarps" / "prewarp_piecewise.png"), best_pre)
        metrics["corrected"] = best_metrics
        metrics["acceptance"] = two_plane_acceptance(
            metrics["uncorrected"], best_metrics, self.state["data"].get("seam_metrics")
        )
        (self.run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        self._mark_completed("REFINE_TWO_PLANE")
        self._transition("ACCEPT_OR_ROLLBACK")
        self._finalize_failure_or_done()

    def _finalize_failure_or_done(self) -> None:
        metrics = json.loads((self.run_dir / "metrics.json").read_text())
        gate = metrics.get("acceptance") or {}
        self._write_final_report(metrics)
        if gate.get("absolute_gate_pass"):
            self._mark_completed("ACCEPT_OR_ROLLBACK")
            self._transition("FINAL_REPRODUCTION")
            self._mark_completed("FINAL_REPRODUCTION")
            self._transition("DONE", "absolute gate pass")
        else:
            # Measured physical limit if we completed a physical attempt
            self.state["data"]["measured_limit"] = True
            self._mark_completed("ACCEPT_OR_ROLLBACK")
            self._transition("ACCEPTANCE_FAILED", gate.get("failure_reason") or "gate_failed")

    def _step_final_reproduction(self) -> None:
        self._mark_completed("FINAL_REPRODUCTION")
        self._transition("DONE", "absolute gate pass")

    def _write_final_report(self, metrics: dict) -> None:
        cor = metrics.get("corrected") or {}
        unc = metrics.get("uncorrected") or {}
        gate = metrics.get("acceptance") or {}
        lines = [
            "# FINAL_TWO_PLANE_REPORT",
            "",
            f"- run: `{self.run_dir}`",
            f"- model: `{self.state.get('data', {}).get('fit', {})}`",
            f"- acceptance_pass: **{gate.get('acceptance_pass')}**",
            f"- uncorrected median/p95: {unc.get('target_median_err_px')} / {unc.get('target_p95_err_px')}",
            f"- corrected median/p95: {cor.get('target_median_err_px')} / {cor.get('target_p95_err_px')}",
            f"- plane A: {cor.get('plane_A')}",
            f"- plane B: {cor.get('plane_B')}",
            f"- seam: {cor.get('seam') or self.state.get('data', {}).get('seam_metrics')}",
            "",
        ]
        (self.run_dir / "FINAL_TWO_PLANE_REPORT.md").write_text("\n".join(lines))
        (self.run_dir / "report.md").write_text("\n".join(lines))

    def run(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Resume after user confirms physical setup
        if (
            self.state.get("state") == "USER_ACTION_REQUIRED"
            and (self.cfg.setup_confirmed or self.cfg.offline)
        ):
            self._transition("DISCOVER_PROJECTOR", "setup confirmed — resume physical pipeline")
        terminal = {
            "DONE",
            "USER_ACTION_REQUIRED",
            "ACCEPTANCE_FAILED",
            "BLOCKED_EXTERNAL",
        }
        handlers = {
            "PREFLIGHT": self.step_preflight,
            "DISCOVER_PROJECTOR": self.step_discover_projector,
            "DISCOVER_CAMERA": self.step_discover_camera,
            "CAPTURE_EMPTY_SCENE": self.step_capture_empty_scene,
            "ANALYSE_ARCHITECTURE": self.step_analyse_architecture,
            "PROPOSE_TARGET_REGION": self.step_propose_target_region,
            "VERIFY_TWO_PLANE_SETUP": self.step_verify_two_plane_setup,
            "GENERATE_CALIBRATION_SEQUENCE": self.step_generate_calibration_sequence,
            "PROJECT_CALIBRATION_SEQUENCE": self.step_project_and_capture_calibration,
            "CAPTURE_CALIBRATION_SEQUENCE": self.step_project_and_capture_calibration,
            "DETECT_MULTI_FRAME_CHARUCO": self.step_detect_and_fit,
            "EXTRACT_CORRESPONDENCES": self.step_detect_and_fit,
            "FIT_SINGLE_H": self.step_detect_and_fit,
            "TEST_MULTI_PLANE": self.step_detect_and_fit,
            "FIT_TWO_H": self.step_detect_and_fit,
            "ASSIGN_SURFACES": self.step_detect_and_fit,
            "VALIDATE_MODEL_SELECTION": self.step_detect_and_fit,
            "ESTIMATE_SEAM": self.step_estimate_seam,
            "VERIFY_SEAM": self.step_verify_seam,
            "BUILD_MASKS": self.step_verify_seam,
            "ESTIMATE_WALL_CEILING_BOUNDARY": self.step_oblique_masks_and_video_region,
            "BUILD_USABLE_PROJECTION_MASK": self.step_oblique_masks_and_video_region,
            "OPTIMISE_MAXIMUM_VIDEO_REGION": self.step_oblique_masks_and_video_region,
            "FIT_ACTIVE_WALL_HOMOGRAPHIES": self.step_oblique_masks_and_video_region,
            "BUILD_PIECEWISE_VIDEO_WARP": self.step_oblique_masks_and_video_region,
            "BUILD_FINAL_EXCLUSION_MASK": self.step_oblique_masks_and_video_region,
            "SAVE_VIDEO_PLAYBACK_CALIBRATION": self.step_oblique_masks_and_video_region,
            "PLAY_DIAGNOSTIC_VIDEO": self.step_oblique_masks_and_video_region,
            "DEFINE_DESIRED_TARGET": self.step_generate_prewarp,
            "GENERATE_PIECEWISE_PREWARP": self.step_generate_prewarp,
            "PROJECT_UNCORRECTED_VALIDATION": self.step_validation_capture_measure,
            "CAPTURE_UNCORRECTED_VALIDATION": self.step_validation_capture_measure,
            "PROJECT_CORRECTED_VALIDATION": self.step_validation_capture_measure,
            "CAPTURE_CORRECTED_VALIDATION": self.step_validation_capture_measure,
            "MEASURE_TWO_PLANE_RESULT": self.step_validation_capture_measure,
            "DIAGNOSE_TWO_PLANE_RESIDUAL": self.step_refine,
            "REFINE_TWO_PLANE": self.step_refine,
            "ACCEPT_OR_ROLLBACK": self._finalize_failure_or_done,
            "FINAL_REPRODUCTION": self._step_final_reproduction,
        }
        try:
            while self.state["state"] not in terminal:
                st = self.state["state"]
                self._log(f"state={st}")
                fn = handlers.get(st)
                if fn is None:
                    self.state["data"]["blocker"] = f"unknown_state:{st}"
                    self._transition("BLOCKED_EXTERNAL", f"unknown state {st}")
                    break
                fn()
                # safety: if handler didn't transition, break
                if self.state["state"] == st and st not in terminal:
                    self._log(f"no transition from {st}")
                    break
        except Exception as e:
            self._log(f"ERROR: {e}\n{traceback.format_exc()}")
            self.state["data"]["blocker"] = str(e)
            self.state["data"]["traceback"] = traceback.format_exc()
            # Software errors are not external blockers
            self._transition("ACCEPTANCE_FAILED", f"software_error:{e}")
        finally:
            if self.projector is not None:
                try:
                    self.projector.shutdown()
                except Exception:
                    pass
        return {
            "state": self.state["state"],
            "run_dir": str(self.run_dir),
            "data": self.state.get("data", {}),
        }


def run_synthetic_mode(run_dir: Path, proj_w: int = 1920, proj_h: int = 1080) -> dict:
    from .synthetic_two_plane import run_synthetic_suite

    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_synthetic_suite(run_dir / "suite")
    (run_dir / "TWO_PLANE_SYNTHETIC_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    # Do not overwrite the immutable checkpoint run
    return summary
