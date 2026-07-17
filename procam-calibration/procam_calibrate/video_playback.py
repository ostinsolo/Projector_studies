"""Piecewise video remapping, diagnostic animation, and projector playback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import apply_H
from .homography_baseline import forward_H_proj_from_source
from .video_region import apply_video_fit


SUPPORTED_CODECS = "OpenCV VideoCapture: mp4/mov/avi (H.264/MPEG typically via system backends); BGR uint8 frames"


def build_piecewise_remap(
    H_a: np.ndarray,
    H_b: np.ndarray,
    H_desired_cam_from_source: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    usable_proj: np.ndarray,
    ceil_proj: np.ndarray,
    proj_w: int,
    proj_h: int,
    source_w: int,
    source_h: int,
) -> dict:
    """Precompute maps: source (sx,sy) -> projector for each plane, then composite.

    Source layout matches the desired camera rectangle content stretched to
    source_w x source_h (video frame after fit letterboxing is handled upstream
    by rendering into a source canvas of desired aspect).

    Here source == desired-content space at source_w x source_h, and
    H_desired maps that space into camera. We use:
      H_proj_from_source_k = inv(H_cam_from_proj_k) @ H_desired_cam_from_source
    with H_desired scaled for source resolution.
    """
    # Scale H_desired from unit/source resolution: assume H_desired defined for source_w x source_h
    H_fwd_a = forward_H_proj_from_source(H_a, H_desired_cam_from_source)
    H_fwd_b = forward_H_proj_from_source(H_b, H_desired_cam_from_source)

    # Build inverse maps for remap: for each proj pixel, sample source
    # OpenCV remap: dst(x,y) = src(map_x(x,y), map_y(x,y))
    # We want proj(p) = source(s) where p = H_fwd @ s => s = inv(H_fwd) @ p
    inv_a = np.linalg.inv(H_fwd_a)
    inv_b = np.linalg.inv(H_fwd_b)
    yy, xx = np.mgrid[0:proj_h, 0:proj_w].astype(np.float64)
    ones = np.ones_like(xx)
    # Plane A
    pa = inv_a[0, 0] * xx + inv_a[0, 1] * yy + inv_a[0, 2] * ones
    qa = inv_a[1, 0] * xx + inv_a[1, 1] * yy + inv_a[1, 2] * ones
    wa = inv_a[2, 0] * xx + inv_a[2, 1] * yy + inv_a[2, 2] * ones
    mapx_a = (pa / wa).astype(np.float32)
    mapy_a = (qa / wa).astype(np.float32)
    pb = inv_b[0, 0] * xx + inv_b[0, 1] * yy + inv_b[0, 2] * ones
    qb = inv_b[1, 0] * xx + inv_b[1, 1] * yy + inv_b[1, 2] * ones
    wb = inv_b[2, 0] * xx + inv_b[2, 1] * yy + inv_b[2, 2] * ones
    mapx_b = (pb / wb).astype(np.float32)
    mapy_b = (qb / wb).astype(np.float32)

    ma = (mask_a > 0) & (usable_proj > 0) & (ceil_proj == 0)
    mb = (mask_b > 0) & (usable_proj > 0) & (ceil_proj == 0)
    # Resolve overlap: prefer A
    mb = mb & ~ma

    return {
        "mapx_a": mapx_a,
        "mapy_a": mapy_a,
        "mapx_b": mapx_b,
        "mapy_b": mapy_b,
        "mask_a": ma.astype(np.uint8) * 255,
        "mask_b": mb.astype(np.uint8) * 255,
        "H_proj_from_source_plane_A": H_fwd_a,
        "H_proj_from_source_plane_B": H_fwd_b,
        "proj_w": proj_w,
        "proj_h": proj_h,
        "source_w": source_w,
        "source_h": source_h,
    }


def save_remap_bundle(out_dir: Path, bundle: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "piecewise_remap.npz",
        mapx_a=bundle["mapx_a"],
        mapy_a=bundle["mapy_a"],
        mapx_b=bundle["mapx_b"],
        mapy_b=bundle["mapy_b"],
        mask_a=bundle["mask_a"],
        mask_b=bundle["mask_b"],
    )
    np.save(out_dir / "H_proj_from_source_plane_A.npy", bundle["H_proj_from_source_plane_A"])
    np.save(out_dir / "H_proj_from_source_plane_B.npy", bundle["H_proj_from_source_plane_B"])
    meta = {
        "proj_w": bundle["proj_w"],
        "proj_h": bundle["proj_h"],
        "source_w": bundle["source_w"],
        "source_h": bundle["source_h"],
        "supported_codecs_note": SUPPORTED_CODECS,
    }
    (out_dir / "remap_meta.json").write_text(json.dumps(meta, indent=2))


def load_remap_bundle(calib_dir: Path) -> dict:
    calib_dir = Path(calib_dir)
    # Prefer video_playback/ subdir
    for base in (calib_dir / "video_playback", calib_dir / "prewarps", calib_dir):
        npz = base / "piecewise_remap.npz"
        if npz.exists():
            z = np.load(npz)
            return {
                "mapx_a": z["mapx_a"],
                "mapy_a": z["mapy_a"],
                "mapx_b": z["mapx_b"],
                "mapy_b": z["mapy_b"],
                "mask_a": z["mask_a"],
                "mask_b": z["mask_b"],
                "H_proj_from_source_plane_A": np.load(base / "H_proj_from_source_plane_A.npy")
                if (base / "H_proj_from_source_plane_A.npy").exists()
                else None,
                "H_proj_from_source_plane_B": np.load(base / "H_proj_from_source_plane_B.npy")
                if (base / "H_proj_from_source_plane_B.npy").exists()
                else None,
                "meta": json.loads((base / "remap_meta.json").read_text())
                if (base / "remap_meta.json").exists()
                else {},
            }
    raise FileNotFoundError(f"No piecewise_remap.npz under {calib_dir}")


def warp_frame_piecewise(frame_bgr: np.ndarray, bundle: dict) -> np.ndarray:
    """Remap a source BGR frame to projector framebuffer."""
    sw = bundle.get("meta", {}).get("source_w") or bundle.get("source_w")
    sh = bundle.get("meta", {}).get("source_h") or bundle.get("source_h")
    if sw and sh and (frame_bgr.shape[1] != sw or frame_bgr.shape[0] != sh):
        frame_bgr = cv2.resize(frame_bgr, (int(sw), int(sh)), interpolation=cv2.INTER_LINEAR)
    wa = cv2.remap(
        frame_bgr,
        bundle["mapx_a"],
        bundle["mapy_a"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    wb = cv2.remap(
        frame_bgr,
        bundle["mapx_b"],
        bundle["mapy_b"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    ma = (bundle["mask_a"] > 0)[:, :, None]
    mb = (bundle["mask_b"] > 0)[:, :, None]
    out = np.where(ma, wa, 0) + np.where(mb, wb, 0)
    return out.astype(np.uint8)


def generate_diagnostic_frame(
    w: int,
    h: int,
    t: float,
    show_seam: bool = False,
    seam_x: Optional[float] = None,
    debug_ceiling: bool = False,
    ceiling_y: Optional[float] = None,
) -> np.ndarray:
    """One diagnostic animation frame in source/desired space."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)
    # Outer white rectangle
    m = int(0.04 * min(w, h))
    cv2.rectangle(img, (m, m), (w - m - 1, h - m - 1), (255, 255, 255), 3)
    # Inset safe area
    m2 = int(0.08 * min(w, h))
    cv2.rectangle(img, (m2, m2), (w - m2 - 1, h - m2 - 1), (180, 180, 180), 1)
    # Grid
    step = max(40, min(w, h) // 12)
    for x in range(m, w - m, step):
        cv2.line(img, (x, m), (x, h - m - 1), (60, 60, 60), 1)
    for y in range(m, h - m, step):
        cv2.line(img, (m, y), (w - m - 1, y), (60, 60, 60), 1)
    # Circles
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 6, (0, 255, 255), 2)
    cv2.circle(img, (w // 4, h // 2), min(w, h) // 12, (0, 200, 0), 2)
    cv2.circle(img, (3 * w // 4, h // 2), min(w, h) // 12, (0, 200, 0), 2)
    # Travelling lines
    hx = int(m + (t % 1.0) * (w - 2 * m))
    hy = int(m + ((t * 0.7) % 1.0) * (h - 2 * m))
    cv2.line(img, (m, hy), (w - m - 1, hy), (0, 255, 0), 2)
    cv2.line(img, (hx, m), (hx, h - m - 1), (255, 128, 0), 2)
    # Diagonal
    d = int((t % 1.0) * (w + h))
    cv2.line(img, (m, m + d), (m + d, m), (200, 200, 0), 1)
    # Moving square crossing centre (seam)
    side = min(w, h) // 10
    sx = int(m + (t % 1.0) * (w - 2 * m - side))
    sy = h // 2 - side // 2
    cv2.rectangle(img, (sx, sy), (sx + side, sy + side), (0, 0, 255), -1)
    # Labels
    cv2.putText(img, "WALL_A", (w // 6, h // 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    cv2.putText(img, "WALL_B", (2 * w // 3, h // 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    for i, (x, y, lab) in enumerate(
        [(m, m, "TL"), (w - m, m, "TR"), (w - m, h - m, "BR"), (m, h - m, "BL")]
    ):
        cv2.putText(img, lab, (x - 20, y + (15 if i < 2 else -10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(img, f"t={t:.2f}", (m, h - m // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if show_seam and seam_x is not None:
        cv2.line(img, (int(seam_x), 0), (int(seam_x), h - 1), (255, 0, 255), 1)
    if debug_ceiling and ceiling_y is not None:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w - 1, int(ceiling_y)), (0, 0, 80), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    return img


def write_diagnostic_video(
    path: Path,
    w: int = 1280,
    h: int = 720,
    n_frames: int = 90,
    fps: float = 30.0,
    show_seam: bool = False,
    seam_x: Optional[float] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        fr = generate_diagnostic_frame(w, h, t, show_seam=show_seam, seam_x=seam_x)
        vw.write(fr)
    vw.release()
    return path


def measure_ceiling_leakage(
    projected_frame: np.ndarray,
    ceil_mask_proj: np.ndarray,
    content_threshold: int = 20,
) -> dict:
    """Fraction of active video pixels that fall in ceiling exclusion."""
    content = np.any(projected_frame > content_threshold, axis=2)
    ceil = ceil_mask_proj > 0
    leak = content & ceil
    n_content = int(content.sum())
    n_leak = int(leak.sum())
    frac = float(n_leak / max(n_content, 1))
    return {
        "active_video_pixels": n_content,
        "ceiling_leak_pixels": n_leak,
        "ceiling_leak_fraction": frac,
        "pass": frac <= 0.001,
    }


def build_desired_H_for_video_rect(
    camera_polygon: list,
    source_w: int,
    source_h: int,
) -> np.ndarray:
    """H_desired_cam_from_source mapping source rectangle → camera video polygon."""
    src = np.array(
        [[0, 0], [source_w - 1, 0], [source_w - 1, source_h - 1], [0, source_h - 1]],
        dtype=np.float32,
    )
    dst = np.array(camera_polygon, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return (H / H[2, 2]).astype(np.float64)


def prepare_playback_calibration(
    run_dir: Path,
    H_a: np.ndarray,
    H_b: np.ndarray,
    seam: dict,
    masks: dict,
    video_region: dict,
    source_w: int = 1280,
    source_h: int = 720,
    proj_w: int = 1920,
    proj_h: int = 1080,
    video_fit: str = "contain",
    aspect: float = 16 / 9,
) -> dict:
    """Build and save static remap + metadata for play-video."""
    run_dir = Path(run_dir)
    out = run_dir / "video_playback"
    out.mkdir(parents=True, exist_ok=True)
    fit = apply_video_fit(source_w, source_h, video_region, video_fit=video_fit)
    # Desired H maps full source frame to fit dest quad (contain may letterbox)
    H_des = build_desired_H_for_video_rect(fit["dest_quad"], source_w, source_h)
    np.save(out / "H_desired_cam_from_source.npy", H_des)
    (out / "video_fit_mapping.json").write_text(json.dumps(fit, indent=2))
    (out / "maximum_video_region.json").write_text(json.dumps(video_region, indent=2))

    usable = masks["usable_mask_projector"]
    ceil = masks["excluded_ceiling_mask_projector"]
    bundle = build_piecewise_remap(
        H_a,
        H_b,
        H_des,
        masks["mask_plane_A"],
        masks["mask_plane_B"],
        usable,
        ceil,
        proj_w,
        proj_h,
        source_w,
        source_h,
    )
    save_remap_bundle(out, bundle)
    # Leakage self-check with diagnostic mid frame
    diag = generate_diagnostic_frame(source_w, source_h, 0.5)
    proj = warp_frame_piecewise(diag, {**bundle, "meta": {"source_w": source_w, "source_h": source_h}})
    cv2.imwrite(str(out / "diagnostic_sample_prewarp.png"), proj)
    leak = measure_ceiling_leakage(proj, ceil)
    (out / "ceiling_leakage_selfcheck.json").write_text(json.dumps(leak, indent=2))
    meta = {
        "calibration_run": str(run_dir),
        "video_fit": video_fit,
        "aspect": aspect,
        "source_resolution": [source_w, source_h],
        "projector_resolution": [proj_w, proj_h],
        "straightness": "camera_viewer_coordinates",
        "supported_codecs": SUPPORTED_CODECS,
        "ceiling_leakage_selfcheck": leak,
    }
    (out / "playback_calibration.json").write_text(json.dumps(meta, indent=2))
    return meta


def play_video_on_projector(
    calib_run: Path,
    input_path: Optional[Path] = None,
    diagnostic: bool = False,
    loop: bool = True,
    screen_id: Optional[int] = None,
    settle_s: float = 0.0,
    max_frames: Optional[int] = None,
) -> dict:
    """Project remapped video. Does not modify calibration artifacts."""
    from .display_control import ProjectorController, list_displays, select_projector_display

    calib_run = Path(calib_run)
    bundle = load_remap_bundle(calib_run)
    meta = bundle.get("meta") or {}
    proj_w = int(meta.get("proj_w", bundle["mapx_a"].shape[1]))
    proj_h = int(meta.get("proj_h", bundle["mapx_a"].shape[0]))
    source_w = int(meta.get("source_w", 1280))
    source_h = int(meta.get("source_h", 720))

    displays = list_displays()
    selected = select_projector_display(
        displays, prefer_resolution=(proj_w, proj_h), prefer_screen_id=screen_id
    )
    if selected is None:
        return {"ok": False, "error": "no_projector_display"}
    projector = ProjectorController(selected, displays)
    projector.start()
    dropped = 0
    shown = 0
    t0 = time.time()
    try:
        if diagnostic or input_path is None:
            n = max_frames or 180
            for i in range(n):
                t = (i % 90) / 89.0
                fr = generate_diagnostic_frame(source_w, source_h, t)
                out = warp_frame_piecewise(fr, {**bundle, "meta": meta})
                projector.show_image(out)
                shown += 1
                time.sleep(1 / 30.0)
                if not loop and i + 1 >= n:
                    break
            if loop and max_frames is None:
                # single diagnostic cycle in autonomous/tests
                pass
        else:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return {"ok": False, "error": f"cannot_open_video:{input_path}"}
            frames_limit = max_frames
            while True:
                ok, fr = cap.read()
                if not ok:
                    if loop and frames_limit is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                out = warp_frame_piecewise(fr, {**bundle, "meta": meta})
                projector.show_image(out)
                shown += 1
                if frames_limit is not None and shown >= frames_limit:
                    break
            cap.release()
    finally:
        projector.shutdown()
    return {
        "ok": True,
        "frames_shown": shown,
        "dropped_frames": dropped,
        "elapsed_s": time.time() - t0,
        "projector": {"width": selected.width, "height": selected.height},
    }
