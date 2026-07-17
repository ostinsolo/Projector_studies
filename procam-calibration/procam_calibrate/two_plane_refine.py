"""Coupled two-plane residual refinement (homography-only, no CSPR)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import apply_H, estimate_H_cam_from_proj
from .homography_baseline import forward_H_proj_from_source, prewarp_from_forward_H
from .residual_refine import STRENGTHS, interpolate_homography_strength
from .two_plane import two_plane_acceptance


MAX_PHYSICAL_ITERS = 4


def candidate_forward_for_plane(
    H_proj_from_source_current: np.ndarray,
    pts_source: np.ndarray,
    pts_cam_detected: np.ndarray,
    H_desired_cam_from_source: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Full-strength candidate: H_proj' = H_proj @ inv(H_cam_obs) @ H_desired."""
    H_cam_obs, stats = estimate_H_cam_from_proj(pts_source, pts_cam_detected)
    H_cand = (
        H_proj_from_source_current
        @ np.linalg.inv(H_cam_obs)
        @ H_desired_cam_from_source
    )
    H_cand = (H_cand / H_cand[2, 2]).astype(np.float64)
    return H_cand, {"H_cam_from_source_observed_fit": stats}


def classify_plane_residual(pts_cam: np.ndarray, pts_desired: np.ndarray) -> str:
    res = pts_cam - pts_desired
    base = float(np.median(np.linalg.norm(res, axis=1)))
    if base < 1e-6:
        return "none"
    mean_v = res.mean(axis=0)
    mean_mag = float(np.linalg.norm(mean_v))
    std = float(np.linalg.norm(res - mean_v, axis=1).std())
    if mean_mag > 0.85 * base and std < 0.35 * base:
        return "translation"
    # Try projective fit residual
    H, _ = estimate_H_cam_from_proj(pts_desired, pts_cam)
    after_h = float(np.median(np.linalg.norm(apply_H(H, pts_desired) - pts_cam, axis=1)))
    if after_h < 0.4 * base:
        return "projective"
    if after_h > 0.7 * base:
        return "noisy"
    return "affine_or_lens_like"


def coupled_accept(
    prev: dict,
    new: dict,
    jitter_median: float,
) -> tuple[bool, str]:
    """Accept only when both planes, global, and seam jointly improve or already pass."""
    if not new.get("valid"):
        return False, "invalid_new"
    if not prev.get("valid"):
        return True, "prev_invalid"

    j = max(float(jitter_median), 0.05)

    def _ok_or_improve(tag: str, key: str) -> bool:
        pv = (prev.get(tag) or {}).get(key)
        nv = (new.get(tag) or {}).get(key)
        if nv is None:
            return False
        if (prev.get(tag) or {}).get("valid") and (pv is not None) and pv <= (
            5.0 if "median" in key else 12.0
        ):
            return nv <= pv + j  # already passing: no regress
        if pv is None:
            return True
        return nv < pv - j

    for plane in ("plane_A", "plane_B"):
        if not _ok_or_improve(plane, "target_median_err_px"):
            return False, f"{plane}_median_regress"
        if not _ok_or_improve(plane, "target_p95_err_px"):
            return False, f"{plane}_p95_regress"

    for key in ("target_median_err_px", "target_p95_err_px"):
        if new.get(key) is None or prev.get(key) is None:
            return False, f"missing_{key}"
        # already absolute-pass globally → allow non-regress; else require improve
        thresh = 5.0 if "median" in key else 12.0
        if prev[key] <= thresh:
            if new[key] > prev[key] + j:
                return False, f"global_{key}_regress"
        elif new[key] >= prev[key] - j:
            return False, f"global_{key}_no_improve"

    sp, sn = prev.get("seam") or {}, new.get("seam") or {}
    if sp and sn and sp.get("median_seam_mismatch_px") is not None:
        if sn.get("median_seam_mismatch_px", 1e9) > sp["median_seam_mismatch_px"] + j:
            return False, "seam_regress"
        if sn.get("p95_seam_mismatch_px", 1e9) > (sp.get("p95_seam_mismatch_px") or 0) + j:
            return False, "seam_p95_regress"

    if (new.get("coverage_fraction") or 0) < 0.80:
        return False, "coverage"

    return True, "accepted"


def apply_coupled_strength(
    H_fwd_a_cur: np.ndarray,
    H_fwd_b_cur: np.ndarray,
    H_fwd_a_full: np.ndarray,
    H_fwd_b_full: np.ndarray,
    control_pts: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    Ha = interpolate_homography_strength(H_fwd_a_cur, H_fwd_a_full, control_pts, strength)
    Hb = interpolate_homography_strength(H_fwd_b_cur, H_fwd_b_full, control_pts, strength)
    return Ha, Hb


def build_piecewise_from_forwards(
    content: np.ndarray,
    H_fwd_a: np.ndarray,
    H_fwd_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> np.ndarray:
    wa = prewarp_from_forward_H(content, H_fwd_a, proj_w, proj_h)
    wb = prewarp_from_forward_H(content, H_fwd_b, proj_w, proj_h)
    ma = (mask_a > 0)[:, :, None]
    mb = (mask_b > 0)[:, :, None]
    return (np.where(ma, wa, 0) + np.where(mb, wb, 0)).astype(np.uint8)


def preserve_best(
    out_dir: Path,
    H_fwd_a: np.ndarray,
    H_fwd_b: np.ndarray,
    prewarp: np.ndarray,
    metrics: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "H_proj_from_source_plane_A_best.npy", H_fwd_a)
    np.save(out_dir / "H_proj_from_source_plane_B_best.npy", H_fwd_b)
    cv2.imwrite(str(out_dir / "prewarp_piecewise_best.png"), prewarp)
    (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def propose_coupled_candidates(
    H_fwd_a: np.ndarray,
    H_fwd_b: np.ndarray,
    keys_a: list[str],
    keys_b: list[str],
    source_by_key: dict[str, list[float]],
    cam_by_key: dict[str, list[float]],
    H_desired: np.ndarray,
    control_pts: np.ndarray,
) -> list[dict[str, Any]]:
    """Build strength-interpolated coupled (A,B) forward candidates."""
    pts_a = np.array([source_by_key[k] for k in keys_a], dtype=np.float64)
    cam_a = np.array([cam_by_key[k] for k in keys_a], dtype=np.float64)
    pts_b = np.array([source_by_key[k] for k in keys_b], dtype=np.float64)
    cam_b = np.array([cam_by_key[k] for k in keys_b], dtype=np.float64)
    full_a, meta_a = candidate_forward_for_plane(H_fwd_a, pts_a, cam_a, H_desired)
    full_b, meta_b = candidate_forward_for_plane(H_fwd_b, pts_b, cam_b, H_desired)
    out = []
    for s in STRENGTHS:
        Ha, Hb = apply_coupled_strength(H_fwd_a, H_fwd_b, full_a, full_b, control_pts, s)
        out.append(
            {
                "strength": s,
                "H_fwd_a": Ha,
                "H_fwd_b": Hb,
                "meta_a": meta_a,
                "meta_b": meta_b,
                "class_a": classify_plane_residual(cam_a, apply_H(H_desired, pts_a)),
                "class_b": classify_plane_residual(cam_b, apply_H(H_desired, pts_b)),
            }
        )
    return out


def evaluate_pair_acceptance(unc: dict, cor: dict, seam_geom: Optional[dict] = None) -> dict:
    return two_plane_acceptance(unc, cor, seam_geom)
