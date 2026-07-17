"""Two connected planar surfaces: dual-homography fit, seam, piecewise pre-warp.

Isolated from the validated single-plane production path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .charuco import apply_H, estimate_H_cam_from_proj
from .homography_baseline import forward_H_proj_from_source, prewarp_from_forward_H


MIN_POINTS_PER_PLANE = 12
ONE_H_MEDIAN_ACCEPT = 2.0
ONE_H_P95_ACCEPT = 5.0
TWO_MODEL_ERROR_MARGIN = 0.55  # two-H robust error must be < margin * one-H
SEAM_MEDIAN_TARGET = 3.0
SEAM_P95_TARGET = 6.0
ASSIGN_REFIT_ITERS = 8


def _robust_error(err: np.ndarray) -> float:
    if len(err) == 0:
        return float("inf")
    return float(np.median(err) + 0.25 * np.percentile(err, 95))


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.astype(np.float32))
    return float(cv2.contourArea(hull))


def _spatial_contiguity_ok(pts: np.ndarray, labels: np.ndarray, plane: int) -> bool:
    """Reject checkerboard interleaving: plane points must form one connected proj cluster."""
    sel = pts[labels == plane]
    if len(sel) < MIN_POINTS_PER_PLANE:
        return False
    # Pairwise NN distances without scipy
    dmat = np.linalg.norm(sel[:, None, :] - sel[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)
    radius = max(3.0 * float(np.median(nn)), 1e-3)
    n = len(sel)
    visited = np.zeros(n, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        i = stack.pop()
        nbrs = np.where(dmat[i] <= radius)[0]
        for j in nbrs:
            if not visited[j]:
                visited[j] = True
                stack.append(int(j))
    return bool(visited.mean() >= 0.9)


def fit_single_homography(
    pts_proj: np.ndarray, pts_cam: np.ndarray, ransac_threshold: float = 3.0
) -> tuple[np.ndarray, np.ndarray, dict]:
    H, stats = estimate_H_cam_from_proj(pts_proj, pts_cam, ransac_threshold=ransac_threshold)
    err = np.linalg.norm(apply_H(H, pts_proj) - pts_cam, axis=1)
    return H, err, stats


def _sequential_two_ransac(
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
    ransac_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial two-model seeds via sequential RANSAC (not by ID order)."""
    H_a, err_a, _ = fit_single_homography(pts_proj, pts_cam, ransac_threshold)
    # Inliers of first model
    inl_a = err_a <= max(ransac_threshold * 1.5, np.percentile(err_a, 40))
    if inl_a.sum() < MIN_POINTS_PER_PLANE or (~inl_a).sum() < MIN_POINTS_PER_PLANE:
        # Spatial seed: split by projector x median (geometry, not ID)
        mid = float(np.median(pts_proj[:, 0]))
        inl_a = pts_proj[:, 0] <= mid
        if inl_a.sum() < MIN_POINTS_PER_PLANE:
            inl_a = pts_proj[:, 1] <= float(np.median(pts_proj[:, 1]))
    H_a, _, _ = fit_single_homography(pts_proj[inl_a], pts_cam[inl_a], ransac_threshold)
    rest = ~inl_a
    if rest.sum() < 4:
        rest = ~inl_a
        # force complement
        rest = np.ones(len(pts_proj), dtype=bool)
        rest[np.where(inl_a)[0][: max(4, MIN_POINTS_PER_PLANE)]] = False
    H_b, _, _ = fit_single_homography(pts_proj[rest], pts_cam[rest], ransac_threshold)
    labels = np.zeros(len(pts_proj), dtype=np.int32)
    labels[rest] = 1
    return H_a, H_b, labels


def fit_two_homographies(
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
    ids: Optional[list[int]] = None,
    ransac_threshold: float = 3.0,
    prefer_two_plane: bool = False,
    seam_x_hint: Optional[float] = None,
) -> dict:
    """Deterministic one-H vs two-H model selection and plane assignment.

    prefer_two_plane: for intentional two-wall physical setups, do not short-circuit
    to one_plane when a single H happens to fit well; still attempt a spatial split.
    seam_x_hint: optional projector-x fold prior (from architecture / brightness).
    """
    pts_proj = np.asarray(pts_proj, dtype=np.float64)
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    n = len(pts_proj)
    if ids is None:
        ids = list(range(n))
    assert len(ids) == n

    H1, err1, stats1 = fit_single_homography(pts_proj, pts_cam, ransac_threshold)
    one_med = float(np.median(err1))
    one_p95 = float(np.percentile(err1, 95))
    one_ok = one_med <= ONE_H_MEDIAN_ACCEPT and one_p95 <= ONE_H_P95_ACCEPT
    comparison = {
        "one_H_median_reproj_px": one_med,
        "one_H_p95_reproj_px": one_p95,
        "one_H_robust_error": _robust_error(err1),
        "one_H_accepts_flat_wall": one_ok,
        "one_H_stats": stats1,
        "prefer_two_plane": prefer_two_plane,
        "seam_x_hint": seam_x_hint,
    }
    if one_ok and not prefer_two_plane:
        return {
            "model": "one_plane",
            "two_plane_hypothesis": False,
            "H_cam_from_proj_single": H1,
            "assignment": {str(i): "single" for i in ids},
            "labels": [-1] * n,
            "comparison": comparison,
            "valid": True,
            "reason": "one_homography_explains_data",
        }

    spatial_locked = False
    seed_x = seam_x_hint
    if seed_x is None and prefer_two_plane:
        # Default fold prior: median projector x among observations
        seed_x = float(np.median(pts_proj[:, 0]))
    if seed_x is not None and prefer_two_plane:
        labels = (pts_proj[:, 0] >= float(seed_x)).astype(np.int32)
        if (labels == 0).sum() >= MIN_POINTS_PER_PLANE and (labels == 1).sum() >= MIN_POINTS_PER_PLANE:
            H_a, _, _ = fit_single_homography(
                pts_proj[labels == 0], pts_cam[labels == 0], ransac_threshold
            )
            H_b, _, _ = fit_single_homography(
                pts_proj[labels == 1], pts_cam[labels == 1], ransac_threshold
            )
            # Residual reassignment collapses when both Hs fit almost equally.
            spatial_locked = True
        else:
            H_a, H_b, labels = _sequential_two_ransac(pts_proj, pts_cam, ransac_threshold)
    elif seam_x_hint is not None:
        labels = (pts_proj[:, 0] >= float(seam_x_hint)).astype(np.int32)
        if (labels == 0).sum() >= MIN_POINTS_PER_PLANE and (labels == 1).sum() >= MIN_POINTS_PER_PLANE:
            H_a, _, _ = fit_single_homography(
                pts_proj[labels == 0], pts_cam[labels == 0], ransac_threshold
            )
            H_b, _, _ = fit_single_homography(
                pts_proj[labels == 1], pts_cam[labels == 1], ransac_threshold
            )
        else:
            H_a, H_b, labels = _sequential_two_ransac(pts_proj, pts_cam, ransac_threshold)
    else:
        H_a, H_b, labels = _sequential_two_ransac(pts_proj, pts_cam, ransac_threshold)
    if not spatial_locked:
        for _ in range(ASSIGN_REFIT_ITERS):
            err_a = np.linalg.norm(apply_H(H_a, pts_proj) - pts_cam, axis=1)
            err_b = np.linalg.norm(apply_H(H_b, pts_proj) - pts_cam, axis=1)
            new_labels = (err_b < err_a).astype(np.int32)
            # Ensure minimum population
            if (new_labels == 0).sum() < MIN_POINTS_PER_PLANE or (
                new_labels == 1
            ).sum() < MIN_POINTS_PER_PLANE:
                break
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            H_a, _, _ = fit_single_homography(
                pts_proj[labels == 0], pts_cam[labels == 0], ransac_threshold
            )
            H_b, _, _ = fit_single_homography(
                pts_proj[labels == 1], pts_cam[labels == 1], ransac_threshold
            )
    comparison["spatial_labels_locked"] = spatial_locked

    # Canonicalize: plane A = left/top projector centroid
    from .two_plane_observations import canonicalize_plane_labels

    labels_before = labels.copy()
    labels = canonicalize_plane_labels(pts_proj, labels)
    if not np.array_equal(labels, labels_before):
        H_a, H_b = H_b, H_a

    err_a = np.linalg.norm(apply_H(H_a, pts_proj) - pts_cam, axis=1)
    err_b = np.linalg.norm(apply_H(H_b, pts_proj) - pts_cam, axis=1)
    err_two = np.where(labels == 0, err_a, err_b)
    two_robust = _robust_error(err_two)
    one_robust = comparison["one_H_robust_error"]
    margin_ok = two_robust < TWO_MODEL_ERROR_MARGIN * one_robust

    n_a = int((labels == 0).sum())
    n_b = int((labels == 1).sum())
    area_a = _hull_area(pts_proj[labels == 0])
    area_b = _hull_area(pts_proj[labels == 1])
    contig_a = _spatial_contiguity_ok(pts_proj, labels, 0)
    contig_b = _spatial_contiguity_ok(pts_proj, labels, 1)
    # Interleaving check along the A↔B centroid axis (not raw x — allows horizontal seams)
    ca = pts_proj[labels == 0].mean(0)
    cb = pts_proj[labels == 1].mean(0)
    axis = cb - ca
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    order = np.argsort(pts_proj @ axis)
    flips = int(np.sum(np.diff(labels[order]) != 0))
    interleaved = flips > max(4, n // 6)

    spatial_ok = (
        n_a >= MIN_POINTS_PER_PLANE
        and n_b >= MIN_POINTS_PER_PLANE
        and area_a > 1.0
        and area_b > 1.0
        and (contig_a if not spatial_locked else True)
        and (contig_b if not spatial_locked else True)
        and (not interleaved if not spatial_locked else True)
    )
    # Forced two-wall: accept spatial model if two-H is not worse than one-H
    forced_ok = bool(
        prefer_two_plane and spatial_ok and two_robust <= one_robust * 1.05
    )
    valid = (spatial_ok and margin_ok) or forced_ok
    comparison["forced_two_plane_ok"] = forced_ok
    comparison["spatial_ok"] = spatial_ok
    assignment = {
        str(ids[k]): ("A" if labels[k] == 0 else "B") for k in range(n)
    }
    comparison.update(
        {
            "two_H_median_reproj_px": float(np.median(err_two)),
            "two_H_p95_reproj_px": float(np.percentile(err_two, 95)),
            "two_H_robust_error": two_robust,
            "error_margin_factor": TWO_MODEL_ERROR_MARGIN,
            "margin_pass": margin_ok,
            "n_plane_A": n_a,
            "n_plane_B": n_b,
            "hull_area_A": area_a,
            "hull_area_B": area_b,
            "contiguity_A": contig_a,
            "contiguity_B": contig_b,
            "label_flips_along_x": flips,
            "interleaved_rejected": interleaved,
            "plane_A_median_reproj_px": float(np.median(err_a[labels == 0])),
            "plane_B_median_reproj_px": float(np.median(err_b[labels == 1])),
        }
    )
    if valid:
        return {
            "model": "two_plane",
            "two_plane_hypothesis": True,
            "valid": True,
            "H_cam_from_proj_plane_A": H_a,
            "H_cam_from_proj_plane_B": H_b,
            "H_cam_from_proj_single": H1,
            "labels": labels.tolist(),
            "assignment": assignment,
            "per_point_err_A": err_a.tolist(),
            "per_point_err_B": err_b.tolist(),
            "comparison": comparison,
            "reason": "ok_prefer_two_plane" if forced_ok and not margin_ok else "ok",
        }
    if one_ok:
        return {
            "model": "one_plane",
            "two_plane_hypothesis": prefer_two_plane,
            "H_cam_from_proj_single": H1,
            "assignment": {str(i): "single" for i in ids},
            "labels": [-1] * n,
            "comparison": comparison,
            "valid": True,
            "reason": "one_homography_explains_data",
        }
    return {
        "model": "two_plane_rejected",
        "two_plane_hypothesis": True,
        "valid": False,
        "H_cam_from_proj_plane_A": H_a,
        "H_cam_from_proj_plane_B": H_b,
        "H_cam_from_proj_single": H1,
        "labels": labels.tolist(),
        "assignment": assignment,
        "per_point_err_A": err_a.tolist(),
        "per_point_err_B": err_b.tolist(),
        "comparison": comparison,
        "reason": "failed_two_plane_gates",
    }


def estimate_straight_seam(
    pts_proj: np.ndarray,
    labels: np.ndarray,
    proj_w: int = 1920,
    proj_h: int = 1080,
    H_a: Optional[np.ndarray] = None,
    H_b: Optional[np.ndarray] = None,
) -> dict:
    """Straight seam in projector coordinates separating plane clusters.

    Uses the A↔B centroid axis as the seam normal, then optimizes the offset so
    the two plane models agree along the line (when H_a/H_b are provided).
    """
    a = pts_proj[labels == 0]
    b = pts_proj[labels == 1]
    ca, cb = a.mean(0), b.mean(0)
    mid0 = 0.5 * (ca + cb)
    # Prefer axis-aligned seam (MVP: vertical or horizontal fold)
    delta = cb - ca
    if abs(delta[0]) >= abs(delta[1]):
        normal = np.array([1.0 if delta[0] >= 0 else -1.0, 0.0])
    else:
        normal = np.array([0.0, 1.0 if delta[1] >= 0 else -1.0])
    mid = mid0.copy()

    if H_a is not None and H_b is not None:
        tang = np.array([-normal[1], normal[0]])
        span = max(abs(delta[0]), abs(delta[1]), 80.0)

        def _score_offset(t: float) -> float:
            m = mid0 + t * normal
            ts = np.linspace(-0.5 * max(proj_w, proj_h), 0.5 * max(proj_w, proj_h), 80)
            samples = m[None, :] + ts[:, None] * tang[None, :]
            inside = (
                (samples[:, 0] >= 0)
                & (samples[:, 0] < proj_w)
                & (samples[:, 1] >= 0)
                & (samples[:, 1] < proj_h)
            )
            if inside.sum() < 10:
                return float("inf")
            samples = samples[inside]
            d = np.linalg.norm(apply_H(H_a, samples) - apply_H(H_b, samples), axis=1)
            d = d[np.isfinite(d)]
            if len(d) == 0:
                return float("inf")
            return float(np.median(d) + 0.25 * np.percentile(d, 95))

        best_t, best_score = 0.0, float("inf")
        for t in np.linspace(-span, span, 161):
            score = _score_offset(t)
            if score < best_score:
                best_score = score
                best_t = float(t)
        # Local refine
        for t in np.linspace(best_t - span / 40, best_t + span / 40, 41):
            score = _score_offset(t)
            if score < best_score:
                best_score = score
                best_t = float(t)
        mid = mid0 + best_t * normal

    a_c, b_c = float(normal[0]), float(normal[1])
    c = float(-normal[0] * mid[0] - normal[1] * mid[1])
    ends = []
    tang = np.array([-normal[1], normal[0]])
    for t in np.linspace(-2 * max(proj_w, proj_h), 2 * max(proj_w, proj_h), 4000):
        p = mid + t * tang
        if 0 <= p[0] < proj_w and 0 <= p[1] < proj_h:
            ends.append(p)
    if len(ends) < 2:
        p0 = np.array([mid[0], 0.0])
        p1 = np.array([mid[0], proj_h - 1.0])
    else:
        p0, p1 = ends[0], ends[-1]

    side_a = np.median((a - mid) @ normal)
    if side_a > 0:
        a_c, b_c, c = -a_c, -b_c, -c
        normal = -normal

    return {
        "type": "straight",
        "line_abc": [a_c, b_c, c],
        "normal": normal.tolist(),
        "midpoint": mid.tolist(),
        "endpoint_0": p0.tolist(),
        "endpoint_1": p1.tolist(),
        "proj_resolution": [proj_w, proj_h],
        "side_convention": "n·(x-mid) < 0 → plane A; >= 0 → plane B",
    }


def build_plane_masks(
    seam: dict,
    proj_w: int,
    proj_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exclusive masks: A=255 where plane A, B=255 where plane B, viz RGB."""
    yy, xx = np.mgrid[0:proj_h, 0:proj_w]
    a, b, c = seam["line_abc"]
    side = a * xx + b * yy + c
    mask_a = (side < 0).astype(np.uint8) * 255
    mask_b = (side >= 0).astype(np.uint8) * 255
    # No overlap / no holes
    assert np.all((mask_a > 0) ^ (mask_b > 0))
    assert np.all((mask_a > 0) | (mask_b > 0))
    viz = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
    viz[mask_a > 0] = (0, 180, 255)  # A
    viz[mask_b > 0] = (255, 120, 0)  # B
    # draw seam
    p0 = tuple(np.round(seam["endpoint_0"]).astype(int))
    p1 = tuple(np.round(seam["endpoint_1"]).astype(int))
    cv2.line(viz, p0, p1, (255, 255, 255), 2)
    return mask_a, mask_b, viz


def seam_mismatch_in_camera(
    pts_proj: np.ndarray,
    labels: np.ndarray,
    H_a: np.ndarray,
    H_b: np.ndarray,
    seam: dict,
    n_samples: int = 40,
) -> dict:
    """Sample seam in ProjFB; compare H_a vs H_b camera projections."""
    p0 = np.array(seam["endpoint_0"], dtype=np.float64)
    p1 = np.array(seam["endpoint_1"], dtype=np.float64)
    ts = np.linspace(0.05, 0.95, n_samples)
    samples = (1 - ts)[:, None] * p0 + ts[:, None] * p1
    ca = apply_H(H_a, samples)
    cb = apply_H(H_b, samples)
    err = np.linalg.norm(ca - cb, axis=1)
    return {
        "n_samples": n_samples,
        "median_seam_mismatch_px": float(np.median(err)),
        "p95_seam_mismatch_px": float(np.percentile(err, 95)),
        "max_seam_mismatch_px": float(err.max()),
        "pass": bool(
            float(np.median(err)) <= SEAM_MEDIAN_TARGET
            and float(np.percentile(err, 95)) <= SEAM_P95_TARGET
        ),
    }


def define_desired_per_plane(
    H_plane: np.ndarray,
    pts_proj_plane: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> np.ndarray:
    """Axis-aligned desired mapping for a plane's projector extent → camera AA rect."""
    if len(pts_proj_plane) < 4:
        corners_p = np.array(
            [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]],
            dtype=np.float32,
        )
    else:
        x0, y0 = pts_proj_plane.min(0)
        x1, y1 = pts_proj_plane.max(0)
        corners_p = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
        )
    cam = apply_H(H_plane, corners_p.astype(np.float64)).astype(np.float32)
    x, y, w, h = cv2.boundingRect(cam.reshape(-1, 1, 2))
    w, h = max(int(w), 2), max(int(h), 2)
    desired = np.array(
        [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners_p, desired).astype(np.float64)


def compose_piecewise_prewarp(
    content: np.ndarray,
    H_a: np.ndarray,
    H_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    H_des_a: np.ndarray,
    H_des_b: np.ndarray,
    proj_w: int,
    proj_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Warp source through each plane forward H and composite with exclusive masks."""
    H_fwd_a = forward_H_proj_from_source(H_a, H_des_a)
    H_fwd_b = forward_H_proj_from_source(H_b, H_des_b)
    wa = prewarp_from_forward_H(content, H_fwd_a, proj_w, proj_h)
    wb = prewarp_from_forward_H(content, H_fwd_b, proj_w, proj_h)
    ma = (mask_a > 0)[:, :, None]
    mb = (mask_b > 0)[:, :, None]
    out = np.where(ma, wa, 0) + np.where(mb, wb, 0)
    # exclusivity
    overlap = int(np.sum((mask_a > 0) & (mask_b > 0)))
    holes = int(np.sum((mask_a == 0) & (mask_b == 0)))
    meta = {
        "overlap_pixels": overlap,
        "hole_pixels": holes,
        "exclusive": overlap == 0 and holes == 0,
        "H_proj_from_source_plane_A": H_fwd_a.tolist(),
        "H_proj_from_source_plane_B": H_fwd_b.tolist(),
    }
    return out.astype(np.uint8), H_fwd_a, H_fwd_b, meta


def save_two_plane_artifacts(out_dir: Path, result: dict, seam: dict, masks: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if result.get("H_cam_from_proj_plane_A") is not None:
        np.save(out_dir / "H_cam_from_proj_plane_A.npy", result["H_cam_from_proj_plane_A"])
        np.save(out_dir / "H_cam_from_proj_plane_B.npy", result["H_cam_from_proj_plane_B"])
    (out_dir / "plane_assignment_by_id.json").write_text(
        json.dumps(result.get("assignment", {}), indent=2)
    )
    (out_dir / "single_vs_two_model_comparison.json").write_text(
        json.dumps(result.get("comparison", {}), indent=2, default=str)
    )
    (out_dir / "seam_model.json").write_text(json.dumps(seam, indent=2))
    if "viz" in masks:
        cv2.imwrite(str(out_dir / "plane_masks_projfb.png"), masks["viz"])
    if "mask_a" in masks:
        cv2.imwrite(str(out_dir / "mask_plane_A.png"), masks["mask_a"])
        cv2.imwrite(str(out_dir / "mask_plane_B.png"), masks["mask_b"])
    if "H_fwd_a" in masks:
        np.save(out_dir / "H_proj_from_source_plane_A.npy", masks["H_fwd_a"])
        np.save(out_dir / "H_proj_from_source_plane_B.npy", masks["H_fwd_b"])
        (out_dir / "forward_prewarp_meta.json").write_text(
            json.dumps(masks.get("compose_meta", {}), indent=2)
        )
    if "prewarp" in masks:
        cv2.imwrite(str(out_dir / "prewarp_piecewise.png"), masks["prewarp"])
    (out_dir / "two_plane_result.json").write_text(
        json.dumps(
            {
                "model": result.get("model"),
                "valid": result.get("valid"),
                "reason": result.get("reason"),
                "two_plane_hypothesis": result.get("two_plane_hypothesis"),
                "seam_metrics": masks.get("seam_metrics"),
            },
            indent=2,
            default=str,
        )
    )


def run_two_plane_fit_from_points(
    pts_proj: np.ndarray,
    pts_cam: np.ndarray,
    ids: list[int],
    out_dir: Path,
    proj_w: int = 1920,
    proj_h: int = 1080,
    content: Optional[np.ndarray] = None,
) -> dict:
    """Full software path: fit → seam → masks → piecewise prewarp."""
    result = fit_two_homographies(pts_proj, pts_cam, ids=ids)
    seam: dict[str, Any] = {}
    masks: dict[str, Any] = {}
    if result.get("model") == "two_plane" and result.get("valid"):
        labels = np.array(result["labels"], dtype=np.int32)
        H_a = result["H_cam_from_proj_plane_A"]
        H_b = result["H_cam_from_proj_plane_B"]
        seam = estimate_straight_seam(pts_proj, labels, proj_w, proj_h, H_a=H_a, H_b=H_b)
        mask_a, mask_b, viz = build_plane_masks(seam, proj_w, proj_h)
        seam_metrics = seam_mismatch_in_camera(pts_proj, labels, H_a, H_b, seam)
        H_des_a = define_desired_per_plane(H_a, pts_proj[labels == 0], proj_w, proj_h)
        H_des_b = define_desired_per_plane(H_b, pts_proj[labels == 1], proj_w, proj_h)
        if content is None:
            content = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)
            content[:] = (40, 40, 40)
            # checker for visibility
            for y in range(0, proj_h, 80):
                for x in range(0, proj_w, 80):
                    if ((x // 80) + (y // 80)) % 2 == 0:
                        content[y : y + 40, x : x + 40] = (220, 220, 220)
        prewarp, H_fwd_a, H_fwd_b, meta = compose_piecewise_prewarp(
            content, H_a, H_b, mask_a, mask_b, H_des_a, H_des_b, proj_w, proj_h
        )
        masks = {
            "mask_a": mask_a,
            "mask_b": mask_b,
            "viz": viz,
            "prewarp": prewarp,
            "H_fwd_a": H_fwd_a,
            "H_fwd_b": H_fwd_b,
            "compose_meta": meta,
            "seam_metrics": seam_metrics,
        }
        result["seam_metrics"] = seam_metrics
    save_two_plane_artifacts(out_dir, result, seam, masks)
    return {**result, "seam": seam, "masks_meta": {k: masks[k] for k in masks if k not in ("mask_a", "mask_b", "viz", "prewarp", "H_fwd_a", "H_fwd_b")}}


SEAM_EXCLUSION_HALF_WIDTH_PX = 12.0


def seam_exclusion_mask(
    seam: dict, keys: list[str], pts_proj: np.ndarray, half_width: float = SEAM_EXCLUSION_HALF_WIDTH_PX
) -> dict:
    """Mark observations too close to the fold for plane-H fitting (not validation)."""
    a, b, c = seam["line_abc"]
    n = float(np.hypot(a, b)) + 1e-12
    dist = np.abs(a * pts_proj[:, 0] + b * pts_proj[:, 1] + c) / n
    excluded = dist <= half_width
    return {
        "half_width_px": half_width,
        "excluded_keys": [keys[i] for i in np.where(excluded)[0]],
        "n_excluded": int(excluded.sum()),
        "keep_mask": (~excluded).tolist(),
    }


def bootstrap_seam_stability(
    pts_proj: np.ndarray,
    labels: np.ndarray,
    H_a: np.ndarray,
    H_b: np.ndarray,
    proj_w: int,
    proj_h: int,
    n_boot: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    rng = rng or np.random.default_rng(0)
    mids = []
    for _ in range(n_boot):
        idx = rng.choice(len(pts_proj), size=len(pts_proj), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        if (labels[idx] == 0).sum() < 8 or (labels[idx] == 1).sum() < 8:
            continue
        s = estimate_straight_seam(pts_proj[idx], labels[idx], proj_w, proj_h, H_a, H_b)
        mids.append(s["midpoint"])
    if not mids:
        return {"n_boot": 0, "stable": False, "midpoint_std_px": None}
    arr = np.array(mids, dtype=np.float64)
    std = arr.std(axis=0)
    return {
        "n_boot": len(mids),
        "midpoint_mean": arr.mean(axis=0).tolist(),
        "midpoint_std_px": std.tolist(),
        "stable": bool(float(np.linalg.norm(std)) <= 25.0),
    }


def define_desired_target_two_plane(
    H_a: np.ndarray,
    H_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    seam: dict,
    proj_w: int,
    proj_h: int,
    cam_w: int,
    cam_h: int,
    source_pts_by_key: dict[str, list[float]],
    labels_by_key: dict[str, str],
) -> dict:
    """Predetermined desired camera target independent of corrected capture."""
    # Global desired: AA bounding box of the union of both physical planes mapped by each H
    corners = np.array(
        [[0, 0], [proj_w - 1, 0], [proj_w - 1, proj_h - 1], [0, proj_h - 1]], dtype=np.float64
    )
    cam_a = apply_H(H_a, corners)
    cam_b = apply_H(H_b, corners)
    all_c = np.vstack([cam_a, cam_b])
    x0, y0 = all_c.min(0)
    x1, y1 = all_c.max(0)
    # clamp into camera with margin
    pad = 20.0
    x0, y0 = max(pad, x0), max(pad, y0)
    x1, y1 = min(cam_w - 1 - pad, x1), min(cam_h - 1 - pad, y1)
    desired_poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    # Map source (ProjFB layout = source) corners → desired AA via a shared H_desired_cam_from_source
    H_des = cv2.getPerspectiveTransform(
        corners.astype(np.float32), np.array(desired_poly, dtype=np.float32)
    ).astype(np.float64)
    desired_by_key = {}
    for k, src in source_pts_by_key.items():
        p = apply_H(H_des, np.array([src], dtype=np.float64))[0]
        desired_by_key[k] = [float(p[0]), float(p[1])]
    # Seam samples in source/projector (same layout) → desired
    p0 = np.array(seam["endpoint_0"], dtype=np.float64)
    p1 = np.array(seam["endpoint_1"], dtype=np.float64)
    ts = np.linspace(0.05, 0.95, 21)
    seam_src = (1 - ts)[:, None] * p0 + ts[:, None] * p1
    seam_des = apply_H(H_des, seam_src)
    return {
        "source_resolution": [proj_w, proj_h],
        "camera_resolution": [cam_w, cam_h],
        "projector_resolution": [proj_w, proj_h],
        "desired_camera_polygon": desired_poly,
        "H_desired_cam_from_source": H_des.tolist(),
        "desired_point_by_key": desired_by_key,
        "desired_seam_samples": seam_des.tolist(),
        "labels_by_key": labels_by_key,
        "pixel_centre_convention": "OpenCV top-left origin; +u right +v down",
        "axis_convention": "+u right, +v down",
        "distortion_state": "none_applied",
        "note": "Desired geometry fixed before corrected capture analysis",
    }


def measure_two_plane_against_desired(
    obs_cam_by_key: dict[str, list[float]],
    desired: dict,
    labels_by_key: dict[str, str],
    expected_keys: Optional[list[str]] = None,
    seam_keys: Optional[list[str]] = None,
) -> dict:
    """Production target metrics with one evaluation key population."""
    detected = sorted(obs_cam_by_key.keys())
    desired_ids = set(desired["desired_point_by_key"].keys())
    expected = sorted(expected_keys) if expected_keys is not None else sorted(desired_ids)
    evaluation = sorted(set(detected) & desired_ids & set(expected))
    id_sets = {
        "detected_raw_keys": detected,
        "expected_visible_keys": expected,
        "evaluation_keys": evaluation,
        "expected_but_not_detected_keys": sorted(set(expected) - set(detected)),
        "detected_but_not_expected_keys": sorted(set(detected) - set(expected)),
    }
    if len(evaluation) < MIN_POINTS_PER_PLANE:
        return {
            "valid": False,
            "failure_reason": f"insufficient_evaluation_keys:{len(evaluation)}",
            "id_sets": id_sets,
            "target_median_err_px": None,
            "target_p95_err_px": None,
        }

    def _plane_stats(plane: str) -> dict:
        keys = [k for k in evaluation if labels_by_key.get(k) == plane]
        if len(keys) < 4:
            return {"valid": False, "n": len(keys), "target_median_err_px": None, "target_p95_err_px": None}
        err = np.array(
            [
                np.linalg.norm(
                    np.array(obs_cam_by_key[k]) - np.array(desired["desired_point_by_key"][k])
                )
                for k in keys
            ]
        )
        return {
            "valid": True,
            "n": len(keys),
            "evaluation_keys": keys,
            "target_median_err_px": float(np.median(err)),
            "target_p95_err_px": float(np.percentile(err, 95)),
            "target_max_err_px": float(err.max()),
            "target_mean_err_px": float(err.mean()),
        }

    err_all = np.array(
        [
            np.linalg.norm(np.array(obs_cam_by_key[k]) - np.array(desired["desired_point_by_key"][k]))
            for k in evaluation
        ]
    )
    plane_a = _plane_stats("A")
    plane_b = _plane_stats("B")
    # Seam mismatch from desired seam samples vs measured if provided
    seam_report = None
    if seam_keys:
        sk = [k for k in seam_keys if k in obs_cam_by_key and k in desired["desired_point_by_key"]]
        if sk:
            se = np.array(
                [
                    np.linalg.norm(
                        np.array(obs_cam_by_key[k]) - np.array(desired["desired_point_by_key"][k])
                    )
                    for k in sk
                ]
            )
            seam_report = {
                "n": len(sk),
                "median_seam_mismatch_px": float(np.median(se)),
                "p95_seam_mismatch_px": float(np.percentile(se, 95)),
                "max_seam_mismatch_px": float(se.max()),
            }

    return {
        "valid": True,
        "id_sets": id_sets,
        "evaluation_keys": evaluation,
        "n_evaluation_keys": len(evaluation),
        "target_median_err_px": float(np.median(err_all)),
        "target_p95_err_px": float(np.percentile(err_all, 95)),
        "target_max_err_px": float(err_all.max()),
        "plane_A": plane_a,
        "plane_B": plane_b,
        "seam": seam_report,
        "coverage_fraction": float(len(evaluation) / max(1, len(expected))),
    }


def two_plane_acceptance(unc: dict, cor: dict, seam_geom: Optional[dict] = None) -> dict:
    """Absolute acceptance for physical two-plane result."""
    reasons = []
    if not unc.get("valid") or not cor.get("valid"):
        return {"acceptance_pass": False, "absolute_gate_pass": False, "failure_reason": "invalid_metrics"}

    def _plane_ok(tag: str) -> bool:
        p = cor.get(tag) or {}
        if not p.get("valid"):
            reasons.append(f"{tag}_invalid")
            return False
        ok = (p.get("target_median_err_px") or 1e9) <= 5.0 and (p.get("target_p95_err_px") or 1e9) <= 12.0
        if not ok:
            reasons.append(f"{tag}_accuracy")
        return ok

    planes_ok = _plane_ok("plane_A") and _plane_ok("plane_B")
    global_ok = (cor.get("target_median_err_px") or 1e9) <= 5.0 and (cor.get("target_p95_err_px") or 1e9) <= 12.0
    if not global_ok:
        reasons.append("global_accuracy")
    improved = (cor.get("target_median_err_px") or 1e9) < (unc.get("target_median_err_px") or 0) and (
        cor.get("target_p95_err_px") or 1e9
    ) < (unc.get("target_p95_err_px") or 0)
    if not improved:
        reasons.append("no_improvement")
    cov_ok = (cor.get("coverage_fraction") or 0) >= 0.80
    if not cov_ok:
        reasons.append("coverage")
    seam_ok = True
    if seam_geom:
        seam_ok = (
            (seam_geom.get("median_seam_mismatch_px") or 1e9) <= SEAM_MEDIAN_TARGET
            and (seam_geom.get("p95_seam_mismatch_px") or 1e9) <= SEAM_P95_TARGET
        )
        if not seam_ok:
            reasons.append("seam")
    abs_pass = bool(planes_ok and global_ok and improved and cov_ok and seam_ok)
    return {
        "acceptance_pass": abs_pass,
        "absolute_gate_pass": abs_pass,
        "planes_ok": planes_ok,
        "global_ok": global_ok,
        "improved": improved,
        "coverage_ok": cov_ok,
        "seam_ok": seam_ok,
        "failure_reason": None if abs_pass else ";".join(reasons) or "failed",
    }
