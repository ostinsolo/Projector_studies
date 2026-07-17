"""Run kamino410 sample with OpenCV 5 compatibility (does not modify upstream)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "research_external" / "procam-calibration"
LOG = Path(__file__).resolve().parents[1] / "logs" / "kamino410_sample.log"
OUT = Path(__file__).resolve().parents[1] / "results" / "kamino410_sample.json"

sys.path.insert(0, str(UPSTREAM))

# OpenCV 5 API shim
if not hasattr(cv2, "structured_light_GrayCodePattern"):

    class _Shim:
        @staticmethod
        def create(w, h):
            return cv2.structured_light.GrayCodePattern_create(w, h)

    cv2.structured_light_GrayCodePattern = _Shim


def _corner_xy(corner) -> tuple[float, float]:
    a = np.asarray(corner, dtype=np.float64).reshape(-1)
    return float(a[0]), float(a[1])


# Monkey-patch calibrate.calibrate's corner indexing by wrapping findChessboardCorners usage
import calibrate as cal  # type: ignore

_orig = cal.calibrate


def calibrate_fixed(dirnames, gc_fname_lists, proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr, camP, camD):
    objps = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * np.mgrid[0 : chess_shape[0], 0 : chess_shape[1]].T.reshape(-1, 2)

    print("Calibrating ...")
    gc_height = int((proj_shape[0] - 1) / gc_step) + 1
    gc_width = int((proj_shape[1] - 1) / gc_step) + 1
    graycode = cv2.structured_light_GrayCodePattern.create(gc_width, gc_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    cam_shape = cv2.imread(gc_fname_lists[0][0], cv2.IMREAD_GRAYSCALE).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    print("  patch size :", patch_size_half * 2 + 1)

    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []
    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        print("  checking '" + dname + "'")
        if len(gc_filenames) != graycode.getNumberOfPatternImages() + 2:
            print("Error : invalid number of images in '" + dname + "'")
            return None

        imgs = []
        for fname in gc_filenames:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if cam_shape != img.shape:
                print("Error : image size of '" + fname + "' is mismatch")
                return None
            imgs.append(img)
        black_img = imgs.pop()
        white_img = imgs.pop()

        res, cam_corners = cv2.findChessboardCorners(white_img, chess_shape)
        if not res:
            print("Error : chessboard was not found in '" + gc_filenames[-2] + "'")
            return None
        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)

        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        for corner, objp in zip(cam_corners, objps):
            cx, cy = _corner_xy(corner)
            c_x = int(round(cx))
            c_y = int(round(cy))
            src_points = []
            dst_points = []
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                        continue
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(gc_step * np.array(proj_pix))
            if len(src_points) < patch_size_half**2:
                print(
                    "    Warning : corner",
                    c_x,
                    c_y,
                    "was skiped because decoded pixels were too few",
                )
                continue
            h_mat, _inl = cv2.findHomography(np.array(src_points), np.array(dst_points))
            point = h_mat @ np.array([cx, cy, 1.0]).transpose()
            point_pix = point[0:2] / point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
        if len(proj_corners) < 3:
            print("Error : too few corners decoded")
            return None
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))

    print("Initial solution of camera's parameters")
    ret, cam_int, cam_dist, _, _ = cv2.calibrateCamera(
        cam_objps_list, cam_corners_list, cam_shape[::-1], None, None, None, None
    )
    print("  RMS :", ret)
    print("Initial solution of projector's parameters")
    ret, proj_int, proj_dist, _, _ = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, proj_shape[::-1], None, None, None, None
    )
    print("  RMS :", ret)
    print("Stereocalibration")
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, _, _ = cv2.stereoCalibrate(
        proj_objps_list,
        cam_corners_list2,
        proj_corners_list,
        cam_int,
        cam_dist,
        proj_int,
        proj_dist,
        None,
    )
    print("  RMS :", ret)
    return {
        "stereo_rms": float(ret),
        "cam_int": np.asarray(cam_int).tolist(),
        "proj_int": np.asarray(proj_int).tolist(),
        "r": np.asarray(cam_proj_rmat).tolist(),
        "t": np.asarray(cam_proj_tvec).tolist(),
    }


cal.calibrate = calibrate_fixed


def main() -> int:
    import json
    import os

    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample = UPSTREAM / "sample_data"
    os.chdir(sample)
    sys.argv = ["calibrate.py", "768", "1024", "9", "7", "75", "1", "-black_thr", "40", "-white_thr", "5"]
    # Capture print by re-running patched path directly
    used_dirnames, gc_fname_lists = [], []
    # Use upstream main's file discovery via calibrate.main after patch
    # Instead call our function through main after replacing
    result_holder = {}

    def _wrap(*a, **k):
        r = calibrate_fixed(*a, **k)
        result_holder["r"] = r
        return r

    cal.calibrate = _wrap
    try:
        cal.main()
    except SystemExit:
        pass
    OUT.write_text(json.dumps(result_holder.get("r") or {"error": "failed"}, indent=2))
    print("Wrote", OUT)
    return 0 if result_holder.get("r") else 1


if __name__ == "__main__":
    raise SystemExit(main())
