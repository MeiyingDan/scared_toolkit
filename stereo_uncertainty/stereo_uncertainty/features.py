"""Todo 3 - no-GT feature builder.

All features here are computable at deployment time (inside a pig/human) WITHOUT
ground truth.  They are the predictive signal the Step-2 network uses to estimate
the depth error of the frozen stereo model:

- ``r``,``g``,``b``  : left rectified colour (texture / specularity context)
- ``disparity``      : the frozen model's prediction (the thing we judge)
- ``depth``          : Z in mm (far surfaces triangulate less accurately)
- ``lr_residual``    : left-right photometric consistency after warping the right
                       image by the predicted disparity (strong proxy for matching
                       failure under occlusion / glare)
- ``disp_grad``      : disparity gradient magnitude (depth discontinuities = error)
- ``glare``          : specular-highlight mask (endoscopic glare breaks matching)

The same per-pixel features are reused for the 3D variant by attaching them as
per-point attributes after reprojection.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from . import scared_io as sio

# Normalisation constants (keep inputs ~O(1) for stable training).
DISP_NORM = 100.0   # typical SCARED disparities are tens of pixels
DEPTH_NORM = 100.0  # typical depths are ~50-300 mm


def _warp_right_to_left(left_gray: np.ndarray, right_gray: np.ndarray,
                        disparity: np.ndarray) -> np.ndarray:
    """Warp the right image into the left frame using the predicted disparity.

    For a rectified pair, the left pixel (x, y) matches the right pixel
    (x - disp, y).  The absolute photometric residual is small where the
    disparity is correct and large where matching failed.
    """
    h, w = left_gray.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    disp = np.nan_to_num(disparity, nan=0.0)
    map_x = (xs - disp).astype(np.float32)
    map_y = ys.astype(np.float32)
    warped = cv2.remap(
        right_gray, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    residual = np.abs(left_gray.astype(np.float32) - warped.astype(np.float32))
    return residual


def _glare_mask(left_rgb: np.ndarray) -> np.ndarray:
    """Soft specular-highlight score in [0,1] from HSV (high V, low S)."""
    hsv = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    glare = np.clip(v - s, 0.0, 1.0) * (v > 0.8)
    return glare.astype(np.float32)


def build_features(
    left_rgb: np.ndarray,
    right_img: np.ndarray,
    pred_disp: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(features HxWxN float32, valid HxW bool)``.

    ``pred_disp`` uses NaN for unknown pixels (as produced by ``load_subpix_png``).
    """
    h, w = pred_disp.shape[:2]
    if left_rgb.shape[:2] != (h, w):
        left_rgb = cv2.resize(left_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if right_img.shape[:2] != (h, w):
        right_img = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)

    left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = (
        cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        if right_img.ndim == 3 else right_img
    )

    depth = sio.depth_from_disparity(pred_disp, Q)  # mm, NaN invalid
    valid = np.isfinite(pred_disp) & (pred_disp > 0) & np.isfinite(depth)

    disp_f = np.nan_to_num(pred_disp, nan=0.0).astype(np.float32)
    depth_f = np.nan_to_num(depth, nan=0.0).astype(np.float32)

    lr_res = _warp_right_to_left(left_gray, right_gray, pred_disp) / 255.0

    gx = cv2.Sobel(disp_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(disp_f, cv2.CV_32F, 0, 1, ksize=3)
    disp_grad = np.sqrt(gx * gx + gy * gy)
    disp_grad = np.clip(disp_grad / 10.0, 0.0, 5.0)  # gentle squashing

    glare = _glare_mask(left_rgb)

    feats = np.stack(
        [
            left_rgb[:, :, 0].astype(np.float32) / 255.0,
            left_rgb[:, :, 1].astype(np.float32) / 255.0,
            left_rgb[:, :, 2].astype(np.float32) / 255.0,
            disp_f / DISP_NORM,
            depth_f / DEPTH_NORM,
            lr_res.astype(np.float32),
            disp_grad.astype(np.float32),
            glare,
        ],
        axis=-1,
    ).astype(np.float32)

    # Zero-out features on invalid pixels so the network sees a consistent "hole".
    feats[~valid] = 0.0
    assert feats.shape[-1] == config.N_FEATURES
    return feats, valid


def build_from_record(rec: dict, scale_factor: float = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a record's inputs and return ``(features, valid, Q)``.

    Reads left/right rectified images and the frozen-model disparity referenced
    by the index record produced in :mod:`dataset_index`.
    """
    scale_factor = scale_factor or config.DISP_SCALE_FACTOR
    left = sio.read_rgb(rec["left"])
    right = sio.read_rgb(rec["right"])
    Q = sio.load_stereo_calib(rec["calib"])["Q"]
    pred_disp = sio.load_subpix_png(rec["pred_disp"], scale_factor=scale_factor)
    if left is None or right is None:
        raise FileNotFoundError(f"missing stereo pair for {rec.get('frame')}")
    feats, valid = build_features(left, right, pred_disp, Q)
    return feats, valid, Q


def reproject_points(pred_disp: np.ndarray, Q: np.ndarray):
    """Return ``(img3d HxWx3, valid HxW)`` for the predicted disparity."""
    img3d = sio.disparity_to_img3d(pred_disp, Q)
    valid = np.isfinite(img3d).all(axis=2)
    return img3d, valid
