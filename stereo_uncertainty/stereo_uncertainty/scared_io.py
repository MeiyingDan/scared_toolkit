"""Minimal, self-contained SCARED I/O helpers.

We deliberately avoid importing ``scaredtk`` so the package can run inside the
``openstereo`` conda environment (which does not have the toolkit installed).
Only the few functions actually needed by the Step-2 pipeline are reimplemented
here; they are behaviour-compatible with ``scaredtk.io`` / ``scaredtk.convertions``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def load_subpix_png(path, scale_factor: float = 128.0) -> np.ndarray:
    """Load a float array stored as a 16-bit PNG (zeros -> NaN = unknown)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    img = cv2.imread(str(path), -1)
    if img is None:
        raise IOError(f"cv2 could not read {path}")
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.uint16:
        img = img / scale_factor
    img[img == 0] = np.nan
    return img


def _parse_opencv_node(node):
    if isinstance(node, dict) and node.get("type_id") == "opencv-matrix":
        rows = int(node["rows"])
        cols = int(node["cols"])
        data = np.array(node["data"], dtype=np.float64)
        return data.reshape(rows, cols)
    return node


def load_stereo_calib(path) -> Dict[str, np.ndarray]:
    """Load a ``stereo_calib.json`` written in OpenCV matrix format."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: _parse_opencv_node(v) for k, v in raw.items()}


def disparity_to_img3d(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Reproject a disparity map to an HxWx3 XYZ image; invalid pixels -> NaN."""
    disparity = disparity.astype(np.float32, copy=False)
    invalid = ~np.isfinite(disparity) | (disparity <= 0)
    disp_clean = np.nan_to_num(disparity)
    img3d = cv2.reprojectImageTo3D(disp_clean, Q.astype(np.float64))
    img3d[invalid] = np.nan
    img3d[~np.isfinite(img3d).all(axis=2)] = np.nan
    return img3d


def depth_from_disparity(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return the Z (depth) channel of the reprojected disparity (NaN invalid)."""
    return disparity_to_img3d(disparity, Q)[:, :, 2]


def read_rgb(path) -> np.ndarray | None:
    """Read an image as RGB uint8, or ``None`` if missing/unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_gray(path) -> np.ndarray | None:
    path = Path(path)
    if not path.is_file():
        return None
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if g is None else g
