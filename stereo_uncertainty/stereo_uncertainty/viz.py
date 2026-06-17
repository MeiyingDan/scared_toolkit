"""Visualisation: lift predicted uncertainty back into a coloured point cloud.

Two colourings, matching the Step-1 convention:
- ``turbo`` : continuous TURBO colormap over the value (mm), clipped at ``vmax``
- ``band``  : discrete clinical bands green/yellow/red (see :mod:`calibration`)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import calibration
from . import ply_io


def colorize_turbo(values_mm, vmax: float = 10.0) -> np.ndarray:
    clipped = np.clip(values_mm, 0.0, vmax)
    norm = (clipped / max(vmax, 1e-6) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(norm.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return bgr[:, ::-1].copy()  # -> RGB


def save_uncertainty_cloud(xyz, values_mm, out_path, mode="band",
                           vmax: float = 10.0, binary: bool = True) -> Path:
    """Write an Nx3 cloud coloured by per-point ``values_mm`` (mm)."""
    xyz = np.asarray(xyz, np.float32)
    values_mm = np.asarray(values_mm, np.float32)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(values_mm)
    xyz, values_mm = xyz[finite], values_mm[finite]
    if mode == "band":
        colors = calibration.band_to_color(calibration.error_to_band(values_mm))
    else:
        colors = colorize_turbo(values_mm, vmax=vmax)
    return ply_io.write_ply(out_path, xyz, colors, binary=binary)


def save_2d_prediction_cloud(pred: dict, out_path, mode="band",
                             vmax: float = 10.0) -> Path:
    """From an :func:`inference.predict_2d` result, save a valid-pixel cloud."""
    valid = pred["valid"]
    img3d = pred["img3d"]
    geo_valid = np.isfinite(img3d).all(axis=2)
    mask = valid & geo_valid
    xyz = img3d[mask]
    sigma = pred["sigma"][mask]
    return save_uncertainty_cloud(xyz, sigma, out_path, mode=mode, vmax=vmax)
