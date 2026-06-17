"""Torch datasets for the 2D (pixel-wise) and 3D (point-wise) variants.

Both datasets share the same supervision: the Step-1 error map (mm) as target,
masked by the intersection of (a) GT coverage from the error-map ``valid`` field
and (b) the feature validity (where the frozen model produced a disparity).

A single sample is built lazily from an index record:
- features via :func:`features.build_from_record`
- target/valid via the exported ``.npz`` error label
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config
from . import features as featmod


def _load_sample_full(rec: dict, scale_factor: float):
    """Return full-resolution ``(feat HxWxC, target HxW, valid HxW, img3d HxWx3)``."""
    feats, feat_valid, Q = featmod.build_from_record(rec, scale_factor=scale_factor)
    data = np.load(rec["err_npz"])
    err = data["err_mm"].astype(np.float32)
    err_valid = data["valid"].astype(bool)
    disp = data["disp"].astype(np.float32)

    valid = feat_valid & err_valid & np.isfinite(err)
    img3d, _ = featmod.reproject_points(
        np.where(disp > 0, disp, np.nan).astype(np.float32), Q
    )
    return feats, err, valid, img3d


# --------------------------------------------------------------------------- #
# 2D pixel-wise dataset                                                        #
# --------------------------------------------------------------------------- #
class Stereo2DErrorDataset(Dataset):
    """Per-pixel error/uncertainty regression dataset.

    Returns dict with:
        feat   : (C, H, W) float32 features
        target : (H, W) float32 error in mm
        valid  : (H, W) float32 in {0,1}
    """

    def __init__(
        self,
        records: List[dict],
        out_size=(256, 320),
        scale_factor: Optional[float] = None,
        max_error_mm: Optional[float] = None,
        augment: bool = False,
    ):
        self.records = records
        self.out_size = tuple(out_size)
        self.scale_factor = scale_factor or config.DISP_SCALE_FACTOR
        self.max_error_mm = max_error_mm or config.MAX_ERROR_MM
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        feats, err, valid, _ = _load_sample_full(rec, self.scale_factor)

        h, w = self.out_size
        feats = cv2.resize(feats, (w, h), interpolation=cv2.INTER_LINEAR)
        err = cv2.resize(err, (w, h), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(
            valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        err = np.clip(err, 0.0, self.max_error_mm)

        if self.augment and np.random.rand() < 0.5:
            feats = feats[:, ::-1].copy()
            err = err[:, ::-1].copy()
            valid = valid[:, ::-1].copy()

        feat_t = torch.from_numpy(feats.transpose(2, 0, 1).copy()).float()
        target_t = torch.from_numpy(err).float()
        valid_t = torch.from_numpy(valid.astype(np.float32))
        return {
            "feat": feat_t,
            "target": target_t,
            "valid": valid_t,
            "method": rec["method"],
            "frame": rec["frame"],
            "group": rec["group"],
        }


# --------------------------------------------------------------------------- #
# 3D point-wise dataset                                                        #
# --------------------------------------------------------------------------- #
class Stereo3DErrorDataset(Dataset):
    """Per-point error/uncertainty regression dataset.

    Returns dict with:
        xyz    : (N, 3) float32 normalised coordinates (centred, unit-scaled)
        pfeat  : (N, F) float32 per-point features (the 2D feature stack)
        target : (N,) float32 error in mm
        center : (3,) float32 / scale : () float32  (to lift back to mm space)
    """

    def __init__(
        self,
        records: List[dict],
        n_points: int = 8192,
        scale_factor: Optional[float] = None,
        max_error_mm: Optional[float] = None,
    ):
        self.records = records
        self.n_points = n_points
        self.scale_factor = scale_factor or config.DISP_SCALE_FACTOR
        self.max_error_mm = max_error_mm or config.MAX_ERROR_MM

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        feats, err, valid, img3d = _load_sample_full(rec, self.scale_factor)

        vmask = valid.reshape(-1)
        xyz_all = img3d.reshape(-1, 3)
        pfeat_all = feats.reshape(-1, feats.shape[-1])
        err_all = err.reshape(-1)

        idx = np.where(vmask)[0]
        if idx.size == 0:
            idx = np.arange(min(self.n_points, xyz_all.shape[0]))
        replace = idx.size < self.n_points
        sel = np.random.choice(idx, size=self.n_points, replace=replace)

        xyz = xyz_all[sel].astype(np.float32)
        pfeat = pfeat_all[sel].astype(np.float32)
        target = np.clip(err_all[sel], 0.0, self.max_error_mm).astype(np.float32)

        center = xyz.mean(axis=0)
        scale = float(np.linalg.norm(xyz - center, axis=1).max() + 1e-6)
        xyz_n = (xyz - center) / scale

        return {
            "xyz": torch.from_numpy(xyz_n).float(),
            "xyz_mm": torch.from_numpy(xyz).float(),
            "pfeat": torch.from_numpy(pfeat).float(),
            "target": torch.from_numpy(target).float(),
            "center": torch.from_numpy(center.astype(np.float32)),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "method": rec["method"],
            "frame": rec["frame"],
            "group": rec["group"],
        }
