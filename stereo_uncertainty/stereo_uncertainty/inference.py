"""Per-frame inference helpers (used by viz, evaluate, clinical_demo).

Loads a trained checkpoint and predicts, for a single index record, the
per-pixel (2D) or per-point (3D) error ``mu`` and calibrated uncertainty
``sigma`` together with the 3D coordinates needed to lift predictions into the
grasp-space point cloud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch

from . import config
from . import engine
from . import features as featmod
from . import scared_io as sio


def load_checkpoint(ckpt_path, device) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = engine.build_model(ckpt["variant"], ckpt.get("cfg", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model": model,
        "variant": ckpt["variant"],
        "method": ckpt.get("method"),
        "sigma_scale": float(ckpt.get("sigma_scale", 1.0)),
        "cfg": ckpt.get("cfg", {}),
    }


@torch.no_grad()
def predict_2d(model, rec, device, out_size=(256, 320), sigma_scale=1.0,
               scale_factor=None) -> Dict[str, np.ndarray]:
    """Predict full-resolution mu/sigma for a 2D model and lift to 3D."""
    scale_factor = scale_factor or config.DISP_SCALE_FACTOR
    feats, valid, Q = featmod.build_from_record(rec, scale_factor=scale_factor)
    H, W = valid.shape
    h, w = out_size
    feats_s = cv2.resize(feats, (w, h), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(feats_s.transpose(2, 0, 1)[None]).float().to(device)
    mu, log_var = model(x)
    mu = mu[0].cpu().numpy()
    sigma = np.exp(0.5 * log_var[0].cpu().numpy()) * sigma_scale
    mu_full = cv2.resize(mu, (W, H), interpolation=cv2.INTER_LINEAR)
    sigma_full = cv2.resize(sigma, (W, H), interpolation=cv2.INTER_LINEAR)

    pred_disp = sio.load_subpix_png(rec["pred_disp"], scale_factor=scale_factor)
    img3d, _ = featmod.reproject_points(
        np.where(pred_disp > 0, pred_disp, np.nan).astype(np.float32), Q
    )
    return {
        "mu": mu_full, "sigma": sigma_full, "valid": valid, "img3d": img3d,
    }


@torch.no_grad()
def predict_3d(model, rec, device, n_points=20000, sigma_scale=1.0,
               scale_factor=None) -> Dict[str, np.ndarray]:
    """Predict per-point mu/sigma for a 3D model over valid points of a frame."""
    scale_factor = scale_factor or config.DISP_SCALE_FACTOR
    feats, valid, Q = featmod.build_from_record(rec, scale_factor=scale_factor)
    pred_disp = sio.load_subpix_png(rec["pred_disp"], scale_factor=scale_factor)
    img3d, geo_valid = featmod.reproject_points(
        np.where(pred_disp > 0, pred_disp, np.nan).astype(np.float32), Q
    )
    vmask = (valid & geo_valid).reshape(-1)
    xyz_all = img3d.reshape(-1, 3)
    pfeat_all = feats.reshape(-1, feats.shape[-1])
    idx = np.where(vmask)[0]
    if idx.size == 0:
        return {"xyz": np.zeros((0, 3)), "mu": np.zeros(0), "sigma": np.zeros(0)}
    if idx.size > n_points:
        idx = np.random.choice(idx, n_points, replace=False)

    xyz = xyz_all[idx].astype(np.float32)
    center = xyz.mean(0)
    scale = float(np.linalg.norm(xyz - center, axis=1).max() + 1e-6)
    xyz_n = (xyz - center) / scale
    pts = np.concatenate([xyz_n, pfeat_all[idx]], axis=1)[None]
    pts_t = torch.from_numpy(pts).float().to(device)
    mu, log_var = model(pts_t)
    mu = mu[0].cpu().numpy()
    sigma = np.exp(0.5 * log_var[0].cpu().numpy()) * sigma_scale
    return {"xyz": xyz, "mu": mu, "sigma": sigma}
