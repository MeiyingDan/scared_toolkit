"""Shared training/eval engine: model + loader builders and batch forward.

Used by both :mod:`train` and :mod:`evaluate` so the two variants ("2d"/"3d")
go through identical plumbing.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .datasets import Stereo2DErrorDataset, Stereo3DErrorDataset
from .models import UNet2D, PointNetSeg, heteroscedastic_nll, masked_l1


def build_model(variant: str, cfg: dict) -> torch.nn.Module:
    if variant == "2d":
        return UNet2D(in_ch=config.N_FEATURES, base=cfg.get("base", 32))
    if variant == "3d":
        in_ch = 3 + config.N_FEATURES
        return PointNetSeg(in_ch=in_ch, feat_dim=cfg.get("feat_dim", 512))
    raise ValueError(f"unknown variant {variant}")


def build_dataset(variant: str, records: List[dict], cfg: dict, train: bool):
    if variant == "2d":
        return Stereo2DErrorDataset(
            records,
            out_size=cfg.get("out_size", (256, 320)),
            max_error_mm=cfg.get("max_error_mm", config.MAX_ERROR_MM),
            augment=train and cfg.get("augment", True),
        )
    return Stereo3DErrorDataset(
        records,
        n_points=cfg.get("n_points", 8192),
        max_error_mm=cfg.get("max_error_mm", config.MAX_ERROR_MM),
    )


def build_loader(variant, records, cfg, train: bool) -> DataLoader:
    ds = build_dataset(variant, records, cfg, train)
    return DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 2 if variant == "2d" else 4),
        shuffle=train,
        num_workers=cfg.get("num_workers", 0),
        drop_last=False,
    )


def forward_batch(variant, model, batch, device) -> Tuple:
    """Return ``(mu, log_var, target, valid)`` as flat 1-D tensors over valid entries."""
    if variant == "2d":
        feat = batch["feat"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        mu, log_var = model(feat)
    else:
        xyz = batch["xyz"].to(device)
        pfeat = batch["pfeat"].to(device)
        points = torch.cat([xyz, pfeat], dim=-1)
        target = batch["target"].to(device)
        valid = torch.ones_like(target)
        mu, log_var = model(points)
    return mu, log_var, target, valid


def batch_loss(mu, log_var, target, valid, l1_weight: float = 0.1):
    nll = heteroscedastic_nll(mu, log_var, target, valid)
    l1 = masked_l1(mu, target, valid)
    return nll + l1_weight * l1, {"nll": float(nll.item()), "l1": float(l1.item())}


@torch.no_grad()
def gather_predictions(variant, model, loader, device) -> Dict[str, np.ndarray]:
    """Run a loader and concatenate flat valid (mu, sigma, target) arrays."""
    model.eval()
    mus, sigmas, targets = [], [], []
    for batch in loader:
        mu, log_var, target, valid = forward_batch(variant, model, batch, device)
        v = valid > 0.5
        mus.append(mu[v].detach().cpu().numpy().ravel())
        sigmas.append(torch.exp(0.5 * log_var)[v].detach().cpu().numpy().ravel())
        targets.append(target[v].detach().cpu().numpy().ravel())
    return {
        "mu": np.concatenate(mus) if mus else np.zeros(0),
        "sigma": np.concatenate(sigmas) if sigmas else np.zeros(0),
        "target": np.concatenate(targets) if targets else np.zeros(0),
    }
