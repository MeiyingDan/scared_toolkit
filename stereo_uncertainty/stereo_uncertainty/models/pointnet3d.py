"""Todo 5 - 3D point-wise uncertainty network (PointNet segmentation head).

A classic PointNet segmentation architecture (shared per-point MLPs + a global
max-pooled context feature) that predicts ``(mu, log_var)`` per point.  We use a
PointNet rather than KPConv/Minkowski to stay dependency-free (no
torch_geometric / spconv), while still operating directly in the 3D grasp space
and supporting permutation invariance and variable point counts.

Input  : ``points`` (B, N, Cin) with Cin = 3 (xyz) + F (per-point features)
Output : ``mu`` (B, N) non-negative error estimate, ``log_var`` (B, N)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp1d(channels):
    layers = []
    for i in range(len(channels) - 1):
        layers += [
            nn.Conv1d(channels[i], channels[i + 1], 1, bias=False),
            nn.BatchNorm1d(channels[i + 1]),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class PointNetSeg(nn.Module):
    def __init__(self, in_ch: int, feat_dim: int = 512):
        super().__init__()
        self.local = _mlp1d([in_ch, 64, 64])
        self.mid = _mlp1d([64, 128, feat_dim])
        # Per-point head consumes local (64) + global (feat_dim) context.
        self.head = nn.Sequential(
            nn.Conv1d(64 + feat_dim, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, points):
        # points: (B, N, Cin) -> (B, Cin, N)
        x = points.transpose(1, 2).contiguous()
        local = self.local(x)                       # (B, 64, N)
        glob = self.mid(local)                       # (B, feat_dim, N)
        glob = torch.max(glob, dim=2, keepdim=True)[0]  # (B, feat_dim, 1)
        n = local.shape[2]
        glob_rep = glob.repeat(1, 1, n)              # (B, feat_dim, N)
        cat = torch.cat([local, glob_rep], dim=1)    # (B, 64+feat_dim, N)
        out = self.head(cat)                         # (B, 2, N)
        mu = F.softplus(out[:, 0])                   # (B, N)
        log_var = out[:, 1]                          # (B, N)
        return mu, log_var
