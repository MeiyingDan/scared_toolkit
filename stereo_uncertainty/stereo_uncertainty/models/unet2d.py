"""Todo 4 - 2D pixel-wise uncertainty network (compact U-Net).

A lightweight encoder-decoder that maps the per-pixel feature stack to two
output channels: ``mu`` (predicted error, mm) and ``log_var`` (log variance).
``mu`` is passed through softplus so the error estimate is non-negative.

Kept dependency-free (plain ``torch.nn``) so it trains on CPU for smoke tests
and on GPU for full runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.bottleneck = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.head = nn.Conv2d(base, 2, 1)

    @staticmethod
    def _pad_to(x, ref):
        dh = ref.shape[-2] - x.shape[-2]
        dw = ref.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [0, dw, 0, dh])
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([self._pad_to(d3, e3), e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self._pad_to(d2, e2), e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self._pad_to(d1, e1), e1], dim=1))

        out = self.head(d1)
        mu = F.softplus(out[:, 0])      # (B, H, W) non-negative error estimate
        log_var = out[:, 1]             # (B, H, W)
        return mu, log_var
