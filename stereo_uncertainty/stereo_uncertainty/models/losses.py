"""Heteroscedastic regression loss shared by both variants.

The network outputs two quantities per pixel/point:
- ``mu``      : the predicted error magnitude (mm), and
- ``log_var`` : the log predictive variance (aleatoric uncertainty).

We minimise the Gaussian negative log-likelihood of the observed Step-1 error
``e`` under N(mu, sigma^2):

    NLL = 0.5 * exp(-log_var) * (e - mu)^2 + 0.5 * log_var

This jointly learns an error estimate (``mu``) and a calibrated-ish uncertainty
(``sigma = exp(0.5*log_var)``).  Everything is masked by ``valid``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def heteroscedastic_nll(mu, log_var, target, valid, log_var_min=-6.0,
                        log_var_max=8.0):
    """Masked Gaussian NLL. All tensors broadcast to the same shape as target."""
    log_var = torch.clamp(log_var, log_var_min, log_var_max)
    inv_var = torch.exp(-log_var)
    nll = 0.5 * inv_var * (target - mu) ** 2 + 0.5 * log_var
    valid = valid.to(nll.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (nll * valid).sum() / denom


def masked_l1(mu, target, valid):
    """Masked L1 between predicted and true error (interpretable mm metric)."""
    valid = valid.to(mu.dtype)
    denom = valid.sum().clamp_min(1.0)
    return (F.l1_loss(mu, target, reduction="none") * valid).sum() / denom
