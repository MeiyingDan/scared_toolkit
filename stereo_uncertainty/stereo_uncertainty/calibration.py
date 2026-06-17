"""Todo 6 - post-hoc uncertainty calibration + clinical error bands.

Variance scaling
----------------
The raw heteroscedastic ``sigma`` is often mis-scaled.  We fit a single positive
temperature ``s`` so that ``sigma_cal = s * sigma`` minimises the Gaussian NLL on
a held-out (validation) set.  This has a closed-form-ish optimum that we find by
a cheap 1-D search; it preserves the *ranking* of uncertainty (so AUSE/AUROC are
unchanged) while fixing the *scale* (so ECE / interval coverage improve).

Clinical bands
--------------
Map an error/uncertainty value (mm) to a discrete grasp-safety band:
    <= SAFE_MM  -> green  (0)
    <= RISKY_MM -> yellow (1)
    else        -> red    (2)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


def _nll(mu, sigma, target):
    sigma = np.clip(sigma, 1e-6, None)
    var = sigma ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (target - mu) ** 2 / var))


@dataclass
class VarianceScaler:
    """Single positive multiplicative scale on sigma."""

    scale: float = 1.0

    def fit(self, mu, sigma, target, n_grid: int = 400):
        mu = np.asarray(mu, np.float64)
        sigma = np.asarray(sigma, np.float64)
        target = np.asarray(target, np.float64)
        # Search log-scale in a broad range; robust and dependency-free.
        scales = np.logspace(-2, 2, n_grid)
        best_s, best_nll = 1.0, np.inf
        for s in scales:
            val = _nll(mu, s * sigma, target)
            if val < best_nll:
                best_nll, best_s = val, s
        self.scale = float(best_s)
        return self

    def transform(self, sigma):
        return self.scale * np.asarray(sigma)


def error_to_band(value_mm, safe_mm: float = None, risky_mm: float = None):
    """Map mm value(s) to band index 0/1/2 (green/yellow/red)."""
    safe_mm = config.SAFE_MM if safe_mm is None else safe_mm
    risky_mm = config.RISKY_MM if risky_mm is None else risky_mm
    value_mm = np.asarray(value_mm)
    band = np.zeros_like(value_mm, dtype=np.int32)
    band[value_mm > safe_mm] = 1
    band[value_mm > risky_mm] = 2
    return band


# Green / Yellow / Red as RGB for visualisation.
BAND_COLORS = np.array(
    [[0, 180, 0], [255, 200, 0], [220, 0, 0]], dtype=np.uint8
)


def band_to_color(band):
    return BAND_COLORS[np.asarray(band).astype(np.int32)]
