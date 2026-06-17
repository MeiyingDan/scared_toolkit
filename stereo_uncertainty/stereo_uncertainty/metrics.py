"""Todo 7 (core) - error-prediction and uncertainty-quality metrics.

All functions operate on flat 1-D numpy arrays of the VALID entries only:
- ``err_true`` : observed Step-1 error (mm)
- ``mu``       : predicted error (mm)
- ``sigma``    : predicted uncertainty / std (mm)

Provided metrics
----------------
- ``mae`` / ``rmse``                 : error-prediction accuracy
- ``spearman``                       : rank corr. of uncertainty vs true error
- ``sparsification`` / ``ause``      : does removing high-uncertainty pixels
                                       reduce error like the oracle?
- ``regression_ece``                 : Gaussian calibration error
- ``grasp_detection``                : AUROC/AUPRC for "unsafe" (err > tau)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.metrics import roc_auc_score, average_precision_score


def mae(mu, err_true) -> float:
    return float(np.mean(np.abs(mu - err_true)))


def rmse(mu, err_true) -> float:
    return float(np.sqrt(np.mean((mu - err_true) ** 2)))


def spearman(score, err_true) -> float:
    if np.std(score) < 1e-12 or np.std(err_true) < 1e-12:
        return 0.0
    rho, _ = spearmanr(score, err_true)
    return float(0.0 if np.isnan(rho) else rho)


def sparsification(score, err_true, n_steps: int = 20):
    """Return (fractions, curve_pred, curve_oracle).

    At each fraction f we DROP the f highest-``score`` entries and report the
    mean true error of the rest.  A good uncertainty removes the worst errors
    fastest, approaching the oracle (which drops by true error).
    """
    order_pred = np.argsort(-score)        # high uncertainty first
    order_oracle = np.argsort(-err_true)   # high true error first
    fracs = np.linspace(0.0, 0.95, n_steps)
    curve_pred, curve_oracle = [], []
    n = len(err_true)
    for f in fracs:
        k = int(f * n)
        keep_pred = order_pred[k:]
        keep_oracle = order_oracle[k:]
        curve_pred.append(float(np.mean(err_true[keep_pred])) if keep_pred.size else 0.0)
        curve_oracle.append(
            float(np.mean(err_true[keep_oracle])) if keep_oracle.size else 0.0
        )
    return fracs, np.array(curve_pred), np.array(curve_oracle)


def ause(score, err_true, n_steps: int = 20) -> float:
    """Area Under the Sparsification Error (lower is better)."""
    fracs, c_pred, c_oracle = sparsification(score, err_true, n_steps)
    denom = c_pred[0] if c_pred[0] > 1e-12 else 1.0
    sparsification_error = (c_pred - c_oracle) / denom
    return float(np.trapz(sparsification_error, fracs))


def regression_ece(mu, sigma, err_true, n_bins: int = 10) -> float:
    """Expected calibration error for a Gaussian predictive distribution.

    For each nominal coverage p, measure the empirical fraction of targets that
    fall within the central p prediction interval and average |empirical - p|.
    """
    sigma = np.clip(sigma, 1e-6, None)
    z = np.abs(err_true - mu) / sigma
    ps = np.linspace(0.05, 0.95, n_bins)
    gaps = []
    for p in ps:
        # central interval half-width in z for coverage p
        zt = norm.ppf(0.5 + p / 2.0)
        emp = float(np.mean(z <= zt))
        gaps.append(abs(emp - p))
    return float(np.mean(gaps))


def grasp_detection(score, err_true, tau: float) -> Dict[str, float]:
    """Detection metrics for 'unsafe' = err_true > tau, ranked by ``score``."""
    label = (err_true > tau).astype(np.int32)
    out = {"unsafe_frac": float(label.mean())}
    if label.min() == label.max():
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
        return out
    out["auroc"] = float(roc_auc_score(label, score))
    out["auprc"] = float(average_precision_score(label, score))
    return out


def all_metrics(mu, sigma, err_true, tau: float) -> Dict[str, float]:
    """Convenience bundle used by the eval suite."""
    return {
        "mae_mm": mae(mu, err_true),
        "rmse_mm": rmse(mu, err_true),
        "spearman_unc_err": spearman(sigma, err_true),
        "spearman_mu_err": spearman(mu, err_true),
        "ause": ause(sigma, err_true),
        "ece": regression_ece(mu, sigma, err_true),
        **{f"grasp_{k}": v for k, v in grasp_detection(sigma, err_true, tau).items()},
    }
