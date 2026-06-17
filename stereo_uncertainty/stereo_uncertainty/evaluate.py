"""Todo 7 - evaluation suite + 2D-vs-3D + cross-domain comparison.

For a trained checkpoint this computes the full metric bundle on the test split
(MAE/RMSE, Spearman, AUSE, ECE, grasp AUROC/AUPRC), both BEFORE and AFTER the
variance-scaling calibration learned on val, measures inference runtime, and
saves sparsification + reliability plots.

Cross-domain (D4D) is handled simply by passing a different ``--index`` built on
D4D records (once D4D predictions + GT are available); no code change needed.

Run::

    python -m stereo_uncertainty.evaluate --variant 2d --method IGEV
    python -m stereo_uncertainty.evaluate --variant 3d --method IGEV
    python -m stereo_uncertainty.evaluate --aggregate     # build comparison.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config
from . import engine
from . import inference
from . import metrics as M
from .calibration import VarianceScaler
from .dataset_index import filter_records, load_index


def _plots(mu, sigma, target, tag: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sparsification
    fracs, c_pred, c_oracle = M.sparsification(sigma, target)
    plt.figure(figsize=(4, 3))
    plt.plot(fracs, c_pred, "-o", ms=3, label="by predicted uncertainty")
    plt.plot(fracs, c_oracle, "-s", ms=3, label="oracle (by true error)")
    plt.xlabel("fraction removed")
    plt.ylabel("mean error of remaining (mm)")
    plt.title(f"Sparsification - {tag}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    sp = out_dir / f"sparsification_{tag}.png"
    plt.savefig(sp, dpi=130)
    plt.close()

    # Reliability (calibration) curve
    sig = np.clip(sigma, 1e-6, None)
    z = np.abs(target - mu) / sig
    from scipy.stats import norm
    ps = np.linspace(0.05, 0.95, 10)
    emp = [float(np.mean(z <= norm.ppf(0.5 + p / 2.0))) for p in ps]
    plt.figure(figsize=(3.5, 3.5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.plot(ps, emp, "-o", ms=3, label="model")
    plt.xlabel("expected coverage")
    plt.ylabel("observed coverage")
    plt.title(f"Calibration - {tag}")
    plt.legend(fontsize=7)
    plt.tight_layout()
    rp = out_dir / f"calibration_{tag}.png"
    plt.savefig(rp, dpi=130)
    plt.close()
    return {"sparsification_png": str(sp), "calibration_png": str(rp)}


@torch.no_grad()
def measure_runtime(variant, model, loader, device, max_batches=5) -> float:
    model.eval()
    times = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        t0 = time.time()
        engine.forward_batch(variant, model, batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0))
    if not times:
        return float("nan")
    bs = loader.batch_size or 1
    return float(np.mean(times) / bs * 1000.0)  # ms per sample


def evaluate(variant, method, index_path, ckpt_path, device_name="auto",
             split="test", tau=None) -> dict:
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_name == "auto" else torch.device(device_name))
    tau = config.GRASP_TAU_MM if tau is None else tau

    ckpt = inference.load_checkpoint(ckpt_path, device)
    cfg = ckpt["cfg"]
    model = ckpt["model"]
    sigma_scale = ckpt["sigma_scale"]

    index = load_index(index_path)
    recs = filter_records(index, split, method)
    if not recs:
        raise RuntimeError(f"no '{split}' records for method {method}")
    loader = engine.build_loader(variant, recs, cfg, train=False)

    preds = engine.gather_predictions(variant, model, loader, device)
    mu, sigma_raw, target = preds["mu"], preds["sigma"], preds["target"]
    if mu.size == 0:
        raise RuntimeError("no valid predictions gathered")

    sigma_cal = sigma_raw * sigma_scale

    raw = M.all_metrics(mu, sigma_raw, target, tau)
    cal = M.all_metrics(mu, sigma_cal, target, tau)
    runtime_ms = measure_runtime(variant, model, loader, device)
    n_params = sum(p.numel() for p in model.parameters())

    tag = f"{variant}_{method}_{split}"
    plot_paths = _plots(mu, sigma_cal, target, tag, config.REPORT_DIR / "plots")

    report = {
        "variant": variant,
        "method": method,
        "split": split,
        "n_valid_points": int(mu.size),
        "tau_mm": float(tau),
        "sigma_scale": float(sigma_scale),
        "n_params_M": round(n_params / 1e6, 3),
        "runtime_ms_per_sample": round(runtime_ms, 2),
        "metrics_raw": raw,
        "metrics_calibrated": cal,
        "plots": plot_paths,
    }
    return report


def aggregate(report_dir: Path) -> Path:
    """Collect all eval_*.json into a single comparison CSV (2D vs 3D, methods)."""
    import pandas as pd
    rows = []
    for jp in sorted(report_dir.glob("eval_*.json")):
        with open(jp) as f:
            r = json.load(f)
        row = {
            "variant": r["variant"], "method": r["method"], "split": r["split"],
            "n_params_M": r["n_params_M"],
            "runtime_ms": r["runtime_ms_per_sample"],
        }
        for k, v in r["metrics_calibrated"].items():
            row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
    df = pd.DataFrame(rows)
    out = report_dir / "comparison.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nComparison table -> {out}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate Step-2 uncertainty network")
    p.add_argument("--variant", choices=["2d", "3d"])
    p.add_argument("--method", default="IGEV")
    p.add_argument("--index", type=Path, default=config.INDEX_DIR / "index.json")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate existing eval_*.json into comparison.csv.")
    args = p.parse_args(argv)

    if args.aggregate:
        aggregate(config.REPORT_DIR)
        return 0

    if args.variant is None:
        p.error("--variant is required unless --aggregate is set")
    ckpt = args.ckpt or config.CKPT_DIR / f"{args.variant}_{args.method}.pt"
    report = evaluate(args.variant, args.method, args.index, ckpt,
                      args.device, args.split, args.tau)

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORT_DIR / f"eval_{args.variant}_{args.method}_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["metrics_calibrated"], indent=2))
    print(f"runtime {report['runtime_ms_per_sample']} ms/sample, "
          f"{report['n_params_M']}M params")
    print(f"Report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
