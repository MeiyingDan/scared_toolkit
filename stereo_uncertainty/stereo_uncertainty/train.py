"""Training loop for both uncertainty variants (todo 4 / todo 5).

Trains a per-method model (default) by filtering the index to one method, then
optimising the heteroscedastic NLL (+ small L1) on the train split and selecting
the checkpoint with the best validation MAE.

Run::

    python -m stereo_uncertainty.train --variant 2d --method IGEV --epochs 20
    python -m stereo_uncertainty.train --variant 3d --method IGEV --epochs 20
    python -m stereo_uncertainty.train --variant 2d --method IGEV --max_iters 2 \
        --epochs 1 --batch_size 1 --out_size 96 120          # CPU smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import config
from . import engine
from . import metrics as M
from .calibration import VarianceScaler
from .dataset_index import filter_records, load_index


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(cfg: dict) -> dict:
    variant = cfg["variant"]
    device = _device(cfg.get("device", "auto"))

    index = load_index(Path(cfg["index"]))
    train_recs = filter_records(index, "train", cfg.get("method"))
    val_recs = filter_records(index, "val", cfg.get("method"))
    if not train_recs:
        raise RuntimeError("no training records (check index / method filter)")
    if not val_recs:
        val_recs = train_recs[: max(1, len(train_recs) // 5)]

    train_loader = engine.build_loader(variant, train_recs, cfg, train=True)
    val_loader = engine.build_loader(variant, val_recs, cfg, train=False)

    model = engine.build_model(variant, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{variant}/{cfg.get('method')}] model params: {n_params/1e6:.2f}M "
          f"| train {len(train_recs)} val {len(val_recs)} | device {device}")

    max_iters = cfg.get("max_iters")
    best_mae = np.inf
    history = []
    ckpt_path = Path(cfg["ckpt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.get("epochs", 20)):
        model.train()
        t0 = time.time()
        running = []
        for it, batch in enumerate(train_loader):
            if max_iters is not None and it >= max_iters:
                break
            mu, log_var, target, valid = engine.forward_batch(
                variant, model, batch, device
            )
            loss, parts = engine.batch_loss(
                mu, log_var, target, valid, cfg.get("l1_weight", 0.1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(parts["l1"])

        preds = engine.gather_predictions(variant, model, val_loader, device)
        val_mae = M.mae(preds["mu"], preds["target"]) if preds["mu"].size else np.inf
        val_ause = M.ause(preds["sigma"], preds["target"]) if preds["mu"].size else 0.0
        history.append({
            "epoch": epoch,
            "train_l1": float(np.mean(running)) if running else 0.0,
            "val_mae": float(val_mae),
            "val_ause": float(val_ause),
            "sec": round(time.time() - t0, 1),
        })
        print(f"  epoch {epoch:03d} | train L1 {history[-1]['train_l1']:.3f} "
              f"| val MAE {val_mae:.3f} mm | val AUSE {val_ause:.4f} "
              f"| {history[-1]['sec']}s")

        if val_mae < best_mae:
            best_mae = val_mae
            # Fit variance scaling on validation predictions for calibration.
            scaler = VarianceScaler()
            if preds["mu"].size:
                scaler.fit(preds["mu"], preds["sigma"], preds["target"])
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": variant,
                    "method": cfg.get("method"),
                    "cfg": cfg,
                    "sigma_scale": scaler.scale,
                    "val_mae": float(val_mae),
                },
                ckpt_path,
            )

    print(f"  best val MAE {best_mae:.3f} mm -> {ckpt_path}")
    report = {"history": history, "best_val_mae": float(best_mae),
              "ckpt": str(ckpt_path)}
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train Step-2 uncertainty network")
    p.add_argument("--variant", choices=["2d", "3d"], required=True)
    p.add_argument("--method", default="IGEV")
    p.add_argument("--index", type=Path, default=config.INDEX_DIR / "index.json")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1_weight", type=float, default=0.1)
    p.add_argument("--out_size", type=int, nargs="+", default=None,
                   help="2D output H W (or a single int for square).")
    p.add_argument("--n_points", type=int, default=8192)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    cfg = vars(args).copy()
    if args.out_size is not None:
        cfg["out_size"] = (
            tuple(args.out_size) if len(args.out_size) == 2
            else (args.out_size[0], args.out_size[0])
        )
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.ckpt is None:
        cfg["ckpt"] = config.CKPT_DIR / f"{args.variant}_{args.method}.pt"

    report = train(cfg)
    rep_path = config.REPORT_DIR / f"train_{args.variant}_{args.method}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Training report -> {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
