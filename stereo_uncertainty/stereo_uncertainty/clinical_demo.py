"""Todo 8 - clinical grasp-safety map + best-view / re-angle selection.

This turns the learned uncertainty into the two clinical artefacts motivating the
project:

1. Grasp-safety map
   For one frame, predict per-pixel uncertainty and render it as clinical bands
   (green <=2 mm, yellow <=10 mm, red otherwise), overlaid on the left image and
   lifted into a coloured point cloud.  For a user ROI it outputs a single
   "safe-to-grasp" score (mean predicted error in the ROI).

2. Best-view / re-angle selection
   Over the frames of a keyframe sequence (the surgeon "moving a little / changing
   angle"), predict the ROI uncertainty per frame and choose the view that
   minimises it.  We validate against the TRUE ROI error (from the Step-1 labels):
   does picking the lowest predicted-uncertainty view actually reduce real error,
   and how close is it to the oracle (lowest true-error) view?

Run::

    python -m stereo_uncertainty.clinical_demo --variant 2d --method IGEV \
        --dataset dataset_9 --keyframe keyframe_2 --roi 400 300 800 700
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import calibration
from . import config
from . import inference
from . import scared_io as sio
from . import viz
from .dataset_index import filter_records, load_index


def _roi_default(h, w):
    return [w // 4, h // 4, 3 * w // 4, 3 * h // 4]


def _roi_mean(arr, valid, roi):
    x0, y0, x1, y1 = roi
    sub = arr[y0:y1, x0:x1]
    sv = valid[y0:y1, x0:x1]
    if sv.sum() == 0:
        return float("nan"), 0
    return float(sub[sv].mean()), int(sv.sum())


def grasp_safety_map(pred, left_rgb, roi, out_png) -> dict:
    """Render a banded uncertainty overlay + ROI box; return ROI score."""
    sigma = pred["sigma"]
    valid = pred["valid"]
    bands = calibration.error_to_band(sigma)
    color = calibration.band_to_color(bands)  # HxWx3 RGB
    overlay = left_rgb.copy()
    m = valid
    overlay[m] = (0.45 * left_rgb[m] + 0.55 * color[m]).astype(np.uint8)

    x0, y0, x1, y1 = roi
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), 3)
    roi_sigma, n = _roi_mean(sigma, valid, roi)
    label = f"ROI mean unc: {roi_sigma:.2f} mm"
    cv2.putText(overlay, label, (x0, max(0, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {"roi_pred_mm": roi_sigma, "roi_valid_px": n,
            "safe": bool(roi_sigma <= config.GRASP_TAU_MM)}


def best_view(records, variant, model, device, sigma_scale, cfg, roi, out_dir):
    """Scan frames, pick min predicted-uncertainty view, validate vs true error."""
    rows = []
    for rec in sorted(records, key=lambda r: r["frame"]):
        if variant == "2d":
            pred = inference.predict_2d(
                model, rec, device,
                out_size=cfg.get("out_size", (256, 320)),
                sigma_scale=sigma_scale,
            )
        else:
            # 3D ROI is defined in image space; fall back to 2D-style reprojection
            # is non-trivial, so the best-view demo uses the 2D predictor.
            raise SystemExit("best-view demo currently supports --variant 2d")

        data = np.load(rec["err_npz"])
        err = data["err_mm"].astype(np.float32)
        err_valid = data["valid"].astype(bool)
        pred_roi, n_pred = _roi_mean(pred["sigma"], pred["valid"], roi)
        true_roi, n_true = _roi_mean(err, err_valid, roi)
        rows.append({
            "frame": rec["frame"],
            "pred_roi_mm": pred_roi,
            "true_roi_mm": true_roi,
            "n_valid": n_true,
        })

    valid_rows = [r for r in rows if np.isfinite(r["pred_roi_mm"])
                  and np.isfinite(r["true_roi_mm"])]
    if not valid_rows:
        raise RuntimeError("no valid ROI frames")

    chosen = min(valid_rows, key=lambda r: r["pred_roi_mm"])
    oracle = min(valid_rows, key=lambda r: r["true_roi_mm"])
    worst = max(valid_rows, key=lambda r: r["true_roi_mm"])

    # Plot predicted vs true ROI error across frames.
    frames = [int(r["frame"]) for r in valid_rows]
    pred_v = [r["pred_roi_mm"] for r in valid_rows]
    true_v = [r["true_roi_mm"] for r in valid_rows]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 3))
    plt.plot(frames, pred_v, "-o", ms=3, label="predicted ROI uncertainty (mm)")
    plt.plot(frames, true_v, "-s", ms=3, label="true ROI error (mm)")
    plt.axvline(int(chosen["frame"]), color="g", ls="--",
                label=f"chosen view (f{chosen['frame']})")
    plt.axvline(int(oracle["frame"]), color="k", ls=":",
                label=f"oracle view (f{oracle['frame']})")
    plt.xlabel("frame (viewpoint)")
    plt.ylabel("mm")
    plt.legend(fontsize=7)
    plt.title("Best-view selection by predicted uncertainty")
    plt.tight_layout()
    plot_path = out_dir / "best_view.png"
    plt.savefig(plot_path, dpi=130)
    plt.close()

    summary = {
        "chosen_frame": chosen["frame"],
        "chosen_true_roi_mm": chosen["true_roi_mm"],
        "oracle_frame": oracle["frame"],
        "oracle_true_roi_mm": oracle["true_roi_mm"],
        "worst_true_roi_mm": worst["true_roi_mm"],
        "regret_mm": chosen["true_roi_mm"] - oracle["true_roi_mm"],
        "improvement_vs_worst_mm": worst["true_roi_mm"] - chosen["true_roi_mm"],
        "n_frames": len(valid_rows),
        "plot": str(plot_path),
    }
    return rows, summary, chosen


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Clinical grasp-safety / best-view demo")
    p.add_argument("--variant", choices=["2d", "3d"], default="2d")
    p.add_argument("--method", default="IGEV")
    p.add_argument("--dataset", required=True)
    p.add_argument("--keyframe", required=True)
    p.add_argument("--index", type=Path, default=config.INDEX_DIR / "index.json")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--roi", type=int, nargs=4, default=None,
                   help="x0 y0 x1 y1 in full-res left-image coords.")
    p.add_argument("--device", default="auto")
    p.add_argument("--out_dir", type=Path, default=config.REPORT_DIR / "clinical")
    args = p.parse_args(argv)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    ckpt_path = args.ckpt or config.CKPT_DIR / f"{args.variant}_{args.method}.pt"
    ckpt = inference.load_checkpoint(ckpt_path, device)
    model, cfg, sigma_scale = ckpt["model"], ckpt["cfg"], ckpt["sigma_scale"]

    index = load_index(args.index)
    group = f"{args.dataset}/{args.keyframe}"
    recs = [r for r in filter_records(index, method=args.method)
            if r["group"] == group]
    if not recs:
        raise SystemExit(f"no records for {args.method} {group} in index")

    # Determine ROI from the first frame's resolution.
    first_left = sio.read_rgb(recs[0]["left"])
    h, w = first_left.shape[:2]
    roi = args.roi or _roi_default(h, w)

    out_dir = Path(args.out_dir) / f"{args.method}_{group.replace('/', '_')}"
    rows, summary, chosen = best_view(
        recs, args.variant, model, device, sigma_scale, cfg, roi, out_dir
    )

    # Grasp-safety map + uncertainty cloud for the chosen (best) view.
    chosen_rec = next(r for r in recs if r["frame"] == chosen["frame"])
    pred = inference.predict_2d(
        model, chosen_rec, device,
        out_size=cfg.get("out_size", (256, 320)), sigma_scale=sigma_scale,
    )
    left_rgb = sio.read_rgb(chosen_rec["left"])
    grasp = grasp_safety_map(pred, left_rgb, roi, out_dir / "grasp_safety_map.png")
    viz.save_2d_prediction_cloud(
        pred, out_dir / "uncertainty_cloud.ply", mode="band"
    )

    report = {
        "method": args.method, "group": group, "variant": args.variant,
        "roi": roi, "grasp_safety": grasp, "best_view": summary,
        "per_frame": rows,
    }
    with open(out_dir / "demo_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"grasp_safety": grasp, "best_view": summary}, indent=2))
    print(f"Artefacts -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
