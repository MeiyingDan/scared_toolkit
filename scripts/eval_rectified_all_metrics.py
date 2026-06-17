#!/usr/bin/env python3
"""Unified rectified evaluation for SCARED dataset 8/9.

All metrics are computed in the **rectified coordinate system**:
- Disparity space (GT ``data/disparity`` vs pred ``data/disparity``):
  EPE (px), D1 (%), bad3 (%), bad5 (%)
- Depth space (GT ``data/depthmap_rectified`` vs depth from pred disparity via Q):
  MAE (mm), RMSE (mm)

Pixel errors are pooled within each keyframe (micro-average over valid pixels),
then keyframe rows are written; an AVG row averages keyframe-level metrics.

D1 follows the KITTI / OpenStereo definition:
  |pred - gt| > 3 px  AND  |pred - gt| / |gt| > 0.05
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import scaredtk.convertions as cvt
import scaredtk.io as sio

METHOD_NAME_OVERRIDES = {
    "HRS_results_testres0.5": "HRS_t05",
    "HRS_results_testres1": "HRS_t10",
}


def load_stereo_calib_json(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def parse_node(node):
        if isinstance(node, dict) and node.get("type_id") == "opencv-matrix":
            rows = int(node["rows"])
            cols = int(node["cols"])
            data = np.asarray(node["data"], dtype=np.float64)
            return data.reshape(rows, cols)
        return node

    return {k: parse_node(v) for k, v in raw.items()}


def load_valid_ids(valid_csv: Path) -> List[int]:
    valid_arr = np.loadtxt(valid_csv, delimiter=",")
    if np.ndim(valid_arr) == 0:
        return [int(valid_arr)]
    return [int(v) for v in valid_arr.tolist()]


def discover_latest_runs(results_root: Path, methods: Optional[List[str]] = None) -> Dict[str, Path]:
    excluded = {"depth_color_examples", "examples", "plots", "tables_like_scared", "pointcloud_examples"}
    runs: Dict[str, Path] = {}
    for candidate in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        if candidate.name in excluded or candidate.name.startswith("dataset_"):
            continue
        run_dirs = sorted([p for p in candidate.iterdir() if p.is_dir()])
        if not run_dirs:
            continue
        latest = run_dirs[-1]
        if not any((latest / d).exists() for d in ["dataset_8", "dataset_9"]):
            continue

        method_name = METHOD_NAME_OVERRIDES.get(candidate.name, candidate.name)
        if method_name.endswith("_results"):
            method_name = method_name[: -len("_results")]
        if methods and method_name not in methods:
            continue
        runs[method_name] = latest
    return runs


def disparity_valid_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.isfinite(gt) & np.isfinite(pred)


def depth_valid_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.isfinite(gt) & np.isfinite(pred)


def compute_disparity_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    valid = disparity_valid_mask(gt, pred)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {}

    err = np.abs(gt - pred)
    epe = float(np.sum(err[valid]) / n)

    gt_v = gt[valid]
    err_v = err[valid]
    d1 = float(np.sum((err_v > 3.0) & (err_v / np.maximum(np.abs(gt_v), 1e-6) > 0.05)) / n * 100.0)
    bad3 = float(np.sum(err_v > 3.0) / n * 100.0)
    bad5 = float(np.sum(err_v > 5.0) / n * 100.0)
    return {"EPE_px": epe, "D1_%": d1, "bad3_%": bad3, "bad5_%": bad5}


def compute_depth_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    valid = depth_valid_mask(gt, pred)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {}

    err = gt[valid] - pred[valid]
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"MAE_mm": mae, "RMSE_mm": rmse}


def eval_keyframe(
    gt_root: Path,
    pred_root: Path,
    dataset: str,
    keyframe: str,
    scale_factor: float,
) -> Optional[Dict[str, float]]:
    gt_kf = gt_root / dataset / keyframe
    pred_kf = pred_root / dataset / keyframe
    gt_disp_dir = gt_kf / "data" / "disparity"
    gt_depth_dir = gt_kf / "data" / "depthmap_rectified"
    pred_disp_dir = pred_kf / "data" / "disparity"
    valid_csv = gt_kf / "valid.csv"
    calib_path = gt_kf / "stereo_calib.json"

    if not all(p.is_dir() for p in [gt_disp_dir, gt_depth_dir, pred_disp_dir]):
        return None
    if not valid_csv.is_file() or not calib_path.is_file():
        return None

    calib = load_stereo_calib_json(calib_path)
    Q = calib["Q"].astype(np.float64)
    valid_ids = load_valid_ids(valid_csv)
    gt_disp_files = sorted([p for p in gt_disp_dir.iterdir() if p.suffix.lower() == ".png"])
    gt_depth_files = sorted([p for p in gt_depth_dir.iterdir() if p.suffix.lower() == ".png"])

    disp_stats = {"n": 0, "epe": 0.0, "d1": 0.0, "bad3": 0.0, "bad5": 0.0}
    depth_stats = {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0}
    frames = 0

    for frame_id in valid_ids:
        pred_disp_path = pred_disp_dir / f"{frame_id:06d}.png"
        if frame_id < 0 or frame_id >= len(gt_disp_files) or not pred_disp_path.is_file():
            continue

        gt_disp = sio.load_subpix_png(gt_disp_files[frame_id], scale_factor=scale_factor).astype(np.float32)
        pred_disp = sio.load_subpix_png(pred_disp_path, scale_factor=scale_factor).astype(np.float32)
        gt_depth = sio.load_subpix_png(gt_depth_files[frame_id], scale_factor=scale_factor).astype(np.float32)
        pred_depth = cvt.disparity_to_depthmap(pred_disp, Q)

        dmet = compute_disparity_metrics(gt_disp, pred_disp)
        if dmet:
            n = int(np.count_nonzero(disparity_valid_mask(gt_disp, pred_disp)))
            disp_stats["n"] += n
            disp_stats["epe"] += dmet["EPE_px"] * n
            disp_stats["d1"] += dmet["D1_%"] * n / 100.0
            disp_stats["bad3"] += dmet["bad3_%"] * n / 100.0
            disp_stats["bad5"] += dmet["bad5_%"] * n / 100.0

        met = compute_depth_metrics(gt_depth, pred_depth)
        if met:
            valid = depth_valid_mask(gt_depth, pred_depth)
            n = int(np.count_nonzero(valid))
            err = gt_depth[valid] - pred_depth[valid]
            depth_stats["n"] += n
            depth_stats["abs_sum"] += float(np.sum(np.abs(err)))
            depth_stats["sq_sum"] += float(np.sum(err ** 2))

        frames += 1

    if frames == 0 or disp_stats["n"] == 0 or depth_stats["n"] == 0:
        return None

    dn = disp_stats["n"]
    depth_n = depth_stats["n"]
    return {
        "frames": frames,
        "EPE_px": disp_stats["epe"] / dn,
        "D1_%": disp_stats["d1"] / dn * 100.0,
        "bad3_%": disp_stats["bad3"] / dn * 100.0,
        "bad5_%": disp_stats["bad5"] / dn * 100.0,
        "MAE_mm": depth_stats["abs_sum"] / depth_n,
        "RMSE_mm": float(np.sqrt(depth_stats["sq_sum"] / depth_n)),
    }


def eval_method(
    gt_root: Path,
    pred_root: Path,
    method_name: str,
    datasets: List[str],
    scale_factor: float,
) -> List[dict]:
    rows: List[dict] = []
    for dataset in datasets:
        for ki in range(5):
            keyframe = f"keyframe_{ki}"
            metrics = eval_keyframe(gt_root, pred_root, dataset, keyframe, scale_factor)
            if metrics is None:
                print(f"[warn] skip {method_name} {dataset}/{keyframe}", file=sys.stderr)
                continue
            row = {
                "method": method_name,
                "dataset": dataset,
                "keyframe": keyframe,
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            rows.append(row)
            print(
                f"{method_name:18s} {dataset} {keyframe}: "
                f"EPE={row['EPE_px']:.3f} D1={row['D1_%']:.2f} "
                f"bad3={row['bad3_%']:.2f} bad5={row['bad5_%']:.2f} "
                f"MAE={row['MAE_mm']:.3f} RMSE={row['RMSE_mm']:.3f}"
            )

    if not rows:
        return rows

    avg = {"method": method_name, "dataset": "AVG", "keyframe": "AVG", "frames": sum(r["frames"] for r in rows)}
    for col in ["EPE_px", "D1_%", "bad3_%", "bad5_%", "MAE_mm", "RMSE_mm"]:
        avg[col] = round(float(np.mean([r[col] for r in rows])), 3)
    rows.append(avg)
    return rows


def write_csv(rows: List[dict], output: Path) -> None:
    if not rows:
        return
    fieldnames = [
        "method", "dataset", "keyframe", "frames",
        "EPE_px", "D1_%", "bad3_%", "bad5_%", "MAE_mm", "RMSE_mm",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified rectified evaluation (disparity + depth metrics)")
    parser.add_argument("gt_root", type=Path, help="SCARED_DATASET_processed root")
    parser.add_argument(
        "pred_root",
        type=Path,
        nargs="?",
        default=None,
        help="Single method run root (omit when using --methods_results_root)",
    )
    parser.add_argument(
        "--methods_results_root",
        type=Path,
        default=None,
        help="Evaluate latest run for each method under this folder (e.g. results_d8d9/methods_results)",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods to evaluate")
    parser.add_argument("--method_name", default=None, help="Label for single-run mode")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV for single-run mode")
    parser.add_argument(
        "--combined_output",
        type=Path,
        default=None,
        help="Combined CSV when evaluating multiple methods",
    )
    parser.add_argument("--scale_factor", type=float, default=128.0)
    parser.add_argument("--datasets", nargs="+", default=["dataset_8", "dataset_9"])
    args = parser.parse_args()

    if args.methods_results_root is not None:
        runs = discover_latest_runs(args.methods_results_root, args.methods)
        if not runs:
            print("No method runs found.", file=sys.stderr)
            return 1
        print("Discovered method runs:")
        for name, run_dir in sorted(runs.items()):
            print(f"  - {name}: {run_dir}")

        combined: List[dict] = []
        for method_name, run_dir in sorted(runs.items()):
            combined.extend(
                eval_method(args.gt_root, run_dir, method_name, args.datasets, args.scale_factor)
            )
            per_method_out = run_dir / "all_metrics.csv"
            method_rows = [r for r in combined if r["method"] == method_name]
            write_csv(method_rows, per_method_out)

        out = args.combined_output or (
            args.methods_results_root.parent / "eval" / "rectified" / "all_metrics_per_keyframe.csv"
        )
        write_csv(combined, out)
        return 0

    if args.pred_root is None:
        parser.error("Provide pred_root or --methods_results_root")
    method_name = args.method_name or args.pred_root.name
    output = args.output or (args.pred_root / "all_metrics.csv")
    rows = eval_method(args.gt_root, args.pred_root, method_name, args.datasets, args.scale_factor)
    write_csv(rows, output)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
