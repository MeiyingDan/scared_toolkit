#!/usr/bin/env python3
"""Evaluate rectified depth MAE from predicted disparity PNGs.

Converts predicted disparity to rectified depth via ``disparity_to_depthmap(Q)``
and compares against GT ``depthmap_rectified`` (mm).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import scaredtk.convertions as cvt
import scaredtk.io as sio


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


def eval_run(
    gt_root: Path,
    pred_root: Path,
    method_name: str,
    scale_factor: float,
    bad_threshold: Optional[float],
    datasets: List[str],
) -> List[dict]:
    rows = []
    for dataset in datasets:
        for keyframe in [f"keyframe_{i}" for i in range(5)]:
            gt_kf = gt_root / dataset / keyframe
            pred_kf = pred_root / dataset / keyframe
            gt_depth_dir = gt_kf / "data" / "depthmap_rectified"
            pred_disp_dir = pred_kf / "data" / "disparity"
            valid_csv = gt_kf / "valid.csv"
            calib_path = gt_kf / "stereo_calib.json"

            if not gt_depth_dir.is_dir() or not pred_disp_dir.is_dir():
                print(f"[warn] skip {dataset}/{keyframe}: missing GT or pred dir", file=sys.stderr)
                continue
            if not valid_csv.is_file() or not calib_path.is_file():
                print(f"[warn] skip {dataset}/{keyframe}: missing valid.csv or calib", file=sys.stderr)
                continue

            calib = load_stereo_calib_json(calib_path)
            Q = calib["Q"].astype(np.float64)
            valid_ids = load_valid_ids(valid_csv)
            gt_files = sorted([p for p in gt_depth_dir.iterdir() if p.suffix.lower() == ".png"])

            mae_lst = []
            bad_lst = []
            for frame_id in valid_ids:
                disp_path = pred_disp_dir / f"{frame_id:06d}.png"
                if frame_id < 0 or frame_id >= len(gt_files) or not disp_path.is_file():
                    continue

                gt_depth = sio.load_subpix_png(gt_files[frame_id], scale_factor=scale_factor).astype(np.float32)
                disparity = sio.load_subpix_png(disp_path, scale_factor=scale_factor).astype(np.float32)
                pred_depth = cvt.disparity_to_depthmap(disparity, Q)

                error = np.abs(gt_depth - pred_depth)
                mae_lst.append(float(np.nanmean(error)))

                if bad_threshold is not None:
                    valid = ~np.isnan(gt_depth)
                    n_valid = int(np.count_nonzero(valid))
                    if n_valid > 0:
                        bad_lst.append(float(np.sum((error > bad_threshold) & valid) / n_valid * 100.0))

            if not mae_lst:
                print(f"[warn] skip {dataset}/{keyframe}: no valid frames", file=sys.stderr)
                continue

            row = {
                "method": method_name,
                "dataset": dataset,
                "keyframe": keyframe,
                "frames": len(mae_lst),
                "MAE_mm": round(float(np.mean(mae_lst)), 3),
            }
            if bad_threshold is not None and bad_lst:
                row[f"bad{bad_threshold:g}_%"] = round(float(np.mean(bad_lst)), 3)
            rows.append(row)
            bad_str = ""
            if bad_threshold is not None and bad_lst:
                bad_str = f" bad{bad_threshold:g}%={row[f'bad{bad_threshold:g}_%']:.3f}"
            print(
                f"{method_name:18s} {dataset} {keyframe}: "
                f"MAE={row['MAE_mm']:.3f}{bad_str} n={row['frames']}"
            )

    if rows:
        avg_row = {
            "method": method_name,
            "dataset": "AVG",
            "keyframe": "AVG",
            "frames": sum(r["frames"] for r in rows),
            "MAE_mm": round(float(np.mean([r["MAE_mm"] for r in rows])), 3),
        }
        bad_col = f"bad{bad_threshold:g}_%" if bad_threshold is not None else None
        if bad_col and all(bad_col in r for r in rows):
            avg_row[bad_col] = round(float(np.mean([r[bad_col] for r in rows])), 3)
        rows.append(avg_row)

    return rows


def write_csv(rows: List[dict], output: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"written: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rectified depth MAE from disparity predictions")
    parser.add_argument("gt_root", type=Path, help="SCARED_DATASET_processed root")
    parser.add_argument("pred_root", type=Path, help="Method run with dataset_8/9 disparity PNGs")
    parser.add_argument("--method_name", default=None, help="Label in output CSV (default: pred folder name)")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--scale_factor", type=float, default=128.0)
    parser.add_argument("--bad_threshold", type=float, default=5.0)
    parser.add_argument("--datasets", nargs="+", default=["dataset_8", "dataset_9"])
    args = parser.parse_args()

    method_name = args.method_name or args.pred_root.name
    output = args.output or (args.pred_root / "results_rectified_depth.csv")

    rows = eval_run(
        args.gt_root,
        args.pred_root,
        method_name,
        args.scale_factor,
        args.bad_threshold,
        args.datasets,
    )
    write_csv(rows, output)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
