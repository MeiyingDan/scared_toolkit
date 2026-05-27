#!/usr/bin/env python3
"""Save colourized depth-map PNGs for qualitative comparison.

Loads 16-bit encoded depth PNGs (subpix format) from each method run, maps valid
depth (mm) to a fixed OpenCV colormap, and writes BGR PNGs to a separate tree
under ``<results_root>/depth_color_examples/``.

By default exports two fixed slots (requested for thesis figures):
  * ``dataset_8 / keyframe_0 / frame 0``
  * ``dataset_9 / keyframe_3 / frame 100``

For each slot, depth range for the colormap is computed from the union of
valid pixels across **all** discovered methods (same ``vmin``/``vmax`` per slot
so colours are comparable). Optional ``--include_gt`` uses the same range for
GT depth as well.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import scaredtk.io as sio


def discover_latest_runs(results_root: Path, methods: Optional[List[str]] = None) -> Dict[str, Path]:
    excluded = {"plots", "tables_like_scared", "pointcloud_examples", "depth_color_examples", "official_eval_d8d9"}
    run_map: Dict[str, Path] = {}
    method_dirs = sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and "_results" in p.name and p.name not in excluded
    )
    for method_dir in method_dirs:
        method_name = method_dir.name.replace("_results", "")
        if methods and method_name not in methods:
            continue
        runs = sorted(q for q in method_dir.iterdir() if q.is_dir())
        if not runs:
            continue
        latest = runs[-1]
        if not any((latest / ds).exists() for ds in ["dataset_8", "dataset_9"]):
            continue
        run_map[method_name] = latest
    return run_map


def _load_depth_png(path: Path, scale_factor: float) -> np.ndarray:
    arr = sio.load_subpix_png(path, scale_factor=scale_factor).astype(np.float32)
    return arr


def _valid_mask(d: np.ndarray) -> np.ndarray:
    return np.isfinite(d) & (d > 0)


def _percentile_range(samples: List[np.ndarray], p_low: float, p_high: float) -> Tuple[float, float]:
    if not samples:
        return 0.0, 1.0
    stacked = np.concatenate(samples, axis=0)
    if stacked.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(stacked, [p_low, p_high]).astype(np.float64)
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def depth_to_bgr(
    depth: np.ndarray,
    vmin: float,
    vmax: float,
    cmap_id: int,
) -> np.ndarray:
    valid = _valid_mask(depth)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    scaled = np.clip((depth.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)
    u8 = np.zeros(depth.shape, dtype=np.uint8)
    u8[valid] = (scaled[valid] * 255.0 + 0.5).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cmap_id)
    out[valid] = colored[valid]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export colourized depth examples for selected frames")
    parser.add_argument("results_root", type=Path, help="Path to results_d8d9/methods_results")
    parser.add_argument("gt_root", type=Path, help="Path to SCARED_DATASET_processed (for optional GT)")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output root (default: <results_root>/depth_color_examples)",
    )
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of method names")
    parser.add_argument("--scale_factor", type=float, default=128.0)
    parser.add_argument(
        "--slots",
        nargs="+",
        default=["dataset_8/keyframe_0/0", "dataset_9/keyframe_3/100"],
        help='Each slot as "dataset_X/keyframe_Y/frame_id"',
    )
    parser.add_argument("--p_low", type=float, default=2.0)
    parser.add_argument("--p_high", type=float, default=98.0)
    parser.add_argument(
        "--colormap",
        default="TURBO",
        help="OpenCV colormap name suffix, e.g. TURBO, INFERNO, MAGMA, VIRIDIS (OpenCV 4+)",
    )
    parser.add_argument("--include_gt", action="store_true", help="Also write GT depth with same depth range")
    args = parser.parse_args()

    cmap_name = "COLORMAP_" + args.colormap.upper()
    if not hasattr(cv2, cmap_name):
        print(f"Unknown colormap {cmap_name}. Try TURBO or INFERNO.")
        return 1
    cmap_id = int(getattr(cv2, cmap_name))

    results_root = args.results_root.resolve()
    gt_root = args.gt_root.resolve()
    out_root = (args.out_dir if args.out_dir is not None else results_root / "depth_color_examples").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_map = discover_latest_runs(results_root, methods=args.methods)
    if not run_map:
        print("No method runs found.")
        return 1

    print("Runs:")
    for m, p in sorted(run_map.items()):
        print(f"  {m}: {p}")

    manifest: Dict[str, object] = {"slots": {}, "colormap": cmap_name}

    for slot in args.slots:
        parts = slot.split("/")
        if len(parts) != 3:
            print(f"[skip] bad slot {slot}")
            continue
        ds, kf, frame_s = parts
        frame_id = int(frame_s)
        frame_name = f"{frame_id:06d}.png"
        tag = f"{ds}_{kf}_frame{frame_name.replace('.png','')}"

        depths_for_range: List[np.ndarray] = []
        method_paths: Dict[str, Path] = {}

        for method, run_dir in sorted(run_map.items()):
            p = run_dir / ds / kf / "data" / "depthmap" / frame_name
            if not p.is_file():
                continue
            d = _load_depth_png(p, args.scale_factor)
            method_paths[method] = p
            vals = d[_valid_mask(d)].ravel()
            if vals.size > 0:
                depths_for_range.append(vals)

        gt_path = gt_root / ds / kf / "data" / "depthmap" / frame_name
        if args.include_gt and gt_path.is_file():
            d = _load_depth_png(gt_path, args.scale_factor)
            vals = d[_valid_mask(d)].ravel()
            if vals.size > 0:
                depths_for_range.append(vals)

        vmin, vmax = _percentile_range(depths_for_range, args.p_low, args.p_high)
        manifest["slots"][tag] = {
            "vmin_mm": vmin,
            "vmax_mm": vmax,
            "p_low": args.p_low,
            "p_high": args.p_high,
            "methods_present": sorted(method_paths.keys()),
        }

        for method, depth_path in sorted(method_paths.items()):
            depth = _load_depth_png(depth_path, args.scale_factor)
            bgr = depth_to_bgr(depth, vmin, vmax, cmap_id)
            out_p = out_root / method / f"{tag}_depth_color.png"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), bgr)

        if args.include_gt and gt_path.is_file():
            depth = _load_depth_png(gt_path, args.scale_factor)
            bgr = depth_to_bgr(depth, vmin, vmax, cmap_id)
            out_p = out_root / "GT" / f"{tag}_depth_color.png"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), bgr)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote colourized depth PNGs under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
