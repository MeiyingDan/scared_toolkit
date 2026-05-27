#!/usr/bin/env python3
"""Export qualitative point-cloud examples from disparity predictions.

For every selected (method, dataset, keyframe, frame) this script turns the
predicted disparity PNG into a coloured point cloud and stores it as a .ply
file. Two colouring modes are supported:

* ``rgb``   - colour every point with the corresponding pixel of the
  rectified left image. Use this to produce photorealistic qualitative
  examples.
* ``error`` - colour every point by the absolute depth error versus the
  ground-truth disparity (``|Z_pred - Z_gt|``) using a colormap. Use this
  to visualise where each method fails. Frames that miss GT are skipped
  in this mode.

A convenience mode ``both`` writes the two variants side by side.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

import scaredtk.convertions as cvt
import scaredtk.io as sio


VALID_COLOR_MODES = ("rgb", "error", "both")


def _parse_opencv_json_node(node):
    if isinstance(node, dict) and node.get("type_id") == "opencv-matrix":
        rows = int(node["rows"])
        cols = int(node["cols"])
        data = np.array(node["data"], dtype=np.float64)
        return data.reshape(rows, cols)
    return node


def load_stereo_calib_json(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: _parse_opencv_json_node(v) for k, v in raw.items()}


def discover_latest_runs(results_root: Path, methods: Optional[List[str]] = None) -> Dict[str, Path]:
    run_map: Dict[str, Path] = {}
    method_dirs = sorted([
        p for p in results_root.iterdir()
        # Support both "<method>_results" and variants like
        # "<method>_results_testres0.5".
        if p.is_dir() and "_results" in p.name
    ])

    for method_dir in method_dirs:
        method_name = method_dir.name.replace("_results", "")
        if methods and method_name not in methods:
            continue

        runs = sorted([p for p in method_dir.iterdir() if p.is_dir()])
        if not runs:
            continue
        run_map[method_name] = runs[-1]

    return run_map


def save_ptcloud_as_ply_rgb(
    path: Path,
    ptcloud: np.ndarray,
    colors_u8: np.ndarray,
    save_binary: bool = True,
) -> Path:
    """Save an Nx3 point cloud with Nx3 uint8 RGB colours as a .ply file."""
    assert ptcloud.dtype == np.float32
    assert colors_u8.dtype == np.uint8
    assert ptcloud.shape[0] == colors_u8.shape[0]
    assert ptcloud.shape[1] == 3 and colors_u8.shape[1] == 3

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.empty(
        ptcloud.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"] = ptcloud[:, 0]
    vertices["y"] = ptcloud[:, 1]
    vertices["z"] = ptcloud[:, 2]
    vertices["red"] = colors_u8[:, 0]
    vertices["green"] = colors_u8[:, 1]
    vertices["blue"] = colors_u8[:, 2]

    el = PlyElement.describe(vertices, "vertex")
    if save_binary:
        PlyData([el]).write(str(path))
    else:
        PlyData([el], text=True).write(str(path))
    return path


def _depth_from_disparity(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return an HxWx3 image3d. Invalid pixels (NaN disparity, zero/negative
    disparity, or non-finite reprojected depth) are replaced with NaN."""
    disparity = disparity.astype(np.float32, copy=False)
    invalid = ~np.isfinite(disparity) | (disparity <= 0)
    img3d = cvt.disparity_to_img3d(disparity, Q)
    # disparity_to_img3d already masks disparity<0; also drop disparity==0
    # (produces inf depth) and any non-finite reprojection results.
    img3d[invalid] = np.nan
    img3d[~np.isfinite(img3d).all(axis=2)] = np.nan
    return img3d


def _colorize_error(err_mm: np.ndarray, max_err_mm: float) -> np.ndarray:
    """Map a 1D error array (mm) to uint8 RGB using a perceptual colormap."""
    clipped = np.clip(err_mm, 0.0, max_err_mm)
    if max_err_mm > 0:
        normalized = (clipped / max_err_mm * 255.0).astype(np.uint8)
    else:
        normalized = np.zeros_like(clipped, dtype=np.uint8)
    # cv2 needs HxW for applyColorMap; reshape trick.
    colored_bgr = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_TURBO)
    colored_bgr = colored_bgr.reshape(-1, 3)
    return colored_bgr[:, ::-1].copy()  # BGR -> RGB


def _load_left_rgb(left_path: Path) -> Optional[np.ndarray]:
    if not left_path.is_file():
        return None
    bgr = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _stride_mask(length: int, stride: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if stride <= 1:
        mask[:] = True
    else:
        mask[::stride] = True
    return mask


def export_one_frame(
    pred_disp_path: Path,
    calib_q: np.ndarray,
    out_dir_for_frame: Path,
    frame_id: int,
    scale_factor: float,
    stride: int,
    min_depth: Optional[float],
    max_depth: Optional[float],
    save_binary: bool,
    color_mode: str,
    left_rgb_path: Optional[Path],
    gt_disp_path: Optional[Path],
    max_error_mm: float,
) -> Tuple[int, List[str]]:
    """Export the point cloud(s) for a single prediction frame.

    Returns (num_files_written, warnings).
    """
    warnings: List[str] = []
    if not pred_disp_path.is_file():
        return 0, [f"missing prediction: {pred_disp_path}"]

    disparity = sio.load_subpix_png(pred_disp_path, scale_factor=scale_factor)
    img3d = _depth_from_disparity(disparity, calib_q)  # HxWx3, NaN invalid

    h, w = img3d.shape[:2]
    xyz_flat = img3d.reshape(-1, 3)
    valid = ~np.isnan(xyz_flat).any(axis=1)

    if min_depth is not None:
        valid &= xyz_flat[:, 2] >= min_depth
    if max_depth is not None:
        valid &= xyz_flat[:, 2] <= max_depth

    if not np.any(valid):
        return 0, [f"no valid points: {pred_disp_path}"]

    # Apply stride in a colour-aware way (same indices kept for XYZ and RGB).
    stride_mask = _stride_mask(h * w, stride)
    keep = valid & stride_mask
    if not np.any(keep):
        return 0, [f"stride dropped all points: {pred_disp_path}"]

    xyz = xyz_flat[keep].astype(np.float32, copy=False)

    want_rgb = color_mode in ("rgb", "both")
    want_err = color_mode in ("error", "both")

    rgb_flat = None
    if want_rgb:
        left_rgb = _load_left_rgb(left_rgb_path) if left_rgb_path is not None else None
        if left_rgb is None:
            warnings.append(f"missing left image for rgb mode: {left_rgb_path}")
            want_rgb = False
        elif left_rgb.shape[:2] != (h, w):
            warnings.append(
                f"left image size {left_rgb.shape[:2]} != disparity size {(h, w)}: {left_rgb_path}"
            )
            want_rgb = False
        else:
            rgb_flat = left_rgb.reshape(-1, 3)

    err_flat = None
    if want_err:
        if gt_disp_path is None or not gt_disp_path.is_file():
            warnings.append(f"missing GT disparity for error mode: {gt_disp_path}")
            want_err = False
        else:
            gt_disparity = sio.load_subpix_png(gt_disp_path, scale_factor=scale_factor)
            if gt_disparity.shape != disparity.shape:
                warnings.append(
                    f"GT disparity size {gt_disparity.shape} != pred {disparity.shape}"
                )
                want_err = False
            else:
                gt_img3d = _depth_from_disparity(gt_disparity, calib_q)
                gt_z = gt_img3d[:, :, 2].reshape(-1)
                pred_z = img3d[:, :, 2].reshape(-1)
                err = np.abs(pred_z - gt_z)
                # Points without GT get NaN -> will be excluded from error cloud
                err_flat = err

    written = 0
    out_dir_for_frame.mkdir(parents=True, exist_ok=True)

    if want_rgb and rgb_flat is not None:
        colors = rgb_flat[keep].astype(np.uint8, copy=False)
        out_path = out_dir_for_frame / f"{frame_id:06d}_rgb.ply"
        save_ptcloud_as_ply_rgb(out_path, xyz, colors, save_binary=save_binary)
        written += 1

    if want_err and err_flat is not None:
        err_keep = err_flat[keep]
        valid_err = ~np.isnan(err_keep)
        if not np.any(valid_err):
            warnings.append(f"no overlap pred/GT: {pred_disp_path}")
        else:
            xyz_err = xyz[valid_err]
            err_mm = err_keep[valid_err]
            colors = _colorize_error(err_mm, max_error_mm).astype(np.uint8, copy=False)
            out_path = out_dir_for_frame / f"{frame_id:06d}_err.ply"
            save_ptcloud_as_ply_rgb(out_path, xyz_err, colors, save_binary=save_binary)
            written += 1

    return written, warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export qualitative point-cloud examples from disparity predictions"
    )
    parser.add_argument("results_root", type=Path, help="Path to results_d8d9/methods_results folder")
    parser.add_argument("gt_root", type=Path, help="Path to SCARED_DATASET_processed")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dataset_8", "dataset_9"],
        help="Dataset IDs to export",
    )
    parser.add_argument(
        "--keyframes",
        nargs="+",
        default=["keyframe_0", "keyframe_1", "keyframe_2", "keyframe_3", "keyframe_4"],
        help="Keyframe IDs to export",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[0, 200, 500, 800],
        help="Frame indices to export (e.g. 0 200 500 800)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to export (e.g. FFS FoundationStereo CREStereo)",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=128.0,
        help="Scale factor used when saving disparity PNGs",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for PLY files (default: <results_root>/pointcloud_examples)",
    )
    parser.add_argument(
        "--include_gt",
        action="store_true",
        help="Also export GT disparity point clouds",
    )
    parser.add_argument(
        "--color_mode",
        choices=VALID_COLOR_MODES,
        default="rgb",
        help="rgb: colour with left image; error: colour by |Z_pred - Z_gt|; both: write both",
    )
    parser.add_argument(
        "--max_error_mm",
        type=float,
        default=10.0,
        help="Clip value (mm) for the error colormap in 'error'/'both' modes",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Keep every N-th pixel for lighter PLY files",
    )
    parser.add_argument("--min_depth", type=float, default=None, help="Minimum Z depth to keep")
    parser.add_argument("--max_depth", type=float, default=None, help="Maximum Z depth to keep")
    parser.add_argument("--ascii", action="store_true", help="Save PLY in ASCII mode")
    args = parser.parse_args()

    results_root = args.results_root
    gt_root = args.gt_root
    out_dir = args.out_dir if args.out_dir is not None else results_root / "pointcloud_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_map = discover_latest_runs(results_root, methods=args.methods)
    if not run_map:
        print("No method runs found. Check results_root/method selection.")
        return 1

    print("Discovered runs:")
    for method, run_dir in sorted(run_map.items()):
        print(f"  - {method}: {run_dir}")
    print(f"Color mode: {args.color_mode} (max_error_mm={args.max_error_mm})")

    exported = 0
    missing = 0
    save_binary = not args.ascii
    warnings_log: List[str] = []

    for dataset in args.datasets:
        for keyframe in args.keyframes:
            calib_path = gt_root / dataset / keyframe / "stereo_calib.json"
            if not calib_path.is_file():
                print(f"[warn] missing calibration: {calib_path}")
                continue

            calib = load_stereo_calib_json(calib_path)
            q = calib.get("Q", None)
            if q is None:
                print(f"[warn] missing Q matrix in {calib_path}")
                continue

            left_rect_dir = gt_root / dataset / keyframe / "data" / "left_rectified"
            gt_disp_dir = gt_root / dataset / keyframe / "data" / "disparity"

            for frame_id in args.frames:
                frame_name = f"{frame_id:06d}.png"
                left_rgb_path = left_rect_dir / frame_name
                gt_disp_path = gt_disp_dir / frame_name

                for method, run_dir in sorted(run_map.items()):
                    pred_disp = run_dir / dataset / keyframe / "data" / "disparity" / frame_name
                    frame_out_dir = out_dir / method / dataset / keyframe

                    written, warns = export_one_frame(
                        pred_disp_path=pred_disp,
                        calib_q=q,
                        out_dir_for_frame=frame_out_dir,
                        frame_id=frame_id,
                        scale_factor=args.scale_factor,
                        stride=args.stride,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        save_binary=save_binary,
                        color_mode=args.color_mode,
                        left_rgb_path=left_rgb_path,
                        gt_disp_path=gt_disp_path,
                        max_error_mm=args.max_error_mm,
                    )
                    exported += written
                    if written == 0:
                        missing += 1
                    warnings_log.extend(warns)

                if args.include_gt:
                    gt_out_dir = out_dir / "GT" / dataset / keyframe
                    # GT only has RGB colouring; error would be zero everywhere.
                    written, warns = export_one_frame(
                        pred_disp_path=gt_disp_path,
                        calib_q=q,
                        out_dir_for_frame=gt_out_dir,
                        frame_id=frame_id,
                        scale_factor=args.scale_factor,
                        stride=args.stride,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        save_binary=save_binary,
                        color_mode="rgb",
                        left_rgb_path=left_rgb_path,
                        gt_disp_path=None,
                        max_error_mm=args.max_error_mm,
                    )
                    exported += written
                    if written == 0:
                        missing += 1
                    warnings_log.extend(warns)

    print(f"Done. Exported {exported} point-cloud files. Missing/empty: {missing}")
    if warnings_log:
        print(f"Collected {len(warnings_log)} warnings. First 10:")
        for w in warnings_log[:10]:
            print(f"  ! {w}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
