#!/usr/bin/env python3
"""Prepare official SCARED evaluation inputs (GT + submissions).

This script does NOT run the official evaluator by itself. It converts:
- GT depth from SCARED_DATASET_processed into official `processed/frameXXXXXX.tiff`
- Predicted disparity PNGs into original-view depth TIFF submissions

Then it prints the exact `main.py` command to run manually.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tifffile

import scaredtk.convertions as cvt
import scaredtk.io as sio


def parse_opencv_json_node(node):
	# Parse OpenCV json matrix nodes into numpy arrays
	if isinstance(node, dict) and node.get("type_id") == "opencv-matrix":
		rows = int(node["rows"])
		cols = int(node["cols"])
		data = np.asarray(node["data"], dtype=np.float64)
		return data.reshape(rows, cols)
	return node


def load_stereo_calib_json(path: Path) -> Dict[str, np.ndarray]:
	# Load stereo calibration json and convert OpenCV matrix fields
	with open(path, "r", encoding="utf-8") as f:
		raw = json.load(f)
	return {k: parse_opencv_json_node(v) for k, v in raw.items()}


def compute_distort_maps(src_k, dst_k, dst_d, h, w):
	# Compute inverse distortion maps for remapping rectified depth to original view
	xvalues = np.arange(w)
	yvalues = np.arange(h)
	xx, yy = np.meshgrid(xvalues, yvalues)
	xx = xx.reshape(-1, 1)
	yy = yy.reshape(-1, 1)
	maps = np.squeeze(cv2.undistortPoints(np.hstack((xx, yy)).astype(np.float32), dst_k, dst_d))
	maps = maps.reshape(h, w, 2)
	map_x = ((src_k[0, 0] * maps[..., 0]) + src_k[0, 2]).astype(np.float32)
	map_y = ((src_k[1, 1] * maps[..., 1]) + src_k[1, 2]).astype(np.float32)
	return np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)


def naive_interpolation(img):
	# Simple NaN interpolation to fill holes after reprojection/remap
	# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
	h, w = img.shape[:2]
	flat = img.reshape(-1)
	ok = ~np.isnan(flat)
	xp = ok.ravel().nonzero()[0]
	fp = flat[~np.isnan(flat)]
	x = np.isnan(flat).ravel().nonzero()[0]
	if len(xp) == 0:
		return np.zeros_like(img, dtype=np.float32)
	flat[np.isnan(flat)] = np.interp(x, xp, fp)
	return flat.reshape(h, w)


def discover_latest_runs(results_root: Path, methods: Optional[List[str]] = None) -> Dict[str, Path]:
	# Discover latest run folder for each method under results_root
	excluded = {"plots", "tables_like_scared", "pointcloud_examples", "official_eval_d8d9"}
	runs = {}
	for candidate in sorted([p for p in results_root.iterdir() if p.is_dir()]):
		if candidate.name in excluded:
			continue
		if candidate.name.startswith("dataset_"):
			continue
		run_dirs = sorted([p for p in candidate.iterdir() if p.is_dir()])
		if not run_dirs:
			continue
		latest = run_dirs[-1]
		if not any((latest / d).exists() for d in ["dataset_8", "dataset_9"]):
			continue

		method_name = candidate.name
		if method_name.endswith("_results"):
			method_name = method_name[: -len("_results")]
		# Normalise some names that are awkward as submission folder names
		method_name = method_name.replace("HRS_results_testres0.5", "HRS_t05")
		method_name = method_name.replace("HRS_results_testres1", "HRS_t10")
		if methods and method_name not in methods:
			continue
		runs[method_name] = latest
	return runs


def load_frame_ids(valid_csv: Path, fallback_len: int) -> List[int]:
	# Read valid frame indices; fallback to full range if valid.csv is missing
	if valid_csv.is_file():
		valid_arr = np.loadtxt(valid_csv, delimiter=",")
		if np.ndim(valid_arr) == 0:
			return [int(valid_arr)]
		return [int(v) for v in valid_arr.tolist()]
	return list(range(fallback_len))


def convert_disparity_to_original_depth(disparity: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
	# Convert rectified disparity into original-camera-view depth
	pt_cloud = cvt.disparity_to_ptcloud(disparity, calib["Q"])
	# rotate ptcloud to the original frame of reference
	pt_cloud = cvt.transform_pts(pt_cloud, cvt.create_RT(R=np.linalg.inv(calib["R1"])))
	# project pointcloud back to image and take depth channel
	img_3d = cvt.ptcloud_to_img3d(pt_cloud, calib["P1"][:3, :3], np.zeros_like(calib["D1"]), disparity.shape[:2])
	depth_rect = img_3d[..., -1].astype(np.float32)

	# adjust projection for initial camera matrix and distortions
	map_x, map_y = compute_distort_maps(calib["P1"][:3, :3], calib["K1"], calib["D1"], disparity.shape[0], disparity.shape[1])
	depth = cv2.remap(depth_rect, map_x, map_y, cv2.INTER_NEAREST)

	# Pixels that were not hit by any projected point come out as 0 (not NaN).
	# Convert zero holes to NaN so naive_interpolation can fill them; otherwise
	# the official evaluator treats the zeros as real predictions and drives
	# the error up by the GT magnitude at those pixels (causing ~10x inflated MAE).
	depth[~np.isfinite(depth)] = np.nan
	depth[depth <= 0] = np.nan
	depth = naive_interpolation(depth).astype(np.float32)
	return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def save_tiff(path: Path, img: np.ndarray):
	# Save float depth as .tiff for official SCARED format
	path.parent.mkdir(parents=True, exist_ok=True)
	img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
	tifffile.imwrite(str(path), img)


def main() -> int:
	# Build official eval input folder structure only (no auto evaluation)
	parser = argparse.ArgumentParser(
		description="Prepare official SCARED evaluation inputs only (GT + submissions, no auto evaluation)."
	)
	parser.add_argument("results_root", type=Path, help="Path to results_d8d9")
	parser.add_argument("gt_processed_root", type=Path, help="Path to SCARED_DATASET_processed")
	parser.add_argument("official_code_dir", type=Path, help="Path to SCARED_DATASET/code")
	parser.add_argument("work_dir", type=Path, help="Working directory for converted GT/submissions")
	parser.add_argument("--datasets", nargs="+", default=["dataset_8", "dataset_9"])
	parser.add_argument(
		"--keyframes",
		nargs="+",
		default=["keyframe_0", "keyframe_1", "keyframe_2", "keyframe_3", "keyframe_4"],
	)
	parser.add_argument("--methods", nargs="+", default=None, help="e.g. FFS FoundationStereo IGEV CREStereo")
	parser.add_argument("--pred_scale_factor", type=float, default=128.0)
	parser.add_argument("--gt_scale_factor", type=float, default=128.0)
	args = parser.parse_args()

	runs = discover_latest_runs(args.results_root, args.methods)
	if not runs:
		print("No valid method runs found.")
		return 1

	gt_official_dir = args.work_dir / "ground_truth_official"
	submissions_dir = args.work_dir / "submissions_official"
	output_dir = args.work_dir / "official_output"

	print("Discovered method runs:")
	for method_name, run_dir in sorted(runs.items()):
		print(f"  - {method_name}: {run_dir}")

	prepared_gt = 0
	prepared_pred = 0

	# For each dataset/keyframe: build official GT and submission files
	for dataset in args.datasets:
		for keyframe in args.keyframes:
			gt_depth_dir = args.gt_processed_root / dataset / keyframe / "data" / "depthmap"
			gt_valid_csv = args.gt_processed_root / dataset / keyframe / "valid.csv"
			calib_path = args.gt_processed_root / dataset / keyframe / "stereo_calib.json"

			if not gt_depth_dir.is_dir() or not calib_path.is_file():
				print(f"[warn] missing GT/calibration for {dataset}/{keyframe}, skip")
				continue

			calib = load_stereo_calib_json(calib_path)
			gt_files = sorted([p for p in gt_depth_dir.iterdir() if p.suffix.lower() in {".png", ".tiff", ".tif"}])
			if not gt_files:
				print(f"[warn] no GT depth files in {gt_depth_dir}")
				continue
			valid_ids = load_frame_ids(gt_valid_csv, fallback_len=len(gt_files))

			# Convert GT depth to official naming: processed/frameXXXXXX.tiff
			for frame_id in valid_ids:
				if frame_id < 0 or frame_id >= len(gt_files):
					continue
				src = gt_files[frame_id]
				if src.suffix.lower() == ".png":
					depth = sio.load_subpix_png(src, scale_factor=args.gt_scale_factor)
				else:
					depth = tifffile.imread(str(src)).astype(np.float32)
				depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
				gt_dst = gt_official_dir / dataset / keyframe / "processed" / f"frame{frame_id:06d}.tiff"
				save_tiff(gt_dst, depth)
				prepared_gt += 1

			# Convert method disparities to original-view depth and save as official submission tiff
			for method_name, run_dir in sorted(runs.items()):
				pred_disp_dir = run_dir / dataset / keyframe / "data" / "disparity"
				if not pred_disp_dir.is_dir():
					print(f"[warn] missing prediction dir for {method_name}: {pred_disp_dir}")
					continue

				for frame_id in valid_ids:
					disp_path = pred_disp_dir / f"{frame_id:06d}.png"
					if not disp_path.is_file():
						continue
					disparity = sio.load_subpix_png(disp_path, scale_factor=args.pred_scale_factor).astype(np.float32)
					depth_original = convert_disparity_to_original_depth(disparity, calib)
					pred_dst = submissions_dir / method_name / dataset / keyframe / f"frame{frame_id:06d}.tiff"
					save_tiff(pred_dst, depth_original)
					prepared_pred += 1

	output_dir.mkdir(parents=True, exist_ok=True)

	# Print the official evaluation command for manual execution
	cmd = [
		sys.executable,
		str(args.official_code_dir / "main.py"),
		"--ground-truth-dir",
		str(gt_official_dir),
		"--submissions-dir",
		str(submissions_dir),
		"--output-dir",
		str(output_dir),
	]

	print("\nPreparation finished.")
	print(f"Prepared GT frames: {prepared_gt}")
	print(f"Prepared prediction frames: {prepared_pred}")
	print(f"Prepared GT dir: {gt_official_dir}")
	print(f"Prepared submissions dir: {submissions_dir}")
	print("\nRun official evaluation manually with:")
	print(" ".join(cmd))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

