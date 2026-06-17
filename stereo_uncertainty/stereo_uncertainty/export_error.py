"""Todo 1 - export raw per-pixel error labels for the Step-2 network.

For every (method, dataset, keyframe, frame) for which both a frozen-model
disparity prediction and SCARED ground-truth disparity exist, this script
computes and stores, aligned to the rectified left image:

- ``err_mm``  : |Z_pred - Z_gt| in millimetres  (the TRAINING TARGET)
- ``depth``   : predicted Z in millimetres        (kept for convenience)
- ``disp``    : predicted disparity               (kept for convenience)
- ``valid``   : boolean mask where both pred and GT are finite

The Step-1 ``export_pointcloud_examples.py`` only saved a TURBO-coloured PLY;
training needs the raw float error, which this module provides as compressed
``.npz`` files (float16 for the float fields, bool for the mask).

Run (from the ``stereo_uncertainty`` repo root, in the openstereo env)::

    python -m stereo_uncertainty.export_error --datasets dataset_8 dataset_9
    python -m stereo_uncertainty.export_error --methods IGEV --limit 5   # smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import config
from . import scared_io as sio


def _list_frames(disp_dir: Path) -> List[str]:
    if not disp_dir.is_dir():
        return []
    return sorted(p.name for p in disp_dir.iterdir() if p.suffix == ".png")


def export_frame(
    pred_disp_path: Path,
    gt_disp_path: Path,
    Q: np.ndarray,
    out_path: Path,
    scale_factor: float,
    overwrite: bool = False,
) -> str:
    """Export a single frame. Returns a status string ('ok'/'skip'/'missing'/...)."""
    if out_path.is_file() and not overwrite:
        return "skip"
    if not pred_disp_path.is_file():
        return "missing-pred"
    if not gt_disp_path.is_file():
        return "missing-gt"

    pred_disp = sio.load_subpix_png(pred_disp_path, scale_factor=scale_factor)
    gt_disp = sio.load_subpix_png(gt_disp_path, scale_factor=scale_factor)
    if pred_disp.shape != gt_disp.shape:
        return "shape-mismatch"

    pred_img3d = sio.disparity_to_img3d(pred_disp, Q)
    gt_img3d = sio.disparity_to_img3d(gt_disp, Q)
    pred_z = pred_img3d[:, :, 2]
    gt_z = gt_img3d[:, :, 2]

    err = np.abs(pred_z - gt_z)
    valid = np.isfinite(err)
    if not np.any(valid):
        return "no-overlap"

    err_out = np.where(valid, err, 0.0).astype(np.float16)
    depth_out = np.nan_to_num(pred_z, nan=0.0).astype(np.float16)
    disp_out = np.nan_to_num(pred_disp, nan=0.0).astype(np.float16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        err_mm=err_out,
        depth=depth_out,
        disp=disp_out,
        valid=valid,
    )
    return "ok"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export raw per-pixel error labels")
    parser.add_argument("--gt_root", type=Path, default=config.GT_ROOT)
    parser.add_argument("--results_root", type=Path, default=config.RESULTS_ROOT)
    parser.add_argument("--out_dir", type=Path, default=config.ERROR_MAP_DIR)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Subset of methods (default: all discovered).")
    parser.add_argument("--datasets", nargs="+",
                        default=["dataset_8", "dataset_9"])
    parser.add_argument("--keyframes", nargs="+", default=None,
                        help="Default: all keyframe_* present.")
    parser.add_argument("--scale_factor", type=float,
                        default=config.DISP_SCALE_FACTOR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max frames per keyframe (smoke testing).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every N-th frame.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    run_map = config.discover_runs(args.results_root, methods=args.methods)
    if not run_map:
        print(f"No method runs found under {args.results_root}", file=sys.stderr)
        return 1
    print("Discovered runs:")
    for m, r in sorted(run_map.items()):
        print(f"  - {m}: {r}")

    counts: dict[str, int] = {}
    for method, run_dir in sorted(run_map.items()):
        for dataset in args.datasets:
            ds_dir = args.gt_root / dataset
            if not ds_dir.is_dir():
                continue
            keyframes = args.keyframes or sorted(
                p.name for p in ds_dir.iterdir() if p.name.startswith("keyframe_")
            )
            for keyframe in keyframes:
                kf_gt = args.gt_root / dataset / keyframe
                calib_path = kf_gt / "stereo_calib.json"
                if not calib_path.is_file():
                    continue
                Q = sio.load_stereo_calib(calib_path).get("Q")
                if Q is None:
                    continue
                gt_disp_dir = kf_gt / "data" / "disparity"
                pred_disp_dir = run_dir / dataset / keyframe / "data" / "disparity"
                frames = _list_frames(pred_disp_dir)
                if args.stride > 1:
                    frames = frames[:: args.stride]
                if args.limit is not None:
                    frames = frames[: args.limit]
                for fname in frames:
                    out_path = (
                        args.out_dir / method / dataset / keyframe
                        / fname.replace(".png", ".npz")
                    )
                    status = export_frame(
                        pred_disp_dir / fname,
                        gt_disp_dir / fname,
                        Q,
                        out_path,
                        scale_factor=args.scale_factor,
                        overwrite=args.overwrite,
                    )
                    counts[status] = counts.get(status, 0) + 1
                print(f"  [{method}/{dataset}/{keyframe}] {len(frames)} frames")

    print("Done. Status counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
