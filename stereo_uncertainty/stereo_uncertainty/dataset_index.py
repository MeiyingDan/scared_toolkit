"""Todo 2 - build a frame index and a leakage-safe train/val/test split.

The index is the join of three things, one row per (method, dataset, keyframe,
frame):

- the Step-1 error label      (``outputs/error_maps/<method>/.../<frame>.npz``)
- the rectified stereo pair   (``SCARED_DATASET_processed/.../left|right_rectified``)
- the frozen-model disparity  (``<method>_results/<run>/.../disparity/<frame>.png``)
- the keyframe calibration     (``stereo_calib.json`` -> Q)

Splitting is done at the (dataset, keyframe) GROUP level (never per-frame) so
adjacent video frames cannot leak between train and test.  The same group->split
assignment is shared across all methods, which keeps per-method models and the
2D-vs-3D comparison on identical scenes.

NOTE on data availability: in this workspace only ``dataset_8``/``dataset_9``
have frozen-model predictions, so the split is over their 10 keyframes rather
than holding out whole datasets.  ``D4D`` is reserved purely for cross-domain
evaluation and is intentionally never placed in train/val.

Run::

    python -m stereo_uncertainty.dataset_index --val_groups 2 --test_groups 2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from . import config


def _frame_id(npz_name: str) -> str:
    return npz_name.replace(".npz", "")


def build_records(
    error_map_dir: Path,
    gt_root: Path,
    results_root: Path,
) -> List[dict]:
    """Scan exported error maps and resolve all sibling input paths."""
    run_map = config.discover_runs(results_root)
    records: List[dict] = []
    if not error_map_dir.is_dir():
        return records

    for method_dir in sorted(p for p in error_map_dir.iterdir() if p.is_dir()):
        method = method_dir.name
        run_dir = run_map.get(method)
        for ds_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            dataset = ds_dir.name
            for kf_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
                keyframe = kf_dir.name
                kf_gt = gt_root / dataset / keyframe
                calib = kf_gt / "stereo_calib.json"
                left_dir = kf_gt / "data" / "left_rectified"
                right_dir = kf_gt / "data" / "right_rectified"
                pred_dir = (
                    run_dir / dataset / keyframe / "data" / "disparity"
                    if run_dir is not None else None
                )
                for npz in sorted(kf_dir.glob("*.npz")):
                    fid = _frame_id(npz.name)
                    png = f"{fid}.png"
                    rec = {
                        "method": method,
                        "dataset": dataset,
                        "keyframe": keyframe,
                        "frame": fid,
                        "group": f"{dataset}/{keyframe}",
                        "err_npz": str(npz),
                        "left": str(left_dir / png),
                        "right": str(right_dir / png),
                        "pred_disp": str(pred_dir / png) if pred_dir else None,
                        "calib": str(calib),
                    }
                    records.append(rec)
    return records


def split_groups(
    groups: List[str],
    val_groups: int,
    test_groups: int,
    seed: int = 0,
    holdout_test: Optional[List[str]] = None,
    holdout_val: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Assign each group to 'train'/'val'/'test'. Explicit holdouts win."""
    holdout_test = set(holdout_test or [])
    holdout_val = set(holdout_val or [])
    assignment: Dict[str, str] = {}

    remaining = [g for g in groups if g not in holdout_test and g not in holdout_val]
    rng = random.Random(seed)
    rng.shuffle(remaining)

    need_test = max(0, test_groups - len(holdout_test))
    need_val = max(0, val_groups - len(holdout_val))

    for g in holdout_test:
        assignment[g] = "test"
    for g in holdout_val:
        assignment[g] = "val"
    for g in remaining[:need_test]:
        assignment[g] = "test"
    for g in remaining[need_test:need_test + need_val]:
        assignment[g] = "val"
    for g in remaining[need_test + need_val:]:
        assignment[g] = "train"
    return assignment


def build_split(
    error_map_dir: Path = None,
    gt_root: Path = None,
    results_root: Path = None,
    val_groups: int = 2,
    test_groups: int = 2,
    seed: int = 0,
    holdout_test: Optional[List[str]] = None,
    holdout_val: Optional[List[str]] = None,
) -> dict:
    error_map_dir = error_map_dir or config.ERROR_MAP_DIR
    gt_root = gt_root or config.GT_ROOT
    results_root = results_root or config.RESULTS_ROOT

    records = build_records(error_map_dir, gt_root, results_root)
    groups = sorted({r["group"] for r in records})
    assignment = split_groups(
        groups, val_groups, test_groups, seed, holdout_test, holdout_val
    )
    for r in records:
        r["split"] = assignment[r["group"]]

    summary = {"train": 0, "val": 0, "test": 0}
    for r in records:
        summary[r["split"]] += 1

    return {
        "meta": {
            "n_records": len(records),
            "groups": assignment,
            "split_counts": summary,
            "methods": sorted({r["method"] for r in records}),
            "seed": seed,
        },
        "records": records,
    }


def load_index(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_records(index: dict, split: str = None, method: str = None) -> List[dict]:
    out = index["records"]
    if split is not None:
        out = [r for r in out if r["split"] == split]
    if method is not None:
        out = [r for r in out if r["method"] == method]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build frame index + split")
    parser.add_argument("--error_map_dir", type=Path, default=config.ERROR_MAP_DIR)
    parser.add_argument("--gt_root", type=Path, default=config.GT_ROOT)
    parser.add_argument("--results_root", type=Path, default=config.RESULTS_ROOT)
    parser.add_argument("--out", type=Path,
                        default=config.INDEX_DIR / "index.json")
    parser.add_argument("--val_groups", type=int, default=2)
    parser.add_argument("--test_groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout_test", nargs="+", default=None,
                        help="Groups forced into test, e.g. dataset_9/keyframe_3")
    parser.add_argument("--holdout_val", nargs="+", default=None)
    args = parser.parse_args(argv)

    index = build_split(
        error_map_dir=args.error_map_dir,
        gt_root=args.gt_root,
        results_root=args.results_root,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        seed=args.seed,
        holdout_test=args.holdout_test,
        holdout_val=args.holdout_val,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Wrote {index['meta']['n_records']} records to {args.out}")
    print("Group assignment:")
    for g, s in sorted(index["meta"]["groups"].items()):
        print(f"  {g}: {s}")
    print("Split counts (records):", index["meta"]["split_counts"])
    print("Methods:", index["meta"]["methods"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
