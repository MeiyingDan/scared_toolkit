#!/usr/bin/env bash
# End-to-end pipeline for the Step-2 learned stereo uncertainty system.
#
# Usage:
#   bash scripts/run_pipeline.sh            # full run (all frames, GPU if avail.)
#   SMOKE=1 bash scripts/run_pipeline.sh    # tiny CPU smoke run
#
# Override the interpreter with PY=... (default: openstereo conda env).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/meiying/miniconda3/envs/openstereo/bin/python}
METHOD=${METHOD:-IGEV}

if [[ "${SMOKE:-0}" == "1" ]]; then
  EXPORT_ARGS="--methods ${METHOD} --datasets dataset_8 dataset_9 --limit 4"
  TRAIN2D="--variant 2d --method ${METHOD} --epochs 1 --batch_size 1 --out_size 96 120 --max_iters 2 --device cpu"
  TRAIN3D="--variant 3d --method ${METHOD} --epochs 1 --batch_size 2 --n_points 1024 --max_iters 2 --device cpu"
  EVAL_DEV="--device cpu"
else
  EXPORT_ARGS="--datasets dataset_8 dataset_9"
  TRAIN2D="--variant 2d --method ${METHOD} --epochs 30 --batch_size 4 --out_size 256 320"
  TRAIN3D="--variant 3d --method ${METHOD} --epochs 30 --batch_size 6 --n_points 8192"
  EVAL_DEV=""
fi

echo "== 1. export error labels =="
$PY -m stereo_uncertainty.export_error $EXPORT_ARGS

echo "== 2. build index + split =="
$PY -m stereo_uncertainty.dataset_index --val_groups 2 --test_groups 2

echo "== 3. train 2D =="
$PY -m stereo_uncertainty.train $TRAIN2D
echo "== 3. train 3D =="
$PY -m stereo_uncertainty.train $TRAIN3D

echo "== 4. evaluate + compare =="
$PY -m stereo_uncertainty.evaluate --variant 2d --method ${METHOD} $EVAL_DEV
$PY -m stereo_uncertainty.evaluate --variant 3d --method ${METHOD} $EVAL_DEV
$PY -m stereo_uncertainty.evaluate --aggregate

echo "== 5. clinical demo =="
$PY -m stereo_uncertainty.clinical_demo --variant 2d --method ${METHOD} \
    --dataset dataset_9 --keyframe keyframe_2 $EVAL_DEV || \
$PY -m stereo_uncertainty.clinical_demo --variant 2d --method ${METHOD} \
    --dataset dataset_8 --keyframe keyframe_0 $EVAL_DEV

echo "Done. See outputs/reports/"
