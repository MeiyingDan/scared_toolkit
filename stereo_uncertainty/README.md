# Step-2: Learned Stereo-Depth Uncertainty

A second-stage network that learns to predict the **per-pixel / per-point depth
error** (and a calibrated **uncertainty**) of a frozen stereo model, using **only
signals available at deployment time** (no ground truth). It is trained against
the Step-1 error maps computed on SCARED, and powers a clinical **grasp-safety**
map and **best-view / re-angle selection** demo.

## Why (the core reframing)

Step 1 (already done) ran several frozen stereo models (IGEV, FoundationStereo,
CREStereo, HRS, FFS, SGM) on SCARED, turned disparity into point clouds, and
coloured each point by `|Z_pred - Z_gt|` (the TURBO `*_err.ply`). That error is
only computable **because SCARED has ground truth**.

Inside a pig/human there is **no ground truth**, so the error itself can never be
an input at deployment. The Step-1 error map is therefore the **training label**,
and Step 2 learns a function:

```
inputs available without GT  ->  predicted depth error  ->  calibrated uncertainty
(RGB, disparity, depth,           (mm, per pixel/point)      (mm / safe-unsafe)
 LR-consistency, gradients, glare)
```

Clinically: "Can I grasp this tissue safely here?" and "if I move/re-angle a
little, does the error in my target region drop?"

## Pipeline

```
export_error  ->  dataset_index  ->  features  ->  train (2D & 3D)  ->  evaluate  ->  clinical_demo
   (labels)         (split)         (no-GT in)     (NLL)              (metrics)      (grasp + view)
```

1. **export_error** (todo 1) - dumps raw `err_mm`, `valid`, `depth`, `disp` per
   frame as compressed `.npz` (Step-1 only saved coloured PLYs).
2. **dataset_index** (todo 2) - joins labels + rectified pair + prediction +
   calibration, and splits at the `(dataset, keyframe)` group level (no frame
   leakage).
3. **features** (todo 3) - no-GT feature builder: RGB, disparity, depth,
   left-right photometric residual, disparity gradient, specular-glare mask.
4. **models/unet2d** (todo 4) + **models/pointnet3d** (todo 5) - 2D pixel-wise and
   3D point-wise heads, both predicting `(mu, log_var)` trained with a
   heteroscedastic Gaussian NLL.
5. **calibration** (todo 6) - post-hoc variance scaling + clinical bands
   (green <=2 mm, yellow <=10 mm, red).
6. **metrics / evaluate** (todo 7) - MAE/RMSE, Spearman, AUSE (sparsification),
   ECE (calibration), grasp AUROC/AUPRC, 2D-vs-3D + cross-domain comparison.
7. **clinical_demo** (todo 8) - grasp-safety overlay + uncertainty point cloud,
   and best-view selection validated against true ROI error (regret vs oracle).

## Quick start

```bash
# tiny CPU smoke run (a few frames, 2 iters):
SMOKE=1 bash scripts/run_pipeline.sh

# full run (all d8/d9 frames; uses GPU if available):
bash scripts/run_pipeline.sh
```

Or step by step (use the openstereo env interpreter):

```bash
PY=/home/meiying/miniconda3/envs/openstereo/bin/python
$PY -m stereo_uncertainty.export_error --datasets dataset_8 dataset_9
$PY -m stereo_uncertainty.dataset_index --val_groups 2 --test_groups 2
$PY -m stereo_uncertainty.train --variant 2d --method IGEV --epochs 30
$PY -m stereo_uncertainty.train --variant 3d --method IGEV --epochs 30
$PY -m stereo_uncertainty.evaluate --variant 2d --method IGEV
$PY -m stereo_uncertainty.evaluate --variant 3d --method IGEV
$PY -m stereo_uncertainty.evaluate --aggregate
$PY -m stereo_uncertainty.clinical_demo --variant 2d --method IGEV \
     --dataset dataset_9 --keyframe keyframe_2 --roi 400 300 800 700
```

## Data layout assumptions

- GT + rectified stereo: `SCARED_DATASET_processed/<dataset>/<keyframe>/data/{left_rectified,right_rectified,disparity}/<frame>.png` and `stereo_calib.json` (provides `Q`).
- Frozen-model predictions: `<...>/methods_results/<METHOD>_results/<run>/<dataset>/<keyframe>/data/disparity/<frame>.png`.
- Disparity PNGs are 16-bit, scaled by `128.0` (configurable: `SU_DISP_SCALE`).

All roots are overridable via env vars (`SU_GT_ROOT`, `SU_RESULTS_ROOT`,
`SU_OUT_ROOT`, `SU_D4D_ROOT`) - see `stereo_uncertainty/config.py`.

## Notes / deviations from the plan

- **Only `dataset_8`/`dataset_9` have frozen-model predictions** in this
  workspace, so labels (which need prediction AND GT) exist only there. The split
  is therefore over their 10 keyframes (`val_groups`/`test_groups` at the
  keyframe level) rather than holding out whole datasets. The indexer is generic:
  add more datasets/methods and they are picked up automatically.
- **Cross-domain (D4D)** is supported by simply pointing `--index` at a D4D-built
  index once D4D predictions + GT are exported; no code change is required. D4D is
  intentionally never placed in train/val.
- **Per-method models** are the default (`--method`); the split is shared across
  methods so the 2D-vs-3D and per-method comparisons use identical scenes.
- The package is self-contained: it does **not** import `scaredtk` (so it runs in
  the `openstereo` env) and ships a minimal PLY reader/writer (no `plyfile`).

## Environment

Use the `openstereo` conda env (torch 2.6). See `requirements.txt`; everything is
already present there except `plyfile`, which this package does not need.

## Tests

```bash
/home/meiying/miniconda3/envs/openstereo/bin/python tests/test_smoke.py
```
