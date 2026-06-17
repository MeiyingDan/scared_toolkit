"""Step-2 learned stereo-depth uncertainty package.

This package implements a second-stage network that learns to predict the
per-pixel / per-point depth error (and a calibrated uncertainty) of a frozen
stereo model, using ONLY signals that are available at deployment time (no
ground truth).  It is trained against the Step-1 error maps computed on SCARED.

Modules
-------
- ``config``        : dataset roots and shared constants.
- ``scared_io``     : minimal, self-contained SCARED I/O (no scaredtk dependency).
- ``ply_io``        : minimal binary PLY reader/writer (no plyfile dependency).
- ``export_error``  : (todo 1) dump raw per-pixel error/depth/disparity labels.
- ``dataset_index`` : (todo 2) build a frame index and train/val/test split.
- ``features``      : (todo 3) no-GT feature builder.
- ``datasets``      : torch datasets (2D pixel-wise and 3D point-wise).
- ``models``        : (todo 4/5) 2D U-Net and 3D PointNet uncertainty heads.
- ``calibration``   : (todo 6) variance scaling + clinical error bands.
- ``metrics``       : (todo 7) MAE/RMSE/Spearman/AUSE/ECE/AUROC.
- ``train``         : training loop for both variants.
- ``evaluate``      : evaluation + 2D-vs-3D + cross-domain comparison.
- ``clinical_demo`` : (todo 8) grasp-safety map + best-view selection.
"""

__version__ = "0.1.0"
