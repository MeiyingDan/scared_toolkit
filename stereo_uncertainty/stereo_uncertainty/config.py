"""Central configuration: dataset roots and shared constants.

All paths can be overridden from the environment so the code stays portable:

- ``SU_GT_ROOT``       -> SCARED_DATASET_processed (rectified stereo + GT disparity)
- ``SU_RESULTS_ROOT``  -> <...>/methods_results (frozen-model disparity predictions)
- ``SU_OUT_ROOT``      -> where this package writes error maps / indices / checkpoints
- ``SU_D4D_ROOT``      -> optional cross-domain dataset root
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset roots (sensible defaults for this workspace; override via env vars). #
# --------------------------------------------------------------------------- #
_MASTER = Path("/home/meiying/Meiying_Masterarbeit")

GT_ROOT = Path(
    os.environ.get("SU_GT_ROOT", _MASTER / "SCARED" / "SCARED_DATASET_processed")
)
RESULTS_ROOT = Path(
    os.environ.get("SU_RESULTS_ROOT", _MASTER / "results_d8d9" / "methods_results")
)
OUT_ROOT = Path(
    os.environ.get(
        "SU_OUT_ROOT",
        Path(__file__).resolve().parents[1] / "outputs",
    )
)
D4D_ROOT = Path(os.environ.get("SU_D4D_ROOT", _MASTER / "D4D_dataset"))

# Derived output sub-directories.
ERROR_MAP_DIR = OUT_ROOT / "error_maps"
INDEX_DIR = OUT_ROOT / "indices"
CKPT_DIR = OUT_ROOT / "checkpoints"
REPORT_DIR = OUT_ROOT / "reports"

# --------------------------------------------------------------------------- #
# Constants.                                                                   #
# --------------------------------------------------------------------------- #
# Scale factor used when the disparity PNGs were written (see scaredtk).
DISP_SCALE_FACTOR = float(os.environ.get("SU_DISP_SCALE", 128.0))

# Clinical error bands in millimetres: <= SAFE green, <= RISKY yellow, else red.
SAFE_MM = float(os.environ.get("SU_SAFE_MM", 2.0))
RISKY_MM = float(os.environ.get("SU_RISKY_MM", 10.0))

# Grasp-safety detection threshold (mm): error above this is "unsafe".
GRASP_TAU_MM = float(os.environ.get("SU_GRASP_TAU_MM", SAFE_MM))

# Cap used for normalising / clipping the error target during training (mm).
MAX_ERROR_MM = float(os.environ.get("SU_MAX_ERROR_MM", 50.0))

# Per-pixel feature channels produced by ``features.build_features``.
FEATURE_NAMES = (
    "r",
    "g",
    "b",
    "disparity",
    "depth",
    "lr_residual",
    "disp_grad",
    "glare",
)
N_FEATURES = len(FEATURE_NAMES)


def discover_runs(results_root: Path | None = None, methods=None) -> dict[str, Path]:
    """Return ``{method_name: latest_run_dir}`` for every ``*_results`` folder.

    A "run" is the most recent timestamped sub-directory inside ``<method>_results``.
    This mirrors ``scripts/export_pointcloud_examples.discover_latest_runs`` but is
    kept here so the package is self-contained.
    """
    results_root = Path(results_root) if results_root is not None else RESULTS_ROOT
    run_map: dict[str, Path] = {}
    if not results_root.is_dir():
        return run_map
    for method_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if "_results" not in method_dir.name:
            continue
        # Normalise e.g. "HRS_results_testres1" -> "HRS".
        method_name = method_dir.name.split("_results")[0]
        if methods and method_name not in methods:
            continue
        runs = sorted(p for p in method_dir.iterdir() if p.is_dir())
        if runs:
            run_map[method_name] = runs[-1]
    return run_map
