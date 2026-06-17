"""End-to-end smoke test (tiny, CPU-only).

Runs the whole pipeline on a handful of frames and asserts that each stage
produces sane artefacts.  Runnable either with pytest or directly::

    /home/meiying/miniconda3/envs/openstereo/bin/python tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereo_uncertainty import config  # noqa: E402
from stereo_uncertainty import export_error, dataset_index, train, evaluate  # noqa: E402
from stereo_uncertainty import ply_io  # noqa: E402


METHOD = "IGEV"


def test_pipeline_smoke():
    # 1. export a few error labels
    rc = export_error.main(
        ["--methods", METHOD, "--datasets", "dataset_8", "dataset_9", "--limit", "3"]
    )
    assert rc == 0
    assert config.ERROR_MAP_DIR.is_dir()

    # 2. build index + split
    idx_path = config.INDEX_DIR / "index.json"
    rc = dataset_index.main(
        ["--out", str(idx_path), "--val_groups", "2", "--test_groups", "2"]
    )
    assert rc == 0
    index = dataset_index.load_index(idx_path)
    assert index["meta"]["n_records"] > 0

    # 3. train both variants (1 epoch, 2 iters, CPU)
    for variant, extra in (
        ("2d", ["--batch_size", "1", "--out_size", "96", "120"]),
        ("3d", ["--batch_size", "2", "--n_points", "1024"]),
    ):
        rc = train.main(
            ["--variant", variant, "--method", METHOD, "--epochs", "1",
             "--max_iters", "2", "--device", "cpu"] + extra
        )
        assert rc == 0
        assert (config.CKPT_DIR / f"{variant}_{METHOD}.pt").is_file()

    # 4. evaluate both variants + aggregate
    for variant in ("2d", "3d"):
        rc = evaluate.main(
            ["--variant", variant, "--method", METHOD, "--device", "cpu"]
        )
        assert rc == 0
        rep = config.REPORT_DIR / f"eval_{variant}_{METHOD}_test.json"
        assert rep.is_file()
    rc = evaluate.main(["--aggregate"])
    assert rc == 0
    assert (config.REPORT_DIR / "comparison.csv").is_file()


def test_ply_roundtrip():
    xyz = np.random.rand(100, 3).astype(np.float32)
    rgb = (np.random.rand(100, 3) * 255).astype(np.uint8)
    p = config.OUT_ROOT / "tmp_roundtrip.ply"
    ply_io.write_ply(p, xyz, rgb)
    xyz2, rgb2 = ply_io.read_ply(p)
    assert xyz2.shape == xyz.shape
    assert np.allclose(xyz2, xyz, atol=1e-5)
    assert np.array_equal(rgb2, rgb)
    p.unlink()


if __name__ == "__main__":
    test_ply_roundtrip()
    print("ply roundtrip OK")
    test_pipeline_smoke()
    print("pipeline smoke OK")
