"""Minimal binary PLY reader/writer (xyz + uint8 rgb).

Avoids a hard dependency on ``plyfile`` (absent in the ``openstereo`` env).
Only the small subset needed by this project is implemented:

- read vertices (x, y, z) and optional (red, green, blue);
- write an Nx3 float cloud with an Nx3 uint8 colour array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def write_ply(path, xyz: np.ndarray, rgb: np.ndarray, binary: bool = True) -> Path:
    """Write an Nx3 float32 cloud with Nx3 uint8 colours to ``path``."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    assert rgb.shape == xyz.shape, "xyz and rgb must have matching shape"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = xyz.shape[0]

    if binary:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        dtype = np.dtype(
            [
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ]
        )
        verts = np.empty(n, dtype=dtype)
        verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        with open(path, "wb") as f:
            f.write(header)
            f.write(verts.tobytes())
    else:
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue",
            "end_header",
        ]
        for i in range(n):
            lines.append(
                f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} "
                f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}"
            )
        path.write_text("\n".join(lines) + "\n")
    return path


def read_ply(path) -> Tuple[np.ndarray, np.ndarray | None]:
    """Read a (binary or ascii) PLY and return ``(xyz, rgb_or_None)``.

    Supports the simple ``vertex`` element layout written by ``write_ply`` and by
    the Step-1 export script (float x/y/z + uchar red/green/blue).
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.find(b"end_header\n")
    if end < 0:
        raise ValueError(f"no end_header in {path}")
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    body = raw[end + len(b"end_header\n"):]

    fmt = "ascii"
    n = 0
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            props.append((parts[1], parts[2]))  # (type, name)

    type_map = {
        "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
        "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
        "float32": "f4", "float64": "f8", "uint8": "u1",
    }

    if fmt.startswith("binary"):
        order = "<" if "little" in fmt else ">"
        dtype = np.dtype([(name, order + type_map[t]) for t, name in props])
        verts = np.frombuffer(body, dtype=dtype, count=n)
        xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
        rgb = None
        if {"red", "green", "blue"}.issubset(verts.dtype.names):
            rgb = np.stack(
                [verts["red"], verts["green"], verts["blue"]], axis=1
            ).astype(np.uint8)
        return xyz, rgb

    # ascii
    names = [name for _, name in props]
    arr = np.fromstring(body.decode("ascii"), sep=" ").reshape(n, len(names))
    cols = {name: arr[:, i] for i, name in enumerate(names)}
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float32)
    rgb = None
    if {"red", "green", "blue"}.issubset(cols):
        rgb = np.stack(
            [cols["red"], cols["green"], cols["blue"]], axis=1
        ).astype(np.uint8)
    return xyz, rgb
