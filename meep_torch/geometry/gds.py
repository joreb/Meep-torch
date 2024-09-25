# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""GDSII layer to Polygon adapter (optional `gdstk` extra)."""
from pathlib import Path

import numpy as np

from .polygons import Polygon


def polygons_from_gds(
    path: "str | Path",
    layer: int,
    datatype: int = 0,
    cell: "str | None" = None,
) -> list[Polygon]:
    """Read a GDSII file and return validated Polygons on ``(layer, datatype)``.

    The top cell is used by default; if the library has multiple top-level
    cells and ``cell`` is None, a ValueError listing the candidates is raised.
    References are resolved by flattening the chosen cell (in place on the
    freshly-read library, which is local and discarded), so referenced
    sub-cells contribute their polygons too.
    """
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "gdstk is required for GDS import; install it with "
            "`uv pip install gdstk` (or `uv pip install --extra gds meep-torch` "
            "for the optional extra)."
        ) from exc

    lib = gdstk.read_gds(Path(path))
    if cell is None:
        tops = lib.top_level()
        if len(tops) != 1:
            names = sorted(c.name for c in tops)
            raise ValueError(
                f"expected exactly one top-level cell in {path}, found "
                f"{len(tops)}: {names}; pass cell=<name> to disambiguate"
            )
        target = tops[0]
    else:
        target = next((c for c in lib.cells if c.name == cell), None)
        if target is None:
            names = sorted(c.name for c in lib.cells)
            raise ValueError(
                f"cell {cell!r} not found in {path}; available: {names}"
            )

    target.flatten()
    out: list[Polygon] = []
    for p in target.get_polygons():
        if p.layer == layer and p.datatype == datatype:
            out.append(Polygon(np.asarray(p.points, dtype=np.float64)))
    return out
