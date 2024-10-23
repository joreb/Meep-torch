# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep (center-origin) -> meep-torch (corner-origin) coordinate converter.

meep centers the cell at the origin: the cell spans [-sx/2, sx/2) x [-sy/2, sy/2)
with grid cells of side dx = 1/resolution. meep-torch grids are corner-origin:
cell (i, j) has its center at ((i+0.5)*dx, (j+0.5)*dx) in [0, nx*dx) x [0, ny*dx).

This is the single place that maps meep world coordinates to grid indices and to
corner-origin Polygon vertex coordinates. Geometry objects, sources, monitors
and PML bands all route through it, so it is tested hard (test_w1_coord_converter).

Infinite (mp.inf) extents on a Block are clipped to the full cell here.
"""
import math
from dataclasses import dataclass

from .vec import Vector3

INF = float("inf")


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int
    dx: float
    sx: float            # cell size um (x)
    sy: float            # cell size um (y)


def grid_spec(cell: Vector3, resolution: float) -> GridSpec:
    """cell_size + resolution -> GridSpec (dx=1/R, nx=round(sx*R))."""
    sx, sy = float(cell.x), float(cell.y)
    dx = 1.0 / float(resolution)
    nx = max(3, int(round(sx / dx))) if sx > 0 else 3
    ny = max(3, int(round(sy / dx))) if sy > 0 else 3
    return GridSpec(nx, ny, dx, sx if sx > 0 else nx * dx, sy if sy > 0 else ny * dx)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_corner(vec: Vector3, gs: GridSpec) -> tuple[float, float]:
    """World (center-origin) -> corner-origin um, clipping mp.inf to the cell."""
    x = gs.sx if math.isinf(vec.x) else vec.x + gs.sx / 2.0
    y = gs.sy if math.isinf(vec.y) else vec.y + gs.sy / 2.0
    x = _clip(x, 0.0, gs.sx)
    y = _clip(y, 0.0, gs.sy)
    return x, y


def rect_corners(center: Vector3, size: Vector3, gs: GridSpec) -> tuple[float, float, float, float]:
    """Center+size (world) -> corner-origin (x0, y0, x1, y1), clipping inf to cell."""
    if math.isinf(size.x):
        x0, x1 = 0.0, gs.sx
    else:
        x0 = _clip(center.x - size.x / 2.0 + gs.sx / 2.0, 0.0, gs.sx)
        x1 = _clip(center.x + size.x / 2.0 + gs.sx / 2.0, 0.0, gs.sx)
    if math.isinf(size.y):
        y0, y1 = 0.0, gs.sy
    else:
        y0 = _clip(center.y - size.y / 2.0 + gs.sy / 2.0, 0.0, gs.sy)
        y1 = _clip(center.y + size.y / 2.0 + gs.sy / 2.0, 0.0, gs.sy)
    return x0, y0, x1, y1


def cell_index(x_world: float, gs: GridSpec) -> int:
    """World x (center-origin) -> grid cell index (cell-edge aligned, matching the
    raw-port examples which place a source at corner_x via round(corner_x/dx)).
    Cell i covers corner coords [i*dx, (i+1)*dx).
    """
    return int(round((x_world + gs.sx / 2.0) / gs.dx))


def cell_index_y(y_world: float, gs: GridSpec) -> int:
    """World y (center-origin) -> grid cell index along y (cell-edge aligned)."""
    return int(round((y_world + gs.sy / 2.0) / gs.dx))


def span_index_range(x0_um: float, x1_um: float, gs: GridSpec, axis: str) -> tuple[int, int]:
    """Corner-origin um span [x0,x1] -> integer cell index range [i0, i1) (lo,hi)."""
    n = gs.nx if axis == "x" else gs.ny
    L = gs.sx if axis == "x" else gs.sy
    i0 = max(0, int(round(x0_um / gs.dx)))
    i1 = min(n, int(round(x1_um / gs.dx)))
    if i1 <= i0:
        i1 = min(n, i0 + 1)
    return i0, i1


def pml_npml(thickness: float, gs: GridSpec) -> int:
    return max(0, int(round(thickness / gs.dx)))