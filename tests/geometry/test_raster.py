# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import numpy as np
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.raster import coverage

G = Grid(nx=40, ny=30, dx=0.1)


def rect(x0, y0, x1, y1):
    return Polygon(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float))


def supersample(poly, grid, S=32):
    """Independent oracle: S x S midpoint sampling per cell."""
    xs = (np.arange(grid.nx * S) + 0.5) * grid.dx / S
    ys = (np.arange(grid.ny * S) + 0.5) * grid.dx / S
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    from matplotlib.path import Path

    inside = Path(poly.vertices).contains_points(
        np.stack([X.ravel(), Y.ravel()], 1)
    ).reshape(grid.nx * S, grid.ny * S)
    return torch.tensor(
        inside.reshape(grid.nx, S, grid.ny, S).mean(axis=(1, 3)), dtype=torch.float64
    )


def test_axis_aligned_rect_exact():
    # rect covering cells [10..20) fully in x, half of cell row j=5 (y from 0.55)
    # NOTE: torch.all(tensor == pytest.approx(x)) fails on PyTorch 2.13 because
    # pytest.approx.__eq__(tensor) returns a Python bool, not a Tensor.
    # torch.allclose preserves the exactness contract while working correctly.
    c = coverage(rect(1.0, 0.55, 2.0, 2.0), G)
    assert torch.allclose(c[10:20, 6:20], torch.ones_like(c[10:20, 6:20]))
    assert torch.allclose(c[10:20, 5], torch.full_like(c[10:20, 5], 0.5))
    assert torch.all(c[:10, :] == 0.0) and torch.all(c[20:, :] == 0.0)


def test_total_area_shoelace_identity():
    tri = Polygon(np.array([[0.31, 0.22], [3.17, 0.95], [1.4, 2.83]], dtype=float))
    c = coverage(tri, G)
    assert c.sum().item() * G.dx**2 == pytest.approx(tri.area, rel=1e-12)


def test_supersample_oracle_triangle():
    tri = Polygon(np.array([[0.31, 0.22], [3.17, 0.95], [1.4, 2.83]], dtype=float))
    c = coverage(tri, G)
    o = supersample(tri, G)
    assert torch.max(torch.abs(c - o)).item() < 2.0 / 32  # oracle granularity bound


def test_rotation_invariance_of_area():
    sq = np.array([[1.5, 1.0], [2.5, 1.0], [2.5, 2.0], [1.5, 2.0]], dtype=float)
    ctr = sq.mean(0)
    th = math.radians(31.0)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    rot = Polygon((sq - ctr) @ R.T + ctr)
    c = coverage(rot, G)
    assert c.sum().item() * G.dx**2 == pytest.approx(1.0, rel=1e-10)
    assert torch.all(c >= 0) and torch.all(c <= 1 + 1e-12)


def test_polygon_partially_outside_grid_clipped():
    c = coverage(rect(-1.0, -1.0, 0.55, 0.55), G)
    assert c[0, 0].item() == pytest.approx(1.0)
    assert c[5, 5].item() == pytest.approx(0.25)  # cell [0.5,0.6)x[0.5,0.6), covered to 0.55


def test_supersample_oracle_concave():
    ell = Polygon(np.array(
        [[0.5, 0.5], [3.0, 0.5], [3.0, 1.2], [1.3, 1.2], [1.3, 2.5], [0.5, 2.5]],
        dtype=float))
    c = coverage(ell, G)
    o = supersample(ell, G)
    assert torch.max(torch.abs(c - o)).item() < 2.0 / 32
    assert c.sum().item() * G.dx**2 == pytest.approx(ell.area, rel=1e-12)
