# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import numpy as np
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.raster import coverage, normals

G = Grid(nx=40, ny=30, dx=0.1)


def test_vertical_interface_normal_points_x():
    # half-plane x < 1.55 filled: boundary cells at i=15, normal = (+1, 0)
    p = Polygon(np.array([[-1, -1], [1.55, -1], [1.55, 4], [-1, 4]], dtype=float))
    c = coverage(p, G)
    n = normals(c)
    band = n[15, 5:25]
    assert torch.allclose(band[:, 0], torch.ones(20, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(band[:, 1], torch.zeros(20, dtype=torch.float64), atol=1e-6)
    assert torch.all(n[5, :, :] == 0)  # interior cell: no normal


def test_diagonal_interface_normal_45deg():
    p = Polygon(np.array([[0, 0], [3.0, 0], [0, 3.0]], dtype=float))
    c = coverage(p, G)
    n = normals(c)
    mask = (c > 0.05) & (c < 0.95)
    mask[0, :] = False  # exclude grid-edge cells where replication skews gradient
    mask[:, 0] = False
    nx_vals = n[..., 0][mask]
    ny_vals = n[..., 1][mask]
    tgt = 1 / np.sqrt(2)
    assert torch.allclose(nx_vals, torch.full_like(nx_vals, tgt), atol=0.15)
    assert torch.allclose(ny_vals, torch.full_like(ny_vals, tgt), atol=0.15)


def test_unit_length_on_boundary():
    p = Polygon(np.array([[0.31, 0.22], [3.17, 0.95], [1.4, 2.83]], dtype=float))
    c = coverage(p, G)
    n = normals(c)
    mask = (c > 0) & (c < 1)
    lens = torch.linalg.norm(n[mask], dim=-1)
    assert torch.allclose(lens, torch.ones_like(lens), atol=1e-9)
