# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import numpy as np
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.modes.fd2d import solve_2d


def solve_at_dx(dx, smooth=True, shift=0.0):
    """Si strip 500x220nm, deliberately offset by `shift` cells to expose staircasing."""
    W, H = 3.0, 2.0
    g = Grid(nx=int(W / dx), ny=int(H / dx), dx=dx)
    x0 = W / 2 - 0.25 + shift * dx
    y0 = H / 2 - 0.11
    core = Polygon(np.array(
        [[x0, y0], [x0 + 0.5, y0], [x0 + 0.5, y0 + 0.22], [x0, y0 + 0.22]], dtype=float))
    rm = rasterize_scene([(core, 3.4777**2)], g, background=1.444**2)
    eps = rm.eps_te_xx if smooth else torch.where(rm.eps_tm > 6.0, 3.4777**2, 1.444**2)
    return solve_2d(eps, dx=dx, f=1 / 1.55, k=1).neff[0].real.item()


def _observed_order(n1, n2, n3):
    """Richardson observed order from three solutions on halving grids.

    With error ~ C*dx**p and dx2 = dx1/2, dx3 = dx2/2, the ratio of consecutive
    differences |n1-n2| / |n2-n3| equals 2**p, so p = log(ratio)/log(2). This
    observed-order estimate from consecutive differences avoids a separate
    reference solve (a 4x-finer ref grid OOMs the SuperLU shift-invert
    factorization on small-memory boxes) and measures the same convergence
    rate. The finest grid here is 300x200 (dx=0.01); dx=0.005 (600x400) OOMs
    on a ~7 GB box, so the halving triplet starts one level coarser than the
    brief's [0.02, 0.01, 0.005].
    """
    d_coarse = abs(n1 - n2)
    d_fine = abs(n2 - n3)
    assert d_fine > 1e-12, f"consecutive difference |n2-n3|={d_fine:.3e} too small to estimate order"
    return math.log(d_coarse / d_fine) / math.log(2)


@pytest.mark.slow
def test_smoothed_convergence_beats_staircase():
    dxs = [0.04, 0.02, 0.01]
    n_smooth = [solve_at_dx(d, True, 0.37) for d in dxs]
    n_stair = [solve_at_dx(d, False, 0.37) for d in dxs]
    p_smooth = _observed_order(*n_smooth)
    p_stair = _observed_order(*n_stair)
    assert p_smooth > 1.5            # near-2nd order with smoothing
    assert p_smooth > p_stair + 0.4  # and clearly better than staircase