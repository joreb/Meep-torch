# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import TMState, init_tm_state


def test_dt_from_courant():
    g = Grid(nx=10, ny=10, dx=0.02)
    assert g.dt == pytest.approx(0.01)


def test_cfl_rejected():
    with pytest.raises(ValueError, match="courant"):
        Grid(nx=10, ny=10, dx=0.02, courant=0.8)


def test_tiny_grid_rejected():
    with pytest.raises(ValueError, match="nx"):
        Grid(nx=2, ny=10, dx=0.02)


def test_state_shapes_pec():
    g = Grid(nx=8, ny=6, dx=0.02)
    s = init_tm_state(g, n_poles=2)
    assert s.Ez.shape == (8, 6)
    assert s.Hx.shape == (8, 5)
    assert s.Hy.shape == (7, 6)
    assert s.Dz.shape == (8, 6)
    assert s.P_cur.shape == (2, 8, 6)
    assert s.psi_Hyx.shape == (6, 4)   # (nx-2, ny-2) PEC-y
    assert s.psi_Hxy.shape == (6, 4)
    assert s.Ez.dtype == torch.float64


def test_state_shapes_pmc_y():
    g = Grid(nx=8, ny=6, dx=0.02)
    s = init_tm_state(g, y_pmc=True)
    assert s.psi_Hyx.shape == (6, 6)   # full ny when y is PMC
    assert s.psi_Hxy.shape == (6, 6)
