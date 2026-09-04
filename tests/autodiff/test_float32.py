# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state


@pytest.mark.slow
def test_float32_stable_at_2000x2000():
    with torch.no_grad():
        g = Grid(nx=2000, ny=2000, dx=0.02, dtype=torch.float32)
        m = build_tm_coeffs(g, eps_inf=2.25, npml=20)
        src = LineSource(g, ix=200, j0=400, j1=1600, profile=GaussianPulse(0.65, 0.2))
        s = init_tm_state(g)
        s, _ = run_tm(s, m, 2000, source=src, checkpoint_every=0)
        assert torch.isfinite(s.Ez).all()
        assert torch.isfinite(s.Hx).all() and torch.isfinite(s.Hy).all()
        assert s.Ez.abs().max() < 1e3       # bounded, no runaway growth
