# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5.5 T1: magnetic Lorentz poles — mu-dispersive slab R/T vs transfer matrix."""
import cmath
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

FCEN = 1.0
F0, GAMMA, SIGMA = 2.0, 0.1, 0.3
L = 2.0


def _mu(f):
    return 1.0 + SIGMA * F0 ** 2 / (F0 ** 2 - f ** 2 - 1j * GAMMA * f)


def _rt_analytic(f):
    mu = _mu(f)
    n = cmath.sqrt(mu)
    Z = cmath.sqrt(mu)
    delta = 2 * math.pi * f * n * L
    cosd, sind = cmath.cos(delta), cmath.sin(delta)
    t = 1.0 / (cosd - 0.5j * (Z + 1 / Z) * sind)
    r = -0.5j * (Z - 1 / Z) * sind * t
    return abs(r), abs(t)


def _run(with_slab):
    # 40 px/lambda: the small |R| ~ 0.117 needs it (8% err at 20 px/lambda)
    grid = Grid(nx=1200, ny=8, dx=0.025)
    kwargs = {}
    if with_slab:
        mask = grid.zeros(1200, 8)
        mask[600:680, :] = 1.0                    # slab x in [15, 17]
        kwargs["m_poles"] = [(F0, GAMMA, SIGMA)]
        kwargs["m_pole_masks"] = [mask]
    coeffs = build_tm_coeffs(grid, eps_inf=grid.zeros(1200, 8) + 1.0,
                             npml=60, y_pmc=True, **kwargs)
    src = LineSource(grid, ix=120, j0=0, j1=8, profile=GaussianPulse(FCEN, 0.2))
    probes = [PointProbe(840, 4), PointProbe(300, 4)]   # transmitted, incident-side
    n_steps = int(round(80.0 / grid.dt))
    state0 = init_tm_state(grid, y_pmc=True, n_mpoles=1 if with_slab else 0)
    _, (st, sr) = run_tm(state0, coeffs, n_steps, source=src, monitors=probes)
    return st.detach().numpy(), sr.detach().numpy(), grid.dt


def _dft(s, dt, f):
    t = np.arange(len(s)) * dt
    return np.trapezoid(s * np.exp(-2j * math.pi * f * t), t)


def test_mu_lorentz_slab_rt():
    st1, sr1, dt = _run(True)
    st0, sr0, _ = _run(False)
    T = abs(_dft(st1, dt, FCEN) / _dft(st0, dt, FCEN))
    R = abs(_dft(sr1 - sr0, dt, FCEN) / _dft(sr0, dt, FCEN))
    r_an, t_an = _rt_analytic(FCEN)
    assert abs(T - t_an) / t_an < 0.03, f"|T|={T:.4f} vs {t_an:.4f}"
    assert abs(R - r_an) / max(r_an, 1e-9) < 0.03, f"|R|={R:.4f} vs {r_an:.4f}"
