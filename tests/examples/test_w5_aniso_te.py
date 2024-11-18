# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5 T5: in-plane anisotropic eps in the TE stepper."""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_te
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_te import init_te_state

FCEN = 1.0


class _EyProbe:
    reduce = "stack"

    def __init__(self, ix, iy):
        self.ix, self.iy = ix, iy

    def sample(self, state, t_idx):
        return state.Ey[self.ix, self.iy]


def _run(coeffs, grid, probes, t_end=40.0):
    src = LineSource(grid, ix=40, j0=20, j1=140, profile=GaussianPulse(FCEN, 0.2))
    src.mask = src.mask[:, :-1]
    _, out = run_te(init_te_state(grid), coeffs, int(round(t_end / grid.dt)),
                    source=src, monitors=probes)
    return [o.detach().numpy() for o in out]


def test_diagonal_tensor_matches_isotropic():
    grid = Grid(nx=200, ny=160, dx=0.05)
    eps = grid.zeros(200, 160) + 2.25
    iso = build_te_coeffs(grid, eps_inf=eps, npml=20)
    tens = build_te_coeffs(grid, eps_inf=eps, npml=20,
                           eps_tensor=(eps, torch.zeros_like(eps), eps))
    probes = [_EyProbe(120, 80)]
    (a,) = _run(iso, grid, probes)
    (b,) = _run(tens, grid, probes)
    assert np.allclose(a, b, atol=1e-12 * np.abs(a).max()), \
        f"max dev {np.abs(a - b).max():.2e}"


def test_rotated_uniaxial_neff():
    # optic axis 45 deg in-plane, no=1.5, ne=2.0 -> for k along x,
    # n_eff^2 = det(eps)/exx = 9.0/3.125 (index ellipse at 45 deg)
    no2, ne2 = 1.5 ** 2, 2.0 ** 2
    exx_v = (ne2 + no2) / 2
    exy_v = (ne2 - no2) / 2
    n_an = math.sqrt((exx_v * exx_v - exy_v * exy_v) / exx_v)   # det/exx, eyy=exx
    grid = Grid(nx=200, ny=160, dx=0.05)
    full = grid.zeros(200, 160)
    coeffs = build_te_coeffs(grid, eps_inf=full + exx_v, npml=20,
                             eps_tensor=(full + exx_v, full + exy_v, full + exx_v))
    p1, p2 = _EyProbe(100, 80), _EyProbe(108, 80)               # dx_probe = 0.4
    s1, s2 = _run(coeffs, grid, [p1, p2])
    t = np.arange(len(s1)) * grid.dt
    kern = np.exp(-2j * math.pi * FCEN * t)
    ph = np.angle(np.trapezoid(s2 * kern, t) / np.trapezoid(s1 * kern, t))
    # e^{+iwt} phasors: wave ~ e^{-ikx}, so ph = -k dx (mod 2 pi); the branch
    # is unique for n in (1.25, 3.75) at dx_probe = 0.4
    dphi = (-ph) % (2 * math.pi)
    n_eff = dphi / (2 * math.pi * FCEN * 0.4)
    assert abs(n_eff - n_an) / n_an < 0.01, f"n_eff={n_eff:.4f} vs {n_an:.4f}"
