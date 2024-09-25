# SPDX-License-Identifier: MIT
# Ported from tidy3d (https://github.com/flexcompute/tidy3d), components/mode
# solver, historic MIT-licensed revision. Copyright (c) Flexcompute Inc. --
# MIT License (ceviche-derived, (c) 2019 Tyler Hughes).
# Adaptations (c) 2026 meep-torch contributors, MIT.
import numpy as np
import pytest
import torch

from meep_torch.modes.fd2d import solve_2d
from meep_torch.modes.fd2d_tensor import solve_2d_tensor, solve_angled


def strip_eps(Nx=200, Ny=140):
    eps = torch.full((Nx, Ny), 1.444**2, dtype=torch.float64)
    eps[80:120, 59:81] = 3.4777**2
    return eps


def diag_tensor(eps):
    t = torch.zeros(3, 3, *eps.shape, dtype=torch.float64)
    for i in range(3):
        t[i, i] = eps
    return t


def test_diagonal_input_reproduces_diagonal_solver():
    eps = strip_eps()
    a = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    b = solve_2d_tensor(diag_tensor(eps), dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    assert b == pytest.approx(a, abs=1e-8)


def test_theta_zero_equals_straight():
    eps = strip_eps()
    a = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    b = solve_angled(eps, dx=0.01, f=1 / 1.55, theta=0.0, k=1).neff[0].real.item()
    assert b == pytest.approx(a, abs=1e-8)


def test_theta_symmetry():
    eps = strip_eps()
    p = solve_angled(eps, dx=0.01, f=1 / 1.55, theta=np.deg2rad(8), k=1).neff[0].real.item()
    m = solve_angled(eps, dx=0.01, f=1 / 1.55, theta=-np.deg2rad(8), k=1).neff[0].real.item()
    assert p == pytest.approx(m, rel=1e-6)


def test_small_theta_perturbation_scaling():
    # neff(theta) - neff(0) must scale ~theta^2 for small theta (perturbation theory:
    # first-order term vanishes by symmetry)
    eps = strip_eps()
    n0 = solve_angled(eps, dx=0.01, f=1 / 1.55, theta=0.0, k=1).neff[0].real.item()
    d2 = solve_angled(eps, dx=0.01, f=1 / 1.55, theta=np.deg2rad(2), k=1).neff[0].real.item() - n0
    d4 = solve_angled(eps, dx=0.01, f=1 / 1.55, theta=np.deg2rad(4), k=1).neff[0].real.item() - n0
    assert d4 / d2 == pytest.approx(4.0, rel=0.35)   # quadratic scaling, tolerant
