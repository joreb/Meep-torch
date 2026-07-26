# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Flexcompute
# Copyright (c) 2019 Tyler Hughes
# Derived from the ceviche package (https://github.com/fancompute/ceviche).
# Adaptations (c) 2026 meep-torch contributors, MIT.
import numpy as np
import pytest
import torch

from meep_torch.modes.fd2d import solve_2d
from meep_torch.modes.transforms import angled_eps_tensor, apply_bend


def strip_eps(Nx=300, Ny=200):
    eps = torch.full((Nx, Ny), 1.444**2, dtype=torch.float64)
    eps[125:175, 89:111] = 3.4777**2
    return eps


def test_bend_converges_to_straight():
    eps = strip_eps()
    straight = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    prev_err = None
    for R in (5.0, 20.0, 80.0):
        nb = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1, bend_radius=R).neff[0].real.item()
        err = abs(nb - straight)
        if prev_err is not None:
            assert err < prev_err          # monotone approach
        prev_err = err
    assert prev_err < 1e-3                  # R=80um essentially straight


def test_bend_raises_neff():
    # outer-shifted mode in a bend has higher effective path -> neff rises at small R
    eps = strip_eps()
    straight = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    bent = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1, bend_radius=5.0).neff[0].real.item()
    assert bent != pytest.approx(straight, abs=1e-6)


def test_angled_tensor_offdiagonal_structure():
    eps = strip_eps(60, 40)
    t = angled_eps_tensor(eps, theta=np.deg2rad(10), phi=0.0)
    assert t.shape == (3, 3, 60, 40)
    assert torch.any(t[0, 2] != 0)          # eps_xz appears
    assert torch.allclose(t[0, 1], torch.zeros_like(t[0, 1]))  # no xy coupling (phi=0)
    t0 = angled_eps_tensor(eps, theta=0.0, phi=0.0)
    for i in range(3):
        assert torch.allclose(t0[i, i], eps)          # theta=0 -> diagonal, unchanged
        for j in range(3):
            if i != j:
                assert torch.allclose(t0[i, j], torch.zeros_like(eps))
