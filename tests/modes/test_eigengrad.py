# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import pytest
import torch

from meep_torch.modes.fd2d import solve_2d, solve_2d_diff


def tiny_eps():
    eps = torch.full((24, 20), 2.0, dtype=torch.float64)
    eps[9:15, 8:12] = 8.0
    return eps


def test_forward_matches_nondiff_path():
    eps = tiny_eps()
    a = solve_2d(eps, dx=0.05, f=1 / 1.55, k=1).neff[0].real.item()
    b = solve_2d_diff(eps, dx=0.05, f=1 / 1.55, k=1)[0].item()
    assert b == pytest.approx(a, rel=1e-12)


def test_gradcheck_neff_wrt_eps():
    eps0 = tiny_eps().requires_grad_(True)
    def f(e):
        return solve_2d_diff(e, dx=0.05, f=1 / 1.55, k=1)[0]
    assert torch.autograd.gradcheck(f, (eps0,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_grad_sign_core_positive():
    eps0 = tiny_eps().requires_grad_(True)
    neff = solve_2d_diff(eps0, dx=0.05, f=1 / 1.55, k=1)[0]
    neff.backward()
    assert torch.all(eps0.grad[10:14, 9:11] > 0)
    assert eps0.grad.abs().sum() > 0


def test_degenerate_guard_raises():
    eps = torch.full((30, 30), 2.0, dtype=torch.float64)
    eps[12:18, 12:18] = 8.0
    with pytest.raises(RuntimeError, match="degenerate"):
        _ = solve_2d_diff(eps.requires_grad_(True), dx=0.05, f=1 / 1.55, k=3).sum().backward()
