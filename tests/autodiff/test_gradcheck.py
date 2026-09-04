# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.stepper_tm import init_tm_state

G = Grid(nx=16, ny=16, dx=0.05)


def _blob():
    x = torch.arange(16, dtype=torch.float64, device=G.device)[:, None]
    y = torch.arange(16, dtype=torch.float64, device=G.device)[None, :]
    return torch.exp(-((x - 8.0) ** 2 + (y - 8.0) ** 2) / 4.0)


def _loss_for(eps=None, chi3=None, pole_mask=None, n_steps=10, checkpoint_every=4):
    poles = [(1.5, 0.2, 0.8)] if pole_mask is not None else []
    masks = [pole_mask] if pole_mask is not None else None
    m = build_tm_coeffs(G, eps_inf=eps, poles=poles, pole_masks=masks, chi3=chi3, npml=3)
    s = init_tm_state(G, n_poles=len(poles))
    s = s._replace(Ez=_blob(), Dz=_blob())
    s, _ = run_tm(s, m, n_steps, checkpoint_every=checkpoint_every)
    return (s.Ez**2).sum()


def test_gradcheck_eps():
    def f(eps):
        return _loss_for(eps=eps)
    eps0 = torch.full((16, 16), 2.0, dtype=torch.float64, device=G.device, requires_grad=True)
    assert torch.autograd.gradcheck(f, (eps0,), eps=1e-6, atol=1e-4)


def test_gradcheck_chi3():
    def f(chi3):
        return _loss_for(eps=2.0, chi3=chi3)
    x0 = torch.full((16, 16), 0.05, dtype=torch.float64, device=G.device, requires_grad=True)
    assert torch.autograd.gradcheck(f, (x0,), eps=1e-6, atol=1e-4)


def test_gradcheck_pole_mask():
    def f(mask):
        return _loss_for(eps=2.0, pole_mask=mask)
    m0 = torch.full((16, 16), 0.5, dtype=torch.float64, device=G.device, requires_grad=True)
    assert torch.autograd.gradcheck(f, (m0,), eps=1e-6, atol=1e-4)


def test_gradcheck_uncheckpointed_matches():
    eps0 = torch.full((16, 16), 2.0, dtype=torch.float64, device=G.device, requires_grad=True)
    l1 = _loss_for(eps=eps0, checkpoint_every=0)
    (g1,) = torch.autograd.grad(l1, eps0)
    l2 = _loss_for(eps=eps0, checkpoint_every=3)
    (g2,) = torch.autograd.grad(l2, eps0)
    assert torch.allclose(g1, g2, atol=1e-12, rtol=1e-10)
