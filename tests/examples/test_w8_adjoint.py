# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W8 T2: adjoint gradient gate.

Autograd through the full differentiable chain
    theta -> eps_inf(theta) -> build_tm_coeffs -> run_tm(checkpoint_every=200)
    -> monitor accumulation -> scalar objective
must match central finite difference to <=1e-3 relative. Two objectives:
(a) |DFT(probe region)|^2 from a DFTFieldsMonitor, (b) flux through a
far-side FluxMonitorTM. The FD is the ground truth here (no meep oracle
needed for a gradient gate).
"""
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor, FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

F0 = 1.0
# GaussianPulse(1.0, 0.4): t0=12.5 (=500 steps), t_end=25.0 (=1000 steps).
# 1400 steps = full pulse + propagation to the far monitor + margin.
N_STEPS = 1400
CKPT = 200


def _blob_eps(theta, grid, cx, cy, r, edge):
    x = torch.arange(grid.nx, dtype=grid.dtype)[:, None]
    y = torch.arange(grid.ny, dtype=grid.dtype)[None, :]
    dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    disk = 0.5 * (1.0 - torch.tanh((dist - r) / edge))
    return 1.0 + theta * disk


def _run(grid, theta, monitor, lossy=False):
    blob = _blob_eps(theta, grid, cx=24.0, cy=24.0, r=7.0, edge=1.0)
    kw = dict(eps_inf=blob, npml=6)
    if lossy:
        # conductivity proportional to theta too: transmitted flux responds
        # strongly to absorption, and the sigma_d path joins the adjoint chain
        kw["sigma_d"] = (blob - 1.0) * 1.0
    coeffs = build_tm_coeffs(grid, **kw)
    src = LineSource(grid, ix=8, j0=18, j1=30, profile=GaussianPulse(F0, 0.4),
                     amp_func=lambda y: 5.0)
    state, (acc,) = run_tm(init_tm_state(grid), coeffs, N_STEPS, source=src,
                           monitors=[monitor], checkpoint_every=CKPT)
    return acc


def _dft_objective(theta):
    grid = Grid(nx=48, ny=48, dx=0.05)
    mon = DFTFieldsMonitor(grid, [F0], component="Ez")
    acc = _run(grid, theta, mon)
    re, im = acc[0, 0], acc[0, 1]              # Re/Im of Ez e^{-iwt} at F0
    return (re[26:36, 18:30] ** 2 + im[26:36, 18:30] ** 2).sum()


def _flux_objective(theta):
    grid = Grid(nx=48, ny=48, dx=0.05)
    mon = FluxMonitorTM(grid, [F0], x_normal=True, idx=34, lo=16, hi=32)
    acc = _run(grid, theta, mon, lossy=True)
    return mon.flux(acc, grid.dx).sum()


def _fd_grad(f, theta0, h=1e-4):
    return (f(theta0 + h) - f(theta0 - h)) / (2 * h)


def _assert_matches_fd(objective, name):
    theta0 = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    J = objective(theta0)
    (g,) = torch.autograd.grad(J, theta0)
    assert abs(float(g)) > 1e-6, f"{name}: degenerate zero gradient"
    g_fd = _fd_grad(objective, float(theta0.detach()))
    err = abs(g - g_fd) / max(abs(g_fd), 1e-12)
    assert err <= 1e-3, f"{name}: autograd {g:.8g} vs FD {g_fd:.8g} (rel {err:.3g})"


def test_adjoint_grad_matches_fd_dft_objective():
    _assert_matches_fd(_dft_objective, "dft-objective")


def test_adjoint_grad_matches_fd_flux_monitor():
    _assert_matches_fd(_flux_objective, "flux-monitor")
