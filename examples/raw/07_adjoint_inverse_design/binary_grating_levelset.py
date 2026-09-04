# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Level-set shape gradient of a binary grating
#
# Adjoint derivative of a grating objective with respect to a *shape*
# parameter — the duty cycle of a sigmoid level-set grating — validated
# against a central finite difference, then used for five Adam steps.
# Because `run_tm` is autograd-differentiable, the level-set adjoint is a
# single `torch.autograd.grad` call.

# %%
"""meep-torch port of meep/python/examples/adjoint_optimization/binary_grating_levelset.py.

Adjoint derivative of a grating objective with respect to a shape
parameter, validated against a brute-force finite difference. The meep
original differentiates the first-order diffraction efficiency of a 1D
grating with respect to the grating HEIGHT (a level-set surface) using
meep.adjoint + autograd. meep-torch has no meep.adjoint -- run_tm is
itself differentiable via torch autograd (W8), so the adjoint gradient is
just `torch.autograd.grad`.

Reduction (per port instructions):
  * the level-set shape parameter is the grating DUTY CYCLE (continuous,
    0..1) rather than the height; the binary grating is painted as
        eps(x) = eps_lo + (eps_hi - eps_lo)
                 * sigmoid((cos(2 pi x / period) - cos(pi * duty)) / edge)
    so the high-index fraction per period equals `duty` and the boundary
    positions are autograd-differentiable in `duty`;
  * the first-order diffraction efficiency -> the total transmitted flux
    through the grating (a clean differentiable scalar; the level-set
    adjoint machinery is the point);
  * a finite multi-period grating with PML on all sides + a horizontal
    Jz line source (planewave from below), transmitted flux monitored
    above the grating.

Gate (printed): autograd dJ/d(duty) vs central finite difference, relative
error <= 1e-2; then 5 Adam steps maximizing transmission, objective per
step printed.

Run: uv run python examples/07_adjoint_inverse_design/binary_grating_levelset.py
"""
import pathlib

import matplotlib
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, SoftSource
from meep_torch.core.stepper_tm import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Differentiable grating and flux objective
#
# The grating band is painted as
# eps(x) = eps_lo + (eps_hi - eps_lo) * sigmoid((cos(2 pi x / P) - cos(pi d)) / edge),
# so the high-index fraction per period equals the duty d and the tooth
# boundaries move differentiably. The objective is transmitted flux above the
# grating under a planewave from below.

# %%
FCEN, DF = 1.0, 0.4
N_STEPS, CKPT = 1200, 200
NX, NY, PML = 120, 80, 10
DX = 0.05
GRID = Grid(nx=NX, ny=NY, dx=DX)
EPS_HI, EPS_LO = 3.5, 1.0
PERIOD = 20 * DX                  # grating period (20 cells)
EDGE = 0.7 * DX                   # level-set smoothness (~1 cell)
J_GRAT0, J_GRAT1 = 32, 48         # grating slab row band (y cells)


def _grating_eps(duty):
    """Differentiable binary-grating eps from a continuous duty-cycle param."""
    x = (torch.arange(NX, dtype=GRID.dtype, device=GRID.device) + 0.5) * DX
    phi = torch.cos(2 * torch.pi * x / PERIOD)
    level = torch.sigmoid((phi - torch.cos(math.pi * duty)) / EDGE)
    eps = torch.ones(NX, NY, dtype=GRID.dtype, device=GRID.device)
    band = EPS_LO + (EPS_HI - EPS_LO) * level          # (NX,)
    eps[:, J_GRAT0:J_GRAT1] = band[:, None]
    return eps


def _objective(duty):
    eps = _grating_eps(duty)
    coeffs = build_tm_coeffs(GRID, eps_inf=eps, npml=PML)
    mask = GRID.zeros(NX, NY)
    mask[PML:NX - PML, PML + 2] = 1.0   # horizontal Jz line -> planewave upward
    src = SoftSource(GRID, mask, GaussianPulse(FCEN, DF))
    mon = FluxMonitorTM(GRID, [FCEN], x_normal=False,
                        idx=NY - PML - 3, lo=PML, hi=NX - PML)
    _, (acc,) = run_tm(init_tm_state(GRID), coeffs, N_STEPS,
                       source=src, monitors=[mon], checkpoint_every=CKPT)
    return mon.flux(acc, GRID.dx).sum()


def _fd_grad(f, theta0, h=1e-3):
    return (f(torch.tensor(theta0 + h)) - f(torch.tensor(theta0 - h))) / (2 * h)


# %% [markdown]
# ## Gradient gate and optimization
#
# First the autograd gradient must match the central finite difference to
# within 1e-2 relative error (the gate); then five Adam steps maximize
# transmission with the duty clamped to [0.05, 0.95].

# %%
def main():
    # --- gradient gate: autograd vs central finite difference ---
    duty0 = torch.tensor(0.5, dtype=GRID.dtype, device=GRID.device, requires_grad=True)
    J = _objective(duty0)
    (g,) = torch.autograd.grad(J, duty0)
    g_fd = _fd_grad(lambda d: _objective(d), float(duty0.detach()))
    err = abs(float(g) - float(g_fd)) / max(abs(float(g_fd)), 1e-12)
    print(f"binary_grating_levelset  grad: autograd={float(g):.6g} "
          f"FD={float(g_fd):.6g}  rel_err={err:.3g}")
    assert abs(float(g)) > 1e-6, "degenerate zero gradient"
    assert err <= 1e-2, f"autograd vs FD rel err {err:.3g} > 1e-2"
    print("binary_grating_levelset  GATE PASS: autograd vs FD rel err "
          f"{err:.3g} <= 1e-2")

    # --- 5 Adam steps maximizing transmitted flux ---
    duty = torch.nn.Parameter(torch.tensor(0.5, dtype=GRID.dtype, device=GRID.device))
    opt = torch.optim.Adam([duty], lr=0.05)
    flux = []
    for step in range(5):
        opt.zero_grad()
        loss = -_objective(duty)
        loss.backward()
        opt.step()
        with torch.no_grad():
            duty.clamp_(min=0.05, max=0.95)
            f = float(-loss)
        flux.append(f)
        print(f"binary_grating_levelset  step {step + 1}: "
              f"transmitted flux = {f:.6g}  duty={float(duty.detach()):.3f}")

    with torch.no_grad():
        eps = _grating_eps(duty.detach()).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(eps.T, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(f"optimized grating eps (duty={float(duty):.3f})")
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
    fig.tight_layout()
    fig.savefig(OUT / "binary_grating_levelset.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT / 'binary_grating_levelset.png'}  "
          f"(flux {flux[0]:.6g} -> {flux[-1]:.6g})")


if __name__ == "__main__":
    main()