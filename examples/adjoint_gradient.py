# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W8 adjoint / inverse-design demo (2D TM).

A full-height line source launches a pulse through a lossy lens-shaped
scatterer whose material (eps AND conductivity, both scaled by the design
variables) is parameterized by a coarse 4x4 control grid bilinearly
upsampled to the cell grid. Transmitted flux through a far-side monitor is
maximized with 5 Adam steps: each step is one full forward FDTD run
differentiated by autograd backward through run_tm(checkpoint_every=200) --
the torch adjoint. The absorbing obstacle blocks ~12x of the beam at
step 0; the optimizer clears it and transmission recovers. Objective per
step is printed; final eps + |Ez| field are saved as
examples/adjoint-gradient.png.

Run: uv run python examples/adjoint_gradient.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent

FCEN, DF = 1.0, 0.4
N_STEPS, CKPT = 1200, 200
NX = NY = 64
PML, CTRL = 8, 4            # npml, control-grid size (CTRL x CTRL)
GRID = Grid(nx=NX, ny=NY, dx=0.05)


def _support(grid):
    """Smooth lens-shaped region (center, radius, edge) where material is tunable."""
    x = torch.arange(grid.nx, dtype=grid.dtype)[:, None]
    y = torch.arange(grid.ny, dtype=grid.dtype)[None, :]
    r = torch.sqrt((x - 32.0) ** 2 + (y - 32.0) ** 2)
    return 0.5 * (1.0 - torch.tanh((r - 14.0) / 2.0))


def _eps_from_alpha(alpha):
    """Bilinear-upsample the CTRL x CTRL control grid onto the cell grid."""
    up = F.interpolate(alpha[None, None], size=(NX, NY), mode="bilinear",
                       align_corners=True)[0, 0]
    return 1.0 + up * _support(GRID)


def _objective(alpha):
    s = _support(GRID)
    up = F.interpolate(alpha[None, None], size=(NX, NY), mode="bilinear",
                       align_corners=True)[0, 0]
    coeffs = build_tm_coeffs(GRID, eps_inf=1.0 + up * s, npml=PML,
                             sigma_d=up * s * 3.0)   # lossy lens: strong gradient
    src = LineSource(GRID, ix=8, j0=8, j1=56, profile=GaussianPulse(FCEN, DF))
    mon = FluxMonitorTM(GRID, [FCEN], x_normal=True, idx=48, lo=8, hi=56)
    state, (acc,) = run_tm(init_tm_state(GRID), coeffs, N_STEPS, source=src,
                           monitors=[mon], checkpoint_every=CKPT)
    return mon.flux(acc, GRID.dx).sum()


def main():
    alpha = torch.nn.Parameter(torch.full((CTRL, CTRL), 1.0, dtype=GRID.dtype))
    opt = torch.optim.Adam([alpha], lr=0.8)

    flux = []
    for step in range(5):
        opt.zero_grad()
        loss = -_objective(alpha)
        loss.backward()
        opt.step()
        with torch.no_grad():
            alpha.clamp_(min=0.0)               # eps >= 1, sigma >= 0
            f = float(-loss)
        flux.append(f)
        print(f"step {step + 1}: transmitted flux = {f:.6g}")

    assert flux[-1] > flux[0], f"objective did not improve: {flux}"

    # final snapshot: eps field + |Ez| amplitude
    with torch.no_grad():
        eps = _eps_from_alpha(alpha.detach())
        coeffs = build_tm_coeffs(GRID, eps_inf=eps, npml=PML)
        src = LineSource(GRID, ix=8, j0=8, j1=56, profile=GaussianPulse(FCEN, DF))
        state, _ = run_tm(init_tm_state(GRID), coeffs, N_STEPS, source=src)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    im1 = ax1.imshow(eps.T, origin="lower", cmap="viridis")
    ax1.set_title("optimized eps")
    fig.colorbar(im1, ax=ax1, shrink=0.85)
    im2 = ax2.imshow(state.Ez.abs().T, origin="lower", cmap="inferno")
    ax2.set_title("|Ez|")
    fig.colorbar(im2, ax=ax2, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "adjoint-gradient.png", dpi=110)
    print(f"saved {OUT / 'adjoint-gradient.png'}  (flux {flux[0]:.6g} -> {flux[-1]:.6g})")


if __name__ == "__main__":
    main()
