# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Filtered topology optimization (meep-adjoint-shaped API)
#
# Full topology optimization workflow with the MaterialGrid + DesignRegion +
# OptimizationProblem API: a conic filter enforces a minimum lengthscale and a
# tanh-projection beta ramp (4 -> 32) drives the design toward binary, while
# Adam maximizes transmitted flux through the design region.

# %%
"""Filtered topology optimization with the meep-adjoint-shaped API.

Maximize transmitted flux through a design region using
MaterialGrid + DesignRegion + OptimizationProblem with the conic-filter
minimum lengthscale and a tanh-projection beta ramp (4 -> 32) toward a
binarized design.

Run: uv run python examples/07_adjoint_inverse_design/topology_opt_filtered.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.adjoint import DesignRegion, MaterialGrid, OptimizationProblem
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Design region and forward model
#
# A 12x10 MaterialGrid maps onto a 20x20-cell design region; `run_fn` is the
# forward FDTD returning flux through a narrow output window, checkpointed
# for the backward pass.

# %%
RES = 10
dx = 1.0 / RES
nx, ny = 80, 50
grid = Grid(nx=nx, ny=ny, dx=dx)
mg = MaterialGrid((12, 10), eps_min=1.0, eps_max=6.0)
region = DesignRegion(mg, ix0=30, ix1=50, iy0=15, iy1=35)


def run_fn(eps_full):
    coeffs = build_tm_coeffs(grid, eps_inf=eps_full, npml=8)
    src = LineSource(grid, ix=12, j0=15, j1=35, profile=GaussianPulse(1.0, 0.3))
    mon = FluxMonitorTM(grid, [1.0], x_normal=True, idx=64, lo=21, hi=29)
    _, (acc,) = run_tm(init_tm_state(grid), coeffs, int(round(30.0 / grid.dt)),
                       source=src, monitors=[mon], checkpoint_every=100)
    return mon.flux(acc, dx)[0]


# %% [markdown]
# ## Beta-ramped Adam loop and final binarized design
#
# 20 Adam iterations with beta stepped 4 -> 8 -> 16 -> 32 every 5 iterations;
# the final design is re-evaluated at beta = 32 and its binarization fraction
# reported.

# %%
base = torch.ones(nx, ny, dtype=torch.float64)
# start near-empty: a gray rho=0.5 slab is itself a good transmitter, so
# binarization from there reads as a net loss -- build the lens up instead
opt = OptimizationProblem(run_fn, base, region, filter_radius=0.25, dx=dx,
                          beta=4.0,
                          rho0=torch.full((12, 10), 0.2, dtype=torch.float64))
# Adam on rho (sign-normalized fixed steps oscillate in this landscape)
rho = opt.rho.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history = []
for it in range(20):
    opt.beta = [4, 8, 16, 32][min(it // 5, 3)]
    adam.zero_grad()
    J = opt._objective(rho.clamp(0.0, 1.0))
    (-J).backward()
    adam.step()
    history.append(float(J))
    print(f"topology_opt_filtered  it={it} beta={opt.beta}: J = {float(J):.4e}")
opt.rho = rho.detach().clamp(0.0, 1.0)
opt.beta = 32.0
f_final, _ = opt(need_gradient=False)
history.append(f_final)

r = tanh_projection(conic_filter(opt.rho, 0.25, dx), 32.0)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.plot(history, "bo-")
ax1.set_xlabel("iteration")
ax1.set_ylabel("transmitted flux")
ax1.set_title(f"objective x{history[-1] / history[0]:.1f}")
ax2.imshow(r.detach().numpy().T, origin="lower", cmap="binary", vmin=0, vmax=1)
ax2.set_title("final design (filtered + beta=32)")
fig.tight_layout()
fig.savefig(OUT / "topology-opt-filtered.png", dpi=150)
plt.close(fig)
frac_binary = float(((r < 0.05) | (r > 0.95)).float().mean())
print(f"topology_opt_filtered  final J x{history[-1] / history[0]:.2f}, "
      f"binarization {frac_binary:.0%}")
