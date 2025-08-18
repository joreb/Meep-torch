# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c1): inverse-designed edge coupler, SiN 150 nm / SiO2
(tidy3d EdgeCoupler-notebook style, on the meep-torch adjoint toolchain).

Effective-index 2D (150 nm SiN slab in SiO2 -> TE0 neff printed): a wide
Gaussian "fiber" field enters from the left; the design region converts
it into the 0.5 um waveguide mode on the right. Objective = transmitted
flux into the waveguide; conic filter (min lengthscale 150 nm) + beta
ramp.

Run: uv run python examples_2/c_edge_coupler.py
"""
import math
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
from meep_torch.modes.fd1d import solve_1d

OUT = pathlib.Path(__file__).parent

# 150 nm SiN slab TE0 effective index (the platform the task specifies)
eps_line = np.full(400, 1.44 ** 2)
eps_line[200 - 7: 200 + 8] = 2.0 ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
print(f"c_edge_coupler  150nm SiN/SiO2 TE0 neff = {NEFF:.4f}")

RES = 20
FCEN = 1 / 1.55
dx = 1.0 / RES
nx, ny = int(16.0 * RES), int(8.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
cy = ny // 2
W_WG = 0.5
W_FIBER = 2.5                                     # fiber MFD stand-in

base = torch.full((nx, ny), 1.44 ** 2, dtype=torch.float64)
i_wg0 = int(9.0 * RES)
jw = int(W_WG / 2 * RES)
base[i_wg0:, cy - jw: cy + jw] = NEFF ** 2        # output waveguide

mg = MaterialGrid((24, 20), eps_min=1.44 ** 2 ** 0.5 * 0 + 1.44 ** 2,
                  eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=int(3.0 * RES), ix1=i_wg0,
                      iy0=cy - int(1.5 * RES), iy1=cy + int(1.5 * RES))


def run_fn(eps_full):
    coeffs = build_tm_coeffs(grid, eps_inf=eps_full, npml=RES)
    src = LineSource(grid, ix=int(1.5 * RES), j0=RES, j1=ny - RES,
                     profile=GaussianPulse(FCEN, 0.1))
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    src.mask = torch.zeros(nx, ny, dtype=torch.float64)
    src.mask[int(1.5 * RES), :] = torch.exp(-((ys - cy * dx) / (W_FIBER / 2)) ** 2)
    mon = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=nx - int(1.5 * RES),
                        lo=cy - 3 * jw, hi=cy + 3 * jw)
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(45.0 / grid.dt)), source=src,
                       monitors=[mon], checkpoint_every=300)
    return mon.flux(acc, dx)[0]


opt = OptimizationProblem(run_fn, base, region, filter_radius=0.15, dx=dx,
                          beta=4.0,
                          rho0=torch.full((24, 20), 0.5, dtype=torch.float64))
rho = opt.rho.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.04)
history = []
for it in range(16):
    opt.beta = [4, 8, 16, 32][min(it // 4, 3)]
    adam.zero_grad()
    J = opt._objective(rho.clamp(0.0, 1.0))
    (-J).backward()
    adam.step()
    history.append(float(J))
    print(f"c_edge_coupler  it={it} beta={opt.beta}: J = {float(J):.4e}")
opt.rho = rho.detach().clamp(0.0, 1.0)

r = tanh_projection(conic_filter(opt.rho, 0.15, dx), 32.0)
eps_fin = region.paint(base, r)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6))
ax1.plot(history, "bo-")
ax1.set_xlabel("iteration")
ax1.set_ylabel("coupled flux")
ax1.set_title(f"edge coupler: objective x{history[-1] / history[0]:.2f}")
ax2.imshow(eps_fin.detach().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax2.set_title("final design (fiber left -> 0.5 um waveguide right)")
ax2.set_xlabel("x (um)")
fig.tight_layout()
fig.savefig(OUT / "c_edge_coupler.png", dpi=150)
plt.close(fig)
print(f"c_edge_coupler  final J x{history[-1] / history[0]:.2f}")
