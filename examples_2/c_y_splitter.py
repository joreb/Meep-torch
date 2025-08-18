# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2): inverse-designed Y splitter, SiN 150 nm / SiO2
(tidy3d Autograd-splitter-notebook style, meep-torch adjoint toolchain).

Effective-index 2D; input waveguide left, two output waveguides right.
The design half-grid is MIRRORED about the axis (symmetry-enforced
50/50); objective = total transmitted flux into the two arms. Balance is
verified by construction and printed.

Run: uv run python examples_2/c_y_splitter.py
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
eps_line = np.full(400, 1.44 ** 2)
eps_line[200 - 7: 200 + 8] = 2.0 ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
print(f"c_y_splitter  150nm SiN/SiO2 TE0 neff = {NEFF:.4f}")

RES = 20
FCEN = 1 / 1.55
dx = 1.0 / RES
nx, ny = int(12.0 * RES), int(8.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
cy = ny // 2
jw = int(0.25 * RES)                              # 0.5 um waveguides
SEP = int(1.2 * RES)                              # arm offset from axis

base = torch.full((nx, ny), 1.44 ** 2, dtype=torch.float64)
i_in1 = int(3.0 * RES)
i_out0 = int(9.0 * RES)
base[:i_in1, cy - jw: cy + jw] = NEFF ** 2
base[i_out0:, cy + SEP - jw: cy + SEP + jw] = NEFF ** 2
base[i_out0:, cy - SEP - jw: cy - SEP + jw] = NEFF ** 2

mg = MaterialGrid((24, 16), eps_min=1.44 ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=i_in1, ix1=i_out0,
                      iy0=cy - int(2.0 * RES), iy1=cy + int(2.0 * RES))


def run_fn(eps_full):
    coeffs = build_tm_coeffs(grid, eps_inf=eps_full, npml=RES)
    src = LineSource(grid, ix=int(1.5 * RES), j0=cy - 2 * jw, j1=cy + 2 * jw,
                     profile=GaussianPulse(FCEN, 0.1))
    m1 = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=nx - int(1.5 * RES),
                       lo=cy + SEP - 3 * jw, hi=cy + SEP + 3 * jw)
    m2 = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=nx - int(1.5 * RES),
                       lo=cy - SEP - 3 * jw, hi=cy - SEP + 3 * jw)
    _, (a1, a2) = run_tm(init_tm_state(grid), coeffs,
                         int(round(40.0 / grid.dt)), source=src,
                         monitors=[m1, m2], checkpoint_every=300)
    return m1.flux(a1, dx)[0], m2.flux(a2, dx)[0]


class SymmetricSplitter(OptimizationProblem):
    def _objective(self, rho_half):
        # mirror the half design about the axis -> exact 50/50 symmetry
        full = torch.cat([rho_half.flip(1), rho_half], dim=1)
        r = conic_filter(full, self.filter_radius, self.dx)
        r = tanh_projection(r, self.beta, self.eta)
        eps = self.region.paint(self.base_eps, r)
        p1, p2 = self.run_fn(eps)
        return p1 + p2


opt = SymmetricSplitter(run_fn, base, region, filter_radius=0.15, dx=dx,
                        beta=4.0,
                        rho0=torch.full((24, 8), 0.5, dtype=torch.float64))
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
    print(f"c_y_splitter  it={it} beta={opt.beta}: J = {float(J):.4e}")
opt.rho = rho.detach().clamp(0.0, 1.0)

full = torch.cat([opt.rho.flip(1), opt.rho], dim=1)
r = tanh_projection(conic_filter(full, 0.15, dx), 32.0)
eps_fin = region.paint(base, r)
with torch.no_grad():
    p1, p2 = run_fn(eps_fin)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6))
ax1.plot(history, "bo-")
ax1.set_xlabel("iteration")
ax1.set_ylabel("total transmitted flux")
ax1.set_title(f"Y splitter: objective x{history[-1] / history[0]:.2f}, "
              f"split {float(p1):.3e} / {float(p2):.3e}")
ax2.imshow(eps_fin.detach().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax2.set_title("final design (input left -> two arms right)")
ax2.set_xlabel("x (um)")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter.png", dpi=150)
plt.close(fig)
imb = abs(float(p1) - float(p2)) / max(float(p1) + float(p2), 1e-30)
print(f"c_y_splitter  final J x{history[-1] / history[0]:.2f}, "
      f"imbalance {imb:.1%} (symmetry-enforced)")
