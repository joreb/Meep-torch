# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch rebuild of fdtdx/examples/optimize_ceviche_corner.py.

The ceviche-challenges 90-degree corner: route the fundamental TM mode
of a horizontal Si waveguide into a vertical one through a 1.6 x 1.6 um
design region. Objective = |S21|^2 (mode-to-mode power transmission via
the ports mode-overlap machinery -- self-normalizing, so T <= 1 and
directly comparable with the ceviche reference). Adjoint gradients
through run_tm, conic filter 100 nm, tanh beta ramp, Adam, L-bend seed.

Run: uv run python examples/fdtdx_ports/optimize_ceviche_corner.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor, FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
WL, RES = 1.55, 25
FCEN = 1 / WL
dx = 1.0 / RES
nx = ny = int(6.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
N_SI, N_OX = 3.45, 1.44
jw = int(0.2 * RES)                                # half-width, wg 0.4 um
cy = ny // 2
cx = nx // 2
d0, d1 = int(2.2 * RES), int(3.8 * RES)            # design box 1.6 um
NPML = RES
T_RUN = 80.0

base = torch.full((nx, ny), N_OX ** 2, dtype=torch.float64)
base[: d0, cy - jw: cy + jw] = N_SI ** 2           # input arm (left)
base[cx - jw: cx + jw, d1:] = N_SI ** 2            # output arm (top)
mg = MaterialGrid((40, 40), eps_min=N_OX ** 2, eps_max=N_SI ** 2)
region = DesignRegion(mg, ix0=d0, ix1=d1, iy0=d0, iy1=d1)

SPAN = int(0.8 * RES)                              # mode-window half-width
p_in = Port(axis="x", index=int(1.4 * RES), lo=cy - SPAN, hi=cy + SPAN,
            direction=+1)
p_out = Port(axis="y", index=int(4.6 * RES), lo=cx - SPAN, hi=cx + SPAN,
             direction=-1)                         # inward = -y
validate_ports([p_in, p_out], grid, NPML)
pulse = GaussianPulse(FCEN, 0.1)


def s_params(eps):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=(p.axis == "x"),
                          idx=p.index, lo=p.lo, hi=p.hi)
            for p in (p_in, p_out)]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=250)
    e1 = eps.detach()[p_in.index, p_in.lo:p_in.hi]
    e2 = eps.detach()[p_out.lo:p_out.hi, p_out.index]
    a1p, a1m = decompose(accs[0], e1, dx, [FCEN], "tm", 1, axis_sign=-1.0)
    a2p, a2m = decompose(accs[1], e2, dx, [FCEN], "tm", 1, axis_sign=+1.0)
    s21 = a2p[0, 0] / a1p[0, 0]                    # +y outgoing / +x incident
    s11 = a1m[0, 0] / a1p[0, 0]
    return s21, s11


def objective(rho):
    r = tanh_projection(conic_filter(rho, 0.10, dx), BETA)
    s21, _ = s_params(region.paint(base, r))
    return (s21.conj() * s21).real


# L-bend seed: both arms continued into the design box
seed = torch.full((40, 40), 0.02, dtype=torch.float64)
gj = int(round((cy - d0) / (d1 - d0) * 40))
gw = int(round(0.2 / 1.6 * 40))
seed[:, gj - gw: gj + gw] = 0.98
seed[gj - gw: gj + gw, gj - gw:] = 0.98
rho = seed.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history = []
for it in range(44):
    BETA = [8, 16, 32, 64][min(it // 11, 3)]
    adam.zero_grad()
    T = objective(rho.clamp(0.0, 1.0))
    (-T).backward()
    adam.step()
    history.append(float(T))
    if it % 5 == 0 or it == 43:
        print(f"optimize_ceviche_corner  it={it} beta={BETA}: "
              f"|S21|^2 = {float(T) * 100:.1f}%")

BETA = 64
rho_f = rho.detach().clamp(0.0, 1.0)
r = tanh_projection(conic_filter(rho_f, 0.10, dx), BETA)
eps_fin = region.paint(base, r).detach()
with torch.no_grad():
    s21, s11 = s_params(eps_fin)
    coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(T_RUN / grid.dt)), source=src,
                       monitors=[mon])
ez = np.abs((acc[0, 0] + 1j * acc[0, 1]).numpy())
T_fin, R_fin = float(abs(s21) ** 2), float(abs(s11) ** 2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
ax1.plot(np.array(history) * 100, "bo-")
ax1.set_xlabel("iteration"); ax1.set_ylabel("|S21|^2  [%]")
ax1.set_title(f"corner bend: T={T_fin * 100:.1f}%  R={R_fin * 100:.1f}%")
ax2.imshow(eps_fin.numpy().T, origin="lower", cmap="viridis",
           extent=[0, 6, 0, 6])
ax2.set_title("final design (eps)")
ax3.imshow(ez.T, origin="lower", cmap="inferno", extent=[0, 6, 0, 6])
ax3.set_title("|Ez| through the corner")
fig.tight_layout()
fig.savefig(OUT / "optimize_ceviche_corner.png", dpi=150)
plt.close(fig)
print(f"optimize_ceviche_corner  final |S21|^2 = {T_fin * 100:.1f}%, "
      f"|S11|^2 = {R_fin * 100:.1f}%; png written")
