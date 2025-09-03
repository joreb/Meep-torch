# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2): inverse-designed Y splitter, SiN 150 nm / SiO2
(tidy3d Autograd-splitter-notebook style, meep-torch adjoint toolchain).

Redone at the fdtdx quality bar: objective = |S21|^2 + |S31|^2 via the
ports mode-overlap machinery (TE0 mode source in, TE0 overlap at each
arm) -- true mode-to-mode transmission, reported in dB. Design half-grid
mirrored about the axis (50/50 by construction); RES 25, rho 40x24.

Run: uv run python examples_2/c_y_splitter.py
"""
import math
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
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
eps_line = np.full(400, 1.44 ** 2)
eps_line[200 - 7: 200 + 8] = 2.0 ** 2
NEFF_SLAB = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
# 2D nominal-index convention (same as case a): the 150 nm-slab effective
# index is only 0.007 above the SiO2 cladding -- a 2D wire at that contrast
# holds no bound mode, so the 2D stand-in uses bulk SiN n = 2.0.
NEFF = 2.0
print(f"c_y_splitter  150nm slab TE0 neff = {NEFF_SLAB:.4f}; "
      f"2D uses n_core = {NEFF}")

RES = 25
FCEN = 1 / 1.55
dx = 1.0 / RES
nx, ny = int(12.0 * RES), int(8.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
cy = ny // 2
jw = int(0.25 * RES)                              # 0.5 um waveguides
SEP = int(1.2 * RES)                              # arm offset from axis
NPML = int(0.8 * RES)
T_RUN = 80.0

base = torch.full((nx, ny), 1.44 ** 2, dtype=torch.float64)
i_in1 = int(3.0 * RES)
i_out0 = int(9.0 * RES)
base[:i_in1, cy - jw: cy + jw] = NEFF ** 2
base[i_out0:, cy + SEP - jw: cy + SEP + jw] = NEFF ** 2
base[i_out0:, cy - SEP - jw: cy - SEP + jw] = NEFF ** 2

mg = MaterialGrid((40, 24), eps_min=1.44 ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=i_in1, ix1=i_out0,
                      iy0=cy - int(2.0 * RES), iy1=cy + int(2.0 * RES))

SPAN = int(0.9 * RES)
p_in = Port(axis="x", index=int(1.4 * RES), lo=cy - SPAN, hi=cy + SPAN,
            direction=+1)
p_up = Port(axis="x", index=nx - int(1.4 * RES), lo=cy + SEP - SPAN,
            hi=cy + SEP + SPAN, direction=-1)
p_dn = Port(axis="x", index=nx - int(1.4 * RES), lo=cy - SEP - SPAN,
            hi=cy - SEP + SPAN, direction=-1)
validate_ports([p_in, p_up, p_dn], grid, NPML)
pulse = GaussianPulse(FCEN, 0.2)                    # t_end=50 < T_RUN


def s_params(eps):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                          lo=p.lo, hi=p.hi) for p in (p_in, p_up, p_dn)]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=250)
    out = []
    for p, acc in zip((p_in, p_up, p_dn), accs):
        el = eps.detach()[p.index, p.lo:p.hi]
        out.append(decompose(acc, el, dx, [FCEN], "tm", 1, axis_sign=-1.0))
    a_in = out[0][0][0, 0]
    return (out[1][0][0, 0] / a_in, out[2][0][0, 0] / a_in,
            out[0][1][0, 0] / a_in)                # s_up, s_dn, s11


def objective(rho_half):
    rho = torch.cat([rho_half, rho_half.flip(1)], dim=1)   # mirror symmetry
    r = tanh_projection(conic_filter(rho, 0.12, dx), BETA)
    s_up, s_dn, _ = s_params(region.paint(base, r))
    return (s_up.conj() * s_up + s_dn.conj() * s_dn).real


# half-seed: S-bend band from the input axis (j=12, top edge of the half)
# to the lower arm center, so the optimizer starts from a working Y-branch
seed = torch.full((40, 12), 0.1, dtype=torch.float64)
j_arm = (2.0 - 1.2) / 4.0 * 24                     # arm center in rho rows
for i in range(40):
    jc = 12 - (12 - j_arm) * (i / 39) ** 1.3
    j0, j1 = int(round(jc)) - 2, int(round(jc)) + 2
    seed[i, max(j0, 0):min(j1, 12)] = 0.9
rho = seed.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.04)
history = []
for it in range(36):
    BETA = [8, 16, 32, 64][min(it // 9, 3)]
    adam.zero_grad()
    J = objective(rho.clamp(0.0, 1.0))
    (-J).backward()
    adam.step()
    history.append(float(J))
    if it % 6 == 0 or it == 35:
        print(f"c_y_splitter  it={it} beta={BETA}: total T = "
              f"{float(J) * 100:.1f}%")

BETA = 64
rho_full = torch.cat([rho.detach().clamp(0, 1),
                      rho.detach().clamp(0, 1).flip(1)], dim=1)
r = tanh_projection(conic_filter(rho_full, 0.12, dx), BETA)
eps_fin = region.paint(base, r).detach()
with torch.no_grad():
    s_up, s_dn, s11 = s_params(eps_fin)
    coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(T_RUN / grid.dt)), source=src,
                       monitors=[mon])
ez = np.abs((acc[0, 0] + 1j * acc[0, 1]).numpy())
T_up, T_dn = float(abs(s_up) ** 2), float(abs(s_dn) ** 2)
tot = T_up + T_dn
il_db = -10 * math.log10(tot)
imb = abs(T_up - T_dn) / tot
np.save(OUT / "c_y_splitter_rho.npy", rho_full.numpy())
print(f"c_y_splitter  |S21|^2={T_up * 100:.1f}% |S31|^2={T_dn * 100:.1f}% "
      f"insertion loss {il_db:.2f} dB, imbalance {imb * 100:.2f}%, "
      f"|S11|^2={float(abs(s11) ** 2) * 100:.2f}%")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ax1.plot(np.array(history) * 100, "bo-")
ax1.set_xlabel("iteration"); ax1.set_ylabel(r"$|S_{21}|^2 + |S_{31}|^2$  [%]")
ax1.set_title(f"Y splitter: IL {il_db:.2f} dB, imbalance {imb * 100:.2f}%")
ax2.imshow(eps_fin.numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax2.set_title("final design (mirror-symmetric)")
ax3.imshow(ez.T, origin="lower", cmap="inferno", extent=[0, nx * dx, 0, ny * dx])
ax3.set_title(r"$|E_z|$ at 1550 nm")
for ax in (ax2, ax3):
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter.png", dpi=150)
plt.close(fig)
print("c_y_splitter.png written")
