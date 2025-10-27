# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2): inverse-designed Y splitter on the REAL SiN platform.

Geometry imported from gdsfactory: mmi1x2_with_sbend at 2.0 um width,
rasterized through the gds adapter and used BOTH as the simulation base
and as the inverse-design seed (the design region covers the MMI body;
rho is seeded from the imported layout). Platform: SiN 150 nm on SiO2,
lateral effective index n_core = neff(150 nm slab TE0), indices from
the meep materials library (Si3N4 Luke, SiO2 Horiba) at 1550 nm. Objective = |S21|^2 +
|S31|^2 via TE0 mode source + per-arm mode overlap.

Run: uv run python examples_2/c_y_splitter.py
"""
import math
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(16)   # two heavy jobs share the 32-core box
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.adjoint import DesignRegion, MaterialGrid, rho_from_eps
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
gf.gpdk.PDK.activate()

# --- real platform: 150 nm SiN slab -> lateral effective index ---
from meep_torch.core.materials import Si3N4, SiO2

# refractive indices from the meep materials library (Lorentz fits):
# Si3N4 K. Luke et al. (2015), SiO2 Horiba TN08 -- evaluated at 1550 nm
N_SIN = Si3N4.n(1 / 1.55).real                      # 1.9963
N_CLAD = SiO2.n(1 / 1.55).real                      # 1.4577
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
W_WG = 2.0
print(f"c_y_splitter  SiN 150nm/SiO2: lateral n_core = {NEFF:.4f}, "
      f"n_clad = {N_CLAD}, W = {W_WG} um")

# --- gdsfactory Y-branch (MMI 1x2 + sbends) at 2 um, output straights
# extended so the arms run into the PML ---
xs = gf.get_cross_section("strip").copy(width=W_WG)
mmi = gf.components.mmi1x2_with_sbend(cross_section=xs)
comp = gf.Component("y_splitter_2um")
ref = comp.add_ref(mmi)
for pname in ("o1", "o2", "o3"):
    st = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
    st.connect("o1", ref.ports[pname])
    comp.add_port(pname, port=st.ports["o2"])
comp.flatten()

RES = 20
FCEN = 1 / 1.55
dx = 1.0 / RES
grid, eps_base, ports = component_to_eps(comp, res=RES, pad=2.2,
                                         n_core=NEFF, n_clad=N_CLAD)
nx, ny = grid.nx, grid.ny
NPML = RES                                          # 1 um
T_RUN = 130.0
print(f"c_y_splitter  grid {nx}x{ny} ({nx * dx:.1f} x {ny * dx:.1f} um)")

o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
o3x, o3y, _ = ports["o3"]

# design region: the MMI body + junction (component x ~ 0.5..11 um)
x_shift = o1x * dx + 4.0 + 0.25                     # component x=0 in grid um
ix0 = int((x_shift + 0.5) * RES)
ix1 = int((x_shift + 11.0) * RES)
iy0 = int(o1y - 3.5 * RES)
iy1 = int(o1y + 3.5 * RES)
NRX, NRY = 105, 70                                  # 100 nm design pixels
mg = MaterialGrid((NRX, NRY), eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)

# rho seed = the imported layout inside the design box (core API)
seed = rho_from_eps(eps_base, region)

SPAN = int(1.8 * RES)
p_in = Port(axis="x", index=max(o1x + 6, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
p_up = Port(axis="x", index=min(o2x - 6, nx - NPML - 4), lo=o2y - SPAN,
            hi=o2y + SPAN, direction=-1)
p_dn = Port(axis="x", index=min(o3x - 6, nx - NPML - 4), lo=o3y - SPAN,
            hi=o3y + SPAN, direction=-1)
validate_ports([p_in, p_up, p_dn], grid, NPML)
pulse = GaussianPulse(FCEN, 0.2)                    # t_end = 50 < T_RUN


def s_params(eps):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                          lo=p.lo, hi=p.hi) for p in (p_in, p_up, p_dn)]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=400)
    out = []
    for p, acc in zip((p_in, p_up, p_dn), accs):
        el = eps.detach()[p.index, p.lo:p.hi]
        out.append(decompose(acc, el, dx, [FCEN], "tm", 1, axis_sign=-1.0))
    a_in = out[0][0][0, 0]
    return (out[1][0][0, 0] / a_in, out[2][0][0, 0] / a_in,
            out[0][1][0, 0] / a_in)


def objective(rho):
    r = tanh_projection(conic_filter(rho, 0.15, dx), BETA)
    s_up, s_dn, _ = s_params(region.paint(eps_base, r))
    return (s_up.conj() * s_up + s_dn.conj() * s_dn).real


rho = seed.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history = []
for it in range(30):
    BETA = [8, 16, 32, 64][min(it // 8, 3)]
    adam.zero_grad()
    J = objective(rho.clamp(0.0, 1.0))
    (-J).backward()
    adam.step()
    history.append(float(J))
    if it % 5 == 0 or it == 29:
        print(f"c_y_splitter  it={it} beta={BETA}: total T = "
              f"{float(J) * 100:.1f}%")

BETA = 64
rho_f = rho.detach().clamp(0, 1)
# mirror-symmetrize (device is nominally symmetric; averages out noise)
rho_f = 0.5 * (rho_f + rho_f.flip(1))
r = tanh_projection(conic_filter(rho_f, 0.15, dx), BETA)
eps_fin = region.paint(eps_base, r).detach()
with torch.no_grad():
    s_up, s_dn, s11 = s_params(eps_fin)
    coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(T_RUN / grid.dt)), source=src,
                       monitors=[mon], checkpoint_every=400)
ez = np.abs((acc[0, 0] + 1j * acc[0, 1]).numpy())
T_up, T_dn = float(abs(s_up) ** 2), float(abs(s_dn) ** 2)
tot = T_up + T_dn
il_db = -10 * math.log10(tot)
imb = abs(T_up - T_dn) / tot
np.save(OUT / "c_y_splitter_rho.npy", rho_f.numpy())
print(f"c_y_splitter  |S21|^2={T_up * 100:.1f}% |S31|^2={T_dn * 100:.1f}% "
      f"IL {il_db:.2f} dB, imbalance {imb * 100:.2f}%, "
      f"|S11|^2={float(abs(s11) ** 2) * 100:.2f}%")

fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
ax1, ax2, ax3, ax4 = axs.flat
ax1.plot(np.array(history) * 100, "bo-")
ax1.set_xlabel("iteration"); ax1.set_ylabel(r"$|S_{21}|^2 + |S_{31}|^2$ [%]")
ax1.set_title(f"IL {il_db:.2f} dB, imbalance {imb * 100:.2f}%")
ax2.imshow(eps_base.numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax2.set_title("gdsfactory mmi1x2_with_sbend seed (W = 2 um)")
ax3.imshow(eps_fin.numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax3.set_title("optimized design (SiN 150 nm eff. index)")
ax4.imshow(ez.T, origin="lower", cmap="inferno", extent=[0, nx * dx, 0, ny * dx])
ax4.set_title(r"$|E_z|$ at 1550 nm")
for ax in (ax2, ax3, ax4):
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter.png", dpi=150)
plt.close(fig)
print("c_y_splitter.png written")
