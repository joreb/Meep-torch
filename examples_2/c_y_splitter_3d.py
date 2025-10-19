# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2b): TRUE-3D validation of the inverse-designed Y splitter.

Rebuilds the optimized 2D design (c_y_splitter_rho.npy), extrudes it to
the real stack -- SiN n = 1.9963 (Luke), 150 nm thick, SiO2 cladding, NO effective
index -- and runs full 3D FDTD (stepper_3d): quasi-TE fundamental mode
launched from the true 3D input cross-section, per-arm guided power vs
a straight-waveguide 3D reference. fdtdx-style two-run S-parameters.

Run after c_y_splitter.py:  uv run python examples_2/c_y_splitter_3d.py
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
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.monitors_3d import FluxMonitor3D, run_3d
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.validate3d import ModePlaneSourceX, extrude

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

# --- rebuild the optimized 2D eps exactly as c_y_splitter.py ---
from meep_torch.core.materials import Si3N4, SiO2

# refractive indices from the meep materials library (Lorentz fits):
# Si3N4 K. Luke et al. (2015), SiO2 Horiba TN08 -- evaluated at 1550 nm
N_SIN = Si3N4.n(1 / 1.55).real                      # 1.9963
N_CLAD = SiO2.n(1 / 1.55).real                      # 1.4577
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
W_WG = 2.0
xs = gf.get_cross_section("strip").copy(width=W_WG)
mmi = gf.components.mmi1x2_with_sbend(cross_section=xs)
comp = gf.Component("y_splitter_2um")
ref = comp.add_ref(mmi)
for pname in ("o1", "o2", "o3"):
    st = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
    st.connect("o1", ref.ports[pname])
    comp.add_port(pname, port=st.ports["o2"])
comp.flatten()
RES2 = 20
grid2, eps_base, ports = component_to_eps(comp, res=RES2, pad=2.2,
                                          n_core=NEFF, n_clad=N_CLAD)
o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
o3x, o3y, _ = ports["o3"]
x_shift = o1x / RES2 + 4.25
ix0, ix1 = int((x_shift + 0.5) * RES2), int((x_shift + 11.0) * RES2)
iy0, iy1 = int(o1y - 3.5 * RES2), int(o1y + 3.5 * RES2)
mg = MaterialGrid((105, 70), eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
rho = torch.from_numpy(np.load(OUT / "c_y_splitter_rho.npy"))
r = tanh_projection(conic_filter(rho, 0.15, 1 / RES2), 64.0)
eps2d = region.paint(eps_base, r).detach()

# straight-guide 2D reference (input bus continued straight)
eps2d_ref = torch.full_like(eps_base, N_CLAD ** 2)
eps2d_ref[:, o1y - int(1.0 * RES2): o1y + int(1.0 * RES2)] = NEFF ** 2

# --- 3D: downsample to RES3, extrude with REAL indices ---
RES3 = 15
H_CORE, PAD_Z = 0.15, 1.5      # N_SIN from the materials library above
FCEN = 1 / 1.55
dx3 = 1.0 / RES3


def to3d(e2):
    n3x = int(e2.shape[0] * RES3 / RES2)
    n3y = int(e2.shape[1] * RES3 / RES2)
    ds = F.interpolate(e2[None, None], size=(n3x, n3y),
                       mode="bilinear", align_corners=False)[0, 0]
    return extrude(ds, n_clad=N_CLAD, n_core=N_SIN, h_core=H_CORE,
                   pad_z=PAD_Z, res=RES3)


eps3 = to3d(eps2d)
eps3_ref = to3d(eps2d_ref)
nx, ny, nz = eps3.shape
NPML = RES3
print(f"c_y_splitter_3d  grid {nx}x{ny}x{nz} "
      f"({nx * dx3:.1f} x {ny * dx3:.1f} x {nz * dx3:.1f} um), "
      f"SiN n={N_SIN} h={H_CORE} um")

PULSE = GaussianPulse(FCEN, 0.2)
IX_SRC = max(int(o1x * RES3 / RES2) + 6, NPML + 4)
T_RUN = 140.0

y2 = lambda j2: int(j2 * RES3 / RES2)
ARM_SPAN = int(1.8 * RES3)
IX_OUT = min(int(o2x * RES3 / RES2) - 6, nx - NPML - 4)


def run_case(e3, wins):
    c = build_3d_coeffs(nx, ny, nz, dx3, eps=e3, npml=NPML)
    src = ModePlaneSourceX(e3, dx3, IX_SRC, FCEN, PULSE, c.dt)
    mons = [FluxMonitor3D(c.dt, [FCEN], normal="x", idx=IX_OUT,
                          win=(w0, w1, NPML, nz - NPML)) for (w0, w1) in wins]
    n_steps = int(round(T_RUN / c.dt))
    _, accs = run_3d(init_3d_state(nx, ny, nz, npml=NPML), c, n_steps,
                     source=src, monitors=mons)
    return [float(m.flux(a, dx3)[0]) for m, a in zip(mons, accs)], src.neff


wins_dev = [(y2(o2y) - ARM_SPAN, y2(o2y) + ARM_SPAN),
            (y2(o3y) - ARM_SPAN, y2(o3y) + ARM_SPAN)]
wins_ref = [(y2(o1y) - ARM_SPAN, y2(o1y) + ARM_SPAN)]
(p_ref,), neff_in = run_case(eps3_ref, wins_ref)
(p_up, p_dn), _ = run_case(eps3, wins_dev)
T_up, T_dn = p_up / p_ref, p_dn / p_ref
tot = T_up + T_dn
print(f"c_y_splitter_3d  input TE0 neff = {neff_in.real:.4f}")
print(f"c_y_splitter_3d  3D: |S21|^2={T_up * 100:.1f}% |S31|^2={T_dn * 100:.1f}% "
      f"total {tot * 100:.1f}% (IL {-10 * math.log10(max(tot, 1e-12)):.2f} dB), "
      f"imbalance {abs(T_up - T_dn) / max(tot, 1e-12) * 100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
kmid = nz // 2
ax1.imshow(eps3[:, :, kmid].numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx3, 0, ny * dx3])
ax1.set_title("3D stack, core plane (SiN n=2.0, 150 nm)")
ax1.set_xlabel("x [um]")
ax2.bar(["S21 (up)", "S31 (dn)"], [T_up * 100, T_dn * 100], color="tab:blue")
ax2.set_ylabel("power [% of straight-guide ref]")
ax2.set_title(f"true-3D validation: total {tot * 100:.1f}%")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter_3d.png", dpi=150)
plt.close(fig)
print("c_y_splitter_3d.png written")
