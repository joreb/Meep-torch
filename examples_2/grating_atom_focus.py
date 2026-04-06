# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Grating coupler focusing onto an atom -- meep-torch port of
examples_2/grating_coupler_4.py (fdtdx, AlN) with SiN, per user spec.

Geometry (2D vertical cross-section x-z, effective 2D of the original
3D scene): SiO2 substrate, 520 nm SiN film, air above. A mode source
drives the slab guide from the left; an 18 um design region in the film
is free-form-optimized so the scattered light FOCUSES on an "atom" --
a point 3.5 um above the film, 1.4 um inward from the device end (the
original's 45-degree target at 2e-6*cos45/5e-6*sin45).

Design: FDFD exact adjoint (pardiso backend, ~1 s/solve), objective =
|Ez(atom)|^2 normalized by a bare-slab reference solve -- reported as
INTENSITY GAIN over the unpatterned slab, the honest focusing metric.
Final design cross-checked in FDTD (run_tm + DFT field at the atom).

lam = 1092 nm (kept from the original -- an ion transition line);
SiN n from the materials library at that wavelength.

Run: uv run python examples_2/grating_atom_focus.py
"""
import math
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors import DFTFieldsMonitor
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.fdfd import make_mode_source, solve_fdfd_tm
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

WL = 1.092
FCEN = 1 / WL
N_SIN = float(Si3N4.n(FCEN).real)
N_SIO = float(SiO2.n(FCEN).real)
RES = 30
dx = 1.0 / RES
NPML = RES
H_FILM = 0.52
L_DEV = 18.0
X0_DEV = 4.0                      # design starts here (after input slab)
LX, LZ = 25.0, 8.0
Z_SUB = 2.5                       # substrate top (film sits on it)
nx, nz = int(LX * RES), int(LZ * RES)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"grating_atom_focus  SiN n={N_SIN:.4f} / SiO2 n={N_SIO:.4f} @1092nm, "
      f"grid {nx}x{nz}")

k_film0 = int(Z_SUB * RES)
k_film1 = k_film0 + int(H_FILM * RES)
base = torch.ones(nx, nz, dtype=torch.float64)            # air
base[:, :k_film0] = N_SIO ** 2                            # substrate
base[:, k_film0:k_film1] = N_SIN ** 2                     # full film seed

# design region: the film layer over the grating length, free-form
ix0, ix1 = int(X0_DEV * RES), int((X0_DEV + L_DEV) * RES)
NRX = int(L_DEV / 0.1)                                    # 100 nm voxels
NRZ = max(int(H_FILM / 0.1), 3)
mg = MaterialGrid((NRX, NRZ), eps_min=1.0, eps_max=N_SIN ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=k_film0, iy1=k_film1)

# atom: 1.4 um inward from the device end, 3.5 um above the film top
ia_x = int((X0_DEV + L_DEV - 1.4) * RES)
ia_z = int((Z_SUB + H_FILM + 3.5) * RES)

SPAN = int(1.5 * RES)
zc = (k_film0 + k_film1) // 2
p_in = Port(axis="x", index=NPML + 8, lo=zc - SPAN, hi=zc + SPAN, direction=+1)
jz = make_mode_source(base, p_in, FCEN, dx)

# reference: PEAK GUIDED intensity in the bare slab (mid-guide column).
# (v1 normalized by the bare-slab intensity AT THE ATOM -- numerically
# zero 3.5 um above an unpatterned slab, which exploded the "gain" to
# 1e11x. Guide-peak normalization is O(1)-meaningful: gain > 1 means
# the focus is brighter than the guided mode itself.)
E_ref = solve_fdfd_tm(base, dx, FCEN, jz, NPML, backend="pardiso")
icol = int((X0_DEV + 2.0) * RES)
I_ref = float((E_ref[icol, k_film0:k_film1].abs() ** 2).max())


def atom_I(eps):
    E = solve_fdfd_tm(eps, dx, FCEN, jz, NPML, backend="pardiso")
    a = E[ia_x, ia_z]
    return (a.conj() * a).real, E


def J_of(rho, beta):
    r = tanh_projection(conic_filter(rho, 0.12, dx), beta)
    I, _ = atom_I(region.paint(base, r))
    return I / I_ref


# --- seed: the ANALYTIC design from grating_atom_analytic.py ---
# (standalone script with its own plot; run it first). Softened 0.3/0.7
# so the projection has gradients (binary seeds stall -- measured).
seed_analytic = torch.from_numpy(
    np.load(OUT / "grating_atom_analytic_seed.npy"))
rho = (0.3 + 0.4 * seed_analytic)[:, None].repeat(1, NRZ).clone().requires_grad_(True)
with torch.no_grad():
    J_seed = float(J_of(seed_analytic[:, None].repeat(1, NRZ), 64.0))
print(f"grating_atom_focus  analytic seed (binary): atom/guide-peak = {J_seed:.3f}")
adam = torch.optim.Adam([rho], lr=0.05)
history, post64 = [], []
best = (-1.0, None)
t0 = time.time()
for it in range(48):
    BETA = [8, 16, 32, 64][min(it // 12, 3)]
    adam.zero_grad()
    J = J_of(rho.clamp(0, 1), BETA)
    (-J).backward()
    adam.step()
    history.append(float(J))
    with torch.no_grad():
        Jp = float(J_of(rho.clamp(0, 1), 64.0))
    post64.append(Jp)
    if Jp > best[0]:
        best = (Jp, rho.detach().clamp(0, 1).clone())
    print(f"grating_atom_focus  it={it} beta={BETA}: gain={float(J):.1f}x "
          f"post64={Jp:.1f}x")

rho_f = best[1]
r = tanh_projection(conic_filter(rho_f, 0.12, dx), 64.0)
eps_fin = region.paint(base, r).detach()
with torch.no_grad():
    I_fin, E_fin = atom_I(eps_fin)
gain = float(I_fin) / I_ref
np.save(OUT / "grating_atom_focus_rho.npy", rho_f.numpy())
print(f"grating_atom_focus  BEST FDFD atom/guide-peak = {gain:.2f} "
      f"({time.time()-t0:.0f}s, {len(history)} iters)")

# --- FDTD cross-check of the final design ---
grid = Grid(nx=nx, ny=nz, dx=dx, device=DEV)
eps_dev = eps_fin.to(DEV)
coeffs = build_tm_coeffs(grid, eps_inf=eps_dev, npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, 0.2), FCEN)
mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
T_RUN = 2 * (5 / 0.2) + LX * 2.1 + 40
with torch.no_grad():
    _, (acc,) = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                       source=src, monitors=[mon], checkpoint_every=400)
acc = acc.cpu()
ez_t = (acc[0, 0] + 1j * acc[0, 1])
# same for bare slab
coeffs_r = build_tm_coeffs(grid, eps_inf=base.to(DEV), npml=NPML)
src_r = ModeSourceTM(grid, coeffs_r, p_in, GaussianPulse(FCEN, 0.2), FCEN)
mon_r = DFTFieldsMonitor(grid, [FCEN], "Ez")
with torch.no_grad():
    _, (acc_r,) = run_tm(init_tm_state(grid), coeffs_r,
                         int(round(T_RUN / grid.dt)), source=src_r,
                         monitors=[mon_r], checkpoint_every=400)
acc_r = acc_r.cpu()
ez_r = (acc_r[0, 0] + 1j * acc_r[0, 1])
gain_t = float(abs(ez_t[ia_x, ia_z]) ** 2
               / (np.abs(ez_r.numpy()[icol, k_film0:k_film1]) ** 2).max())
rel = abs(gain - gain_t) / max(gain_t, 1e-12)
print(f"grating_atom_focus  FDTD cross-check gain = {gain_t:.1f}x "
      f"(FDFD {gain:.1f}x, rel diff {rel*100:.1f}%)")

fig, axs = plt.subplots(2, 2, figsize=(13, 7))
a1, a2, a3, a4 = axs.flat
a1.plot(post64, "bo-", label="post-step @64")
a1.plot(history, "c.--", lw=0.8, label="pre-step")
a1.set_xlabel("iteration"); a1.set_ylabel("atom intensity / guide peak")
a1.set_title(f"atom/guide-peak: FDFD {gain:.2f} | FDTD {gain_t:.2f}")
a1.legend(fontsize=7)
a2.imshow(eps_fin.cpu().numpy().T, origin="lower", cmap="viridis",
          extent=[0, LX, 0, LZ], aspect="auto")
a2.plot([ia_x * dx], [ia_z * dx], "r*", ms=12)
a2.set_title("design (star = atom)")
a3.imshow(E_fin.abs().numpy().T, origin="lower", cmap="inferno",
          extent=[0, LX, 0, LZ], aspect="auto")
a3.plot([ia_x * dx], [ia_z * dx], "c*", ms=12)
a3.set_title("|Ez| FDFD")
a4.imshow(np.abs(ez_t.numpy()).T, origin="lower", cmap="inferno",
          extent=[0, LX, 0, LZ], aspect="auto")
a4.plot([ia_x * dx], [ia_z * dx], "c*", ms=12)
a4.set_title("|Ez| FDTD cross-check")
for a in (a3, a4):
    a.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "grating_atom_focus.png", dpi=150)
print("grating_atom_focus.png written")
