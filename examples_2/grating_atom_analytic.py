# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""STANDALONE analytically-designed focusing grating (atom target).

Closed-form chirped grating: tooth m sits where the guided phase plus
the free-space path to the atom is stationary modulo 2*pi,
    neff*x_m + R(x_m) = C + m*lam,
R(x) = tooth-to-atom distance, neff from the same fd1d port-mode solve
the FDFD source uses. 50% local duty. The design is evaluated in FDFD
(no optimization here) with its own plot; the binary tooth pattern is
saved to grating_atom_analytic_seed.npy and consumed as the SEED by
grating_atom_focus.py (analytic -> 2D optimize -> 3D pipeline).

Run: uv run python examples_2/grating_atom_analytic.py
"""
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.fdfd import make_mode_source, port_mode_line, solve_fdfd_tm
from meep_torch.ports.port import Port

WL = 1.092
FCEN = 1 / WL
N_SIN = float(Si3N4.n(FCEN).real)
N_SIO = float(SiO2.n(FCEN).real)
RES = 30
dx = 1.0 / RES
NPML = RES
H_FILM, L_DEV, X0_DEV, Z_SUB = 0.52, 18.0, 4.0, 2.5
LX, LZ = 25.0, 8.0
nx, nz = int(LX * RES), int(LZ * RES)
k0f, k1f = int(Z_SUB * RES), int(Z_SUB * RES) + int(H_FILM * RES)

base = torch.ones(nx, nz, dtype=torch.float64)
base[:, :k0f] = N_SIO ** 2
base[:, k0f:k1f] = N_SIN ** 2

ia_x = int((X0_DEV + L_DEV - 1.4) * RES)
ia_z = int((Z_SUB + H_FILM + 3.5) * RES)
SPAN = int(1.5 * RES)
zc = (k0f + k1f) // 2
p_in = Port(axis="x", index=NPML + 8, lo=zc - SPAN, hi=zc + SPAN, direction=+1)

_, neff = port_mode_line(base, p_in, FCEN, dx)
xa, za = ia_x * dx, (ia_z - k1f) * dx


def phase(x):
    return neff * x + math.hypot(xa - x, za)


C0 = phase(X0_DEV)
teeth, m = [], 1
while True:
    try:
        xm = brentq(lambda x: phase(x) - C0 - m * WL, X0_DEV - 0.5,
                    X0_DEV + L_DEV + 2.0)
    except ValueError:
        break
    if xm > X0_DEV + L_DEV:
        break
    teeth.append(xm)
    m += 1
lam_loc = np.diff([X0_DEV] + teeth)
print(f"grating_atom_analytic  neff={neff:.4f}, {len(teeth)} teeth, "
      f"local period {lam_loc.min():.2f}..{lam_loc.max():.2f} um")

NRX = int(L_DEV / 0.1)
NRZ = max(int(H_FILM / 0.1), 3)
xs_r = X0_DEV + (np.arange(NRX) + 0.5) * 0.1
# apodized duty: weak out-coupling first so power survives to the teeth
# nearest the atom (constant 50% duty dumped everything in the first few
# teeth -- measured 0.001 atom/guide-peak). Linear ramp 12% -> 55%.
seed = np.zeros(NRX)
nT = len(teeth)
for i, xm in enumerate(teeth):
    duty = 0.12 + (0.55 - 0.12) * i / max(nT - 1, 1)
    w_loc = duty * (lam_loc[i] if i < len(lam_loc) else lam_loc[-1])
    seed[(xs_r > xm - w_loc / 2) & (xs_r < xm + w_loc / 2)] = 1.0
# the OPTIMIZER seed is the phase-only 50%-duty variant: measured basins
# -- optimizing from the apodized seed stalls at 0.22, from the dense
# phase-only pattern it reaches 0.48 (the optimizer carves its own
# apodization better than the linear ramp).
seed_opt = np.zeros(NRX)
for i, xm in enumerate(teeth):
    w_loc = 0.5 * (lam_loc[i] if i < len(lam_loc) else lam_loc[-1])
    seed_opt[(xs_r > xm - w_loc / 2) & (xs_r < xm + w_loc / 2)] = 1.0
np.save(OUT / "grating_atom_analytic_seed.npy", seed_opt)

# evaluate the analytic design in FDFD (its own standalone result)
mg = MaterialGrid((NRX, NRZ), eps_min=1.0, eps_max=N_SIN ** 2)
region = DesignRegion(mg, ix0=int(X0_DEV * RES), ix1=int((X0_DEV + L_DEV) * RES),
                      iy0=k0f, iy1=k1f)
# PARTIAL ETCH: full-etch air teeth reflect so hard the mode dies at
# the first tooth (measured: field empty past x=5, gain 0.004). Etch
# only the top 40% of the film -- the standard fab answer.
rho = torch.ones(NRX, NRZ, dtype=torch.float64)
n_etch = max(int(0.4 * NRZ), 1)
rho[:, NRZ - n_etch:] = 1.0 - torch.from_numpy(seed)[:, None].repeat(1, n_etch)
rho = 1.0 - rho if False else rho  # teeth=etched: seed==1 -> air in top rows
rho[:, NRZ - n_etch:] = torch.where(
    torch.from_numpy(seed)[:, None].repeat(1, n_etch) > 0.5, 0.0, 1.0)
eps_dev = region.paint(base, rho)
jz = make_mode_source(base, p_in, FCEN, dx)
E_ref = solve_fdfd_tm(base, dx, FCEN, jz, NPML, backend="pardiso")
icol = int((X0_DEV + 2.0) * RES)
I_ref = float((E_ref[icol, k0f:k1f].abs() ** 2).max())
E = solve_fdfd_tm(eps_dev, dx, FCEN, jz, NPML, backend="pardiso")
gain = float((E[ia_x, ia_z].conj() * E[ia_x, ia_z]).real) / I_ref
print(f"grating_atom_analytic  atom/guide-peak = {gain:.3f} "
      f"(analytic design, no optimization)")

fig, axs = plt.subplots(2, 2, figsize=(13, 7))
a1, a2, a3, a4 = axs.flat
a1.step(xs_r, seed, where="mid")
a1.set_xlabel("x [um]"); a1.set_ylabel("tooth")
a1.set_title(f"analytic chirp: {len(teeth)} teeth, "
             f"period {lam_loc.min():.2f}->{lam_loc.max():.2f} um")
a2.imshow(eps_dev.numpy().T, origin="lower", cmap="viridis",
          extent=[0, LX, 0, LZ], aspect="auto")
a2.plot([ia_x * dx], [ia_z * dx], "r*", ms=12)
a2.set_title(f"analytic design (star=atom), neff={neff:.3f}")
a3.imshow(E.abs().numpy().T, origin="lower", cmap="inferno",
          extent=[0, LX, 0, LZ], aspect="auto")
a3.plot([ia_x * dx], [ia_z * dx], "c*", ms=12)
a3.set_title(f"|Ez| FDFD -- atom/guide-peak = {gain:.2f}")
prof = (E.abs() ** 2)[ia_x, :].numpy()
a4.plot(np.arange(nz) * dx, prof / max(prof.max(), 1e-30))
a4.axvline(ia_z * dx, color="r", lw=0.8)
a4.set_xlabel("z [um]"); a4.set_title("vertical cut through atom column")
for a in (a3,):
    a.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "grating_atom_analytic.png", dpi=150)
print("grating_atom_analytic.png written")
