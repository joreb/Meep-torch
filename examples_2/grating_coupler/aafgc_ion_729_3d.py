# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # AAFGC 3D validation (CE plane)
#
# 3D validation of the AAFGC rebuild at the paper's CE plane.
#
# The npj paper defines CE (coupling efficiency) as the power crossing a
# plane 5 um above the chip surface. The full 68 um ion path in 3D is a
# ~437M-cell grid (beyond the lean-stepper budget); the CE plane is the
# paper's own intermediate metric and fits comfortably. The 2D-optimized
# tooth pattern (`output/aafgc_ion_729_rho.npy`) is extruded across a 6 um
# strip; quasi-TE mode in, DFT of Ey on the CE plane, normalized by the same
# monitor in a bare-strip reference run. Also reported: the lateral
# (y) beam width at the CE plane -- the divergence the 2D design never
# saw.
#
# Run after aafgc_ion_729.py:
# `uv run python examples_2/grating_coupler/aafgc_ion_729_3d.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.monitors_3d import DFTFieldsMonitor3D, FluxMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean
from meep_torch.ports.validate3d import ModePlaneSourceX

# %% [markdown]
# ## Parameters and 2D permittivity rebuild
# Same stack constants as the 2D script; the saved rho is repainted onto a
# cropped 2D grid (the beam at the CE plane stays near the launch region).

# %%
WL = 0.729
FCEN = 1 / WL
N_SIN, N_SIO = 2.0, 1.44
RES2 = 20
H_FILM, OX_BOT, OX_TOP = 0.22, 3.0, 5.0
L_DEV, X0_DEV = 30.0, 6.0
H_CE = 5.0                                    # CE plane above chip surface
LX2 = 80.0                                    # 2D script domain width

# rebuild the 2D eps (same painting as aafgc_ion_729.py)
LX = 50.0                                     # cropped: beam at CE is near launch
LZ = OX_BOT + H_FILM + OX_TOP + H_CE + 2.0
nx2, nz2 = int(LX * RES2), int(LZ * RES2)
k0f = int(OX_BOT * RES2)
k1f = k0f + max(int(H_FILM * RES2), 4)
k_surf = int((OX_BOT + H_FILM + OX_TOP) * RES2)
base2 = torch.full((nx2, nz2), 1.0, dtype=torch.float64)
base2[:, :k_surf] = N_SIO ** 2
base2[:, k0f:k1f] = N_SIN ** 2
rho = torch.from_numpy(np.load(OUT / "aafgc_ion_729_rho.npy"))
mg = MaterialGrid(rho.shape, eps_min=N_SIO ** 2, eps_max=N_SIN ** 2)
region = DesignRegion(mg, ix0=int(X0_DEV * RES2), ix1=int((X0_DEV + L_DEV) * RES2),
                      iy0=k0f, iy1=k1f)
r = tanh_projection(conic_filter(rho, 0.12, 1 / RES2), 64.0)
eps2 = region.paint(base2, r).detach()

# %% [markdown]
# ## Extrude to 3D
# The 2D design becomes a 6 um-wide strip (2 um cladding pad each side); the
# background keeps the oxide/vacuum layering. float32 on GPU when available.

# %%
RES3 = 20
dx3 = 1.0 / RES3
W_STRIP, PAD_Y = 6.0, 2.0
NPML = RES3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT3 = torch.float32
nx, nz = nx2, nz2
ny = int((W_STRIP + 2 * PAD_Y) * RES3)
j0, j1 = int(PAD_Y * RES3), int((PAD_Y + W_STRIP) * RES3)


def to3d(e2):
    bg = torch.full((nz,), 1.0, dtype=torch.float64)
    bg[:k_surf] = N_SIO ** 2
    e3 = bg[None, None, :].repeat(nx, ny, 1).clone()
    e3[:, j0:j1, :] = e2[:, None, :]
    return e3.to(DEV, dtype=DT3)


eps3 = to3d(eps2)
eps3_ref = to3d(base2)
print(f"aafgc_ion_729_3d  grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.0f}M cells on {DEV}")

IX_SRC = NPML + 6
k_ce = int((OX_BOT + H_FILM + OX_TOP + H_CE) * RES3)
icol = int((X0_DEV - 2.0) * RES3)
T_RUN = 2 * (5 / 0.2) + (LX + LZ) * 1.6 + 50


# %% [markdown]
# ## Device and reference runs
# Two lean-stepper runs: the device with a z-normal flux monitor at the CE
# plane (+ an Ey DFT for the beam picture), and a bare-strip reference with an
# x-normal input flux monitor for normalization.

# %%
# CE via POYNTING flux (the |Ey|^2-sum proxy printed CE=1.44 -- over
# unity, the eta_abs estimator trap: field-square sums over differently
# oriented planes are not powers). Flux up through the CE plane over
# flux forward through the input plane of a bare-strip reference.
def run_case(e3, mon_specs):
    c = build_lean(nx, ny, nz, dx3, e3, NPML, dtype=DT3, device=DEV)
    src = ModePlaneSourceX(e3, dx3, IX_SRC, FCEN, GaussianPulse(FCEN, 0.2), c.dt)
    mons = [spec(c.dt) for spec in mon_specs]
    _, accs = run_3d_lean(c, int(round(T_RUN / c.dt)), source=src,
                          monitors=mons, compile=(DEV == "cuda"))
    if DEV == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    del c, src
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return mons, accs


mons_d, accs_d = run_case(eps3, [
    lambda dt: FluxMonitor3D(dt, [FCEN], normal="z", idx=k_ce,
                             win=(NPML, nx - NPML, NPML, ny - NPML)),
    lambda dt: DFTFieldsMonitor3D(dt, [FCEN], "Ey", plane=(2, k_ce)),
])
mons_r, accs_r = run_case(eps3_ref, [
    lambda dt: FluxMonitor3D(dt, [FCEN], normal="x", idx=icol,
                             win=(NPML, ny - NPML, NPML, nz - NPML)),
])
P_ce = float(mons_d[0].flux(accs_d[0], dx3)[0])
P_in = float(mons_r[0].flux(accs_r[0], dx3)[0])
CE = abs(P_ce) / abs(P_in)
E_ce = (accs_d[1][0, 0] + 1j * accs_d[1][0, 1]).cpu().squeeze()

# %% [markdown]
# ## Results and plot
# CE as a Poynting flux ratio, the lateral 1/e^2 beam width at the CE plane,
# and a two-panel figure: |Ey|^2 map on the CE plane + the lateral profile.

# %%
# lateral beam width at the CE plane (1/e^2 of the y-profile at beam max x)
I_xy = (E_ce.abs() ** 2).numpy()
ix_pk = int(I_xy.sum(axis=1).argmax())
prof_y = I_xy[ix_pk]
above = np.where(prof_y > prof_y.max() / np.e ** 2)[0]
w_y = (above.max() - above.min()) * dx3 if len(above) else float("nan")
print(f"aafgc_ion_729_3d  CE (Poynting) = {CE:.3f} "
      f"({10*np.log10(max(CE,1e-12)):.1f} dB), lateral 1/e^2 width at CE "
      f"plane = {w_y:.2f} um (2D FE proxy -0.4 dB; paper CE at this plane)")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.imshow(I_xy.T, origin="lower", cmap="inferno",
          extent=[0, nx * dx3, 0, ny * dx3], aspect="auto")
a1.set_title(f"|Ey|^2 on the CE plane (5 um above chip), CE={CE:.2f}")
a1.set_xlabel("x [um]"); a1.set_ylabel("y [um]")
a2.plot(np.arange(len(prof_y)) * dx3, prof_y / prof_y.max())
a2.set_xlabel("y [um]")
a2.set_title(f"lateral profile at x={ix_pk*dx3:.1f} um, width {w_y:.1f} um")
fig.tight_layout()
fig.savefig(OUT / "aafgc_ion_729_3d.png", dpi=150)
print("aafgc_ion_729_3d.png written")
