# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Dipole antenna above a PEC ground plane (1D)
#
# meep-torch port of meep/python/examples/antenna_pec_ground_plane_1D.py. 1D
# dipole antenna above a PEC ground plane: a thin (nearly-1D) 2D grid with an
# Ez point dipole at height h above a metallic (eps=1e7) ground-plane strip.
# The upward radiation reflects off the ground plane and interferes with the
# direct field, producing a standing-wave pattern in the time-averaged |Ez|^2
# profile — fringe spacing lambda/(2n). Output:
# `output/antenna_pec_ground_plane_1D.png`.
#
# The meep original computes the *far-field radiation pattern* via
# Brillouin-zone integration over wavevectors (k_point / near-to-far / DFT
# flux). Those features (Bloch boundaries, near2far, DFT flux) are UNSUPPORTED
# in meep-torch, so this port keeps only the near-field interference physics
# and plots |Ez|^2 vs height.
#
# meep -> meep-torch mapping:
# ```
#   mp.Vector3(0,0,size_z) 1D cell, res 50        ->  Grid(nx=50, ny=240, dx=0.05) (res 20)
#   mp.PML(1.0) on +z                            ->  npml=20 (1.0 um / dx 0.05)
#   mp.Low/Z/Metallic ground plane               ->  eps[:, 20:22] = 1e7 (PEC approx)
#   mp.GaussianSource(1/0.65, fwidth=0.1*fc) Ez  ->  GaussianPulse(fcen, fwidth) 1-cell LineSource
#   dipole at height 1.25 um above ground        ->  ix=25, iy=45
# ```
# Run: `uv run python examples/02_sources/antenna_pec_ground_plane_1D.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Geometry and source
# A thin lateral strip approximates 1D; the PEC strip sits just inside the
# bottom PML and the dipole 1.25 um above it.

# %%
# --- parameters (meep natural units, um) ---
RES = 20                       # pixels/um (reduced from 50 for runtime)
DX = 1.0 / RES
PML_UM = 1.0
BULK_UM = 10.0
N_BG = 1.2
H_UM = 1.25                    # antenna height above ground
WVL_UM = 0.65
FCEN = 1.0 / WVL_UM
FWIDTH = 0.1 * FCEN

NPML = int(round(PML_UM / DX))                    # 20
NX = 50                                            # thin lateral: NPML=20 each side -> 10 bulk
NY = 2 * NPML + int(round(BULK_UM / DX))           # 20 + 200 + 20 = 240

grid = Grid(nx=NX, ny=NY, dx=DX)
eps = torch.full((NX, NY), N_BG * N_BG, dtype=torch.float64, device=grid.device)
# PEC ground plane: a high-eps strip just inside the bottom PML (Ez ~ 0 mirror).
eps[:, NPML : NPML + 2] = 1e7

coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

# Ez point dipole at the lateral center, height H above the ground plane.
ix_src = NX // 2
iy_src = NPML + int(round(H_UM / DX))              # ground at row NPML -> +H
src = LineSource(grid, ix=ix_src, j0=iy_src, j1=iy_src + 1, profile=GaussianPulse(FCEN, FWIDTH))

# %% [markdown]
# ## Run
# Step the fields and accumulate |Ez|^2 over the tail of the run to get the
# time-averaged standing-wave profile.

# %%
state = init_tm_state(grid)
N_STEPS = int(round(90.0 / grid.dt))               # ~90 time units / dt(=0.025) = 3600
AVG_START = N_STEPS - 1500                          # accumulate |Ez|^2 over the tail
prof = torch.zeros(NY, dtype=torch.float64, device=grid.device)
for t in range(N_STEPS):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)
    if t >= AVG_START:
        prof = prof + state.Ez[ix_src, :] ** 2

# %% [markdown]
# ## Fringe analysis
# Peak spacing in the averaged |Ez|^2 profile should match the half-wavelength
# in the medium, lambda/(2n).

# %%
# --- measure standing-wave fringe spacing from the time-averaged profile ---
bulk_lo = iy_src + 3
bulk_hi = NY - NPML - 2
y_axis = np.arange(bulk_lo, bulk_hi) * DX - (iy_src * DX)   # height above dipole
p = prof[bulk_lo:bulk_hi].cpu().numpy()
# peaks of the oscillatory |Ez|^2 -> half-wavelength fringes
peaks = np.where((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]))[0] + 1
if len(peaks) >= 2:
    fringe_um = float(np.mean(np.diff(y_axis[peaks])))
else:
    fringe_um = float("nan")
lambda_med = WVL_UM / N_BG
theory_fringe = lambda_med / 2.0

# %% [markdown]
# ## Plot

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
v = float(state.Ez.abs().max()) or 1.0
ax1.imshow(eps.cpu().T, origin="lower", cmap="binary", alpha=0.9)
ax1.imshow(state.Ez.cpu().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax1.axvline(ix_src, color="y", ls=":", lw=0.8)
ax1.set_title("Ez snapshot (PEC ground plane at bottom)")
ax1.set_xlabel("x (cells, dx=0.05 um)")
ax1.set_ylabel("y (cells)")
ax2.plot(y_axis, p, "b-")
ax2.set_title("Time-avg |Ez|^2 above dipole (standing wave)")
ax2.set_xlabel("height above dipole (um)")
ax2.set_ylabel("|Ez|^2 (a.u.)")
fig.tight_layout()
fig.savefig(OUT / "antenna_pec_ground_plane_1D.png", dpi=150)
plt.close(fig)

print("antenna_pec_ground_plane_1D.png")
print(
    f"fringe_spacing={fringe_um:.4f} um  (theory lambda/(2n)={theory_fringe:.4f} um, "
    f"lambda_med={lambda_med:.4f} um, max|Ez|={v:.4e})"
)
