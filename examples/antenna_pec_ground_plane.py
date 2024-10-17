# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/antenna_pec_ground_plane.py.

2D Ez dipole antenna above a PEC ground plane: a square 2D grid with an Ez point
dipole at height h above a metallic (eps=1e7) ground-plane strip along the
bottom. The near-field Ez pattern shows the antenna radiation reflecting off
the ground plane.

The meep original computes the *far-field radiation pattern* via a near-to-far
transform (add_near2far / get_farfield) using two opposite-sign dipoles plus
mirror symmetry to emulate the PEC. Near-to-far-field transforms are
UNSUPPORTED in meep-torch, so this port uses an explicit PEC strip and plots the
near-field Ez pattern only. The analytic array-factor comparison is omitted.

meep -> meep-torch mapping:
  mp.Vector3(sxy,sxy) cell, res 100            ->  Grid(nx=175, ny=175, dx=0.04) (res 25)
  mp.PML(0.5)                                  ->  npml=12 (0.5 um / dx 0.04)
  PEC ground plane (via mirror symmetry trick) ->  eps[:, 12:14] = 1e7 (explicit PEC strip)
  mp.GaussianSource(1/0.65, fwidth=0.2*fc) Ez ->  GaussianPulse(fcen, fwidth) 1-cell LineSource
  dipole at height 1.25 um above ground        ->  ix=center, iy=12+round(1.25/dx)

Run: uv run python examples/antenna_pec_ground_plane.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent

# --- parameters (meep natural units, um) ---
RES = 25                        # pixels/um (reduced from 100 for runtime)
DX = 1.0 / RES
PML_UM = 0.5
BULK_UM = 6.0
N_BG = 1.2
H_UM = 1.25
WVL_UM = 0.65
FCEN = 1.0 / WVL_UM
FWIDTH = 0.2 * FCEN

SXY = 2 * PML_UM + BULK_UM                       # 7.0 um
N = int(round(SXY / DX))                           # 175
NPML = int(round(PML_UM / DX))                    # 12

grid = Grid(nx=N, ny=N, dx=DX)
eps = torch.full((N, N), N_BG * N_BG, dtype=torch.float64)
# PEC ground plane: high-eps strip just inside the bottom PML.
eps[:, NPML : NPML + 2] = 1e7

coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

# Ez point dipole at lateral center, height H above the ground plane.
ix_src = N // 2
iy_src = NPML + 2 + int(round(H_UM / DX))
src = LineSource(grid, ix=ix_src, j0=iy_src, j1=iy_src + 1, profile=GaussianPulse(FCEN, FWIDTH))

state = init_tm_state(grid)
N_STEPS = int(round(70.0 / grid.dt))               # ~70 time units / dt(=0.02) = 3500
for t in range(N_STEPS):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

v = float(state.Ez.abs().max()) or 1.0
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(state.Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.plot(ix_src, iy_src, "y*", ms=10)
ax.set_title("Dipole antenna above PEC ground plane (near-field Ez)")
ax.set_xlabel("x (cells, dx=0.04 um)")
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "antenna_pec_ground_plane.png", dpi=150)
plt.close(fig)

print("antenna_pec_ground_plane.png")
print(f"max|Ez|={v:.4e}  (dipole at ({ix_src},{iy_src}), H={H_UM} um, res={RES})")