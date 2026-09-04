# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # 2D edge emitter at a cleaved facet
#
# A cleaved multilayer slab waveguide (substrate / cladding / active /
# cladding / air) with an Ez dipole in the active region near the facet.
# The port keeps the near-field physics: guided light reaches the cleaved
# facet and partially radiates into the air side.

# %%
"""meep-torch port of meep/python/examples/edge_emitter_2D.py.

2D edge emitter: a cleaved multilayer slab waveguide (substrate / cladding /
active / cladding / air) with an Ez dipole in the active region near the
cleaved facet. The guided/radiated field is imaged in the near field; radiation
escaping into the air side at the facet is visible.

The meep original computes the *extraction efficiency* via Brillouin-zone
integration over kz (k_point Bloch boundaries), dft_ldos, and DFT flux monitors.
Bloch boundaries, dft_ldos, and DFT flux are UNSUPPORTED in meep-torch, so this
port keeps only the near-field physics and plots the Ez field. The extraction-
efficiency integration is omitted.

meep -> meep-torch mapping:
  cell 5.0 x 4.3 um, res 25                     ->  Grid(nx=125, ny=108, dx=0.04)
  mp.PML(0.5)                                   ->  npml=12 (0.5 um / dx 0.04)
  mp.Block substrate n=3.2                      ->  eps[0:88, 12:37]  = 10.24
  mp.Block cladding n=1.5                       ->  eps[0:88, 37:50]  = 2.25  / [50:58]=2.25
  mp.Block active  n=2.3                        ->  eps[0:88, 50:58]  = 5.29
  mp.GaussianSource(1/1.0, fwidth=0.1*fc) Ez    ->  GaussianPulse (broadened to 0.2*fc for runtime)
  dipole 0.5 um inside facet, active center      ->  ix=75, iy=54

Run: uv run python examples/08_devices/edge_emitter_2D.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Layer stack and grid
#
# The stack is painted row-by-row into eps up to the facet column ix_facet;
# to its right the cell is air. Layer indices: substrate n=3.2, cladding
# n=1.5, active n=2.3.

# %%
# --- parameters (meep natural units, um) ---
RES = 25                        # pixels/um (meep original: 25)
DX = 1.0 / RES
PML_UM = 0.5
SUB_UM = 1.0
CLAD_UM = 0.5
ACTIVE_UM = 0.3
AIR_UM = 1.0
SIDE_UM = 3.0
DIPOLE_POS_UM = 0.5             # 0.5 um inside the cleaved facet
N_SUB = 3.2
N_CLAD = 1.5
N_ACTIVE = 2.3
WVL_UM = 1.0
FCEN = 1.0 / WVL_UM
FWIDTH = 0.2 * FCEN            # broadened from 0.1*fc for shorter pulse / runtime

CELL_X = PML_UM + SIDE_UM + AIR_UM + PML_UM     # 5.0
CELL_Y = PML_UM + SUB_UM + 2 * CLAD_UM + ACTIVE_UM + AIR_UM + PML_UM  # 4.3
NX = int(round(CELL_X / DX))                     # 125
NY = int(round(CELL_Y / DX))                      # 108
NPML = int(round(PML_UM / DX))                   # 12
FACET_X = PML_UM + SIDE_UM                       # 3.5 um (stack ends, air begins)

grid = Grid(nx=NX, ny=NY, dx=DX)
eps = torch.ones((NX, NY), dtype=torch.float64, device=grid.device)   # air background (n=1)
ix_facet = int(round(FACET_X / DX))               # 88 (right edge of stack)

# y boundaries (rows), from the bottom up:
y0 = 0
y_pml1 = int(round(PML_UM / DX))                  # 12
y_sub = int(round((PML_UM + SUB_UM) / DX))        # 37
y_clad1 = int(round((PML_UM + SUB_UM + CLAD_UM) / DX))      # 50
y_act = int(round((PML_UM + SUB_UM + CLAD_UM + ACTIVE_UM) / DX))  # 58
y_clad2 = int(round((PML_UM + SUB_UM + 2 * CLAD_UM + ACTIVE_UM) / DX))  # 63
# above y_clad2 up to NY is air
eps[:ix_facet, y_pml1:y_sub] = N_SUB ** 2          # substrate
eps[:ix_facet, y_sub:y_clad1] = N_CLAD ** 2       # cladding (lower)
eps[:ix_facet, y_clad1:y_act] = N_ACTIVE ** 2     # active
eps[:ix_facet, y_act:y_clad2] = N_CLAD ** 2       # cladding (upper)

# %% [markdown]
# ## Dipole source and run
#
# A Gaussian-pulse Ez dipole sits at the active-layer center, 0.5 um inside
# the facet, and the fields are stepped for ~80 time units.

# %%
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

# Ez point dipole in the active region, DIPOLE_POS_UM inside the facet.
ix_src = int(round((FACET_X - DIPOLE_POS_UM) / DX))   # 75
iy_src = int(round((PML_UM + SUB_UM + CLAD_UM + 0.5 * ACTIVE_UM) / DX))  # 54
src = LineSource(grid, ix=ix_src, j0=iy_src, j1=iy_src + 1, profile=GaussianPulse(FCEN, FWIDTH))

state = init_tm_state(grid)
N_STEPS = int(round(80.0 / grid.dt))               # ~80 time units / dt(=0.02) = 4000
for t in range(N_STEPS):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

# %% [markdown]
# ## Near-field snapshot
#
# Ez over the eps outline, with the facet and dipole position marked.

# %%
v = float(state.Ez.abs().max()) or 1.0
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(eps.cpu().T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(state.Ez.cpu().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.axvline(ix_facet, color="y", ls=":", lw=0.8, label="cleaved facet")
ax.plot(ix_src, iy_src, "y*", ms=10)
ax.set_title("Edge emitter -- Ez near field at cleaved facet")
ax.set_xlabel("x (cells, dx=0.04 um)")
ax.set_ylabel("y (cells)")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "edge_emitter_2D.png", dpi=150)
plt.close(fig)

print("edge_emitter_2D.png")
print(f"max|Ez|={v:.4e}  (dipole at ({ix_src},{iy_src}), {DIPOLE_POS_UM} um inside facet)")