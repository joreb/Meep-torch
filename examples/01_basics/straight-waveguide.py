# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Straight waveguide (CW Ez)
#
# meep-torch port of meep/python/examples/straight-waveguide.py — the first meep
# tutorial: permittivity + Ez field of a 2D straight waveguide driven by a CW
# source. Expect a guided CW beam confined to the eps=12 strip, saved to
# `output/straight-waveguide.png`.
#
# meep -> meep-torch mapping:
# ```
#   mp.Vector3(16,8,0) cell, resolution 10  ->  Grid(nx=160, ny=80, dx=0.1)
#   mp.Block(inf,1,inf) eps=12 at y in [-0.5,0.5] ->  eps[:, 35:45] = 12.0
#   mp.PML(1.0)                             ->  npml=10 (1.0 um / dx 0.1)
#   mp.ContinuousSource(0.15) Ez @ x=-7     ->  LineSource(ix=10, j0=35, j1=45, CWRamp(0.15))
#   sim.run(until=200)                      ->  200 / (courant*dx) = 4000 steps
# ```
# Run: `uv run python examples/01_basics/straight-waveguide.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Geometry
# A 16x8 um cell at dx=0.1 with a 1 um wide eps=12 strip through the middle —
# the raster equivalent of meep's infinite Block.

# %%
# cell 16x8 um, centered at origin; dx=0.1 -> 160x80, x in [-8,8), y in [-4,4)
grid = Grid(nx=160, ny=80, dx=0.1)
eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
eps[:, 35:45] = 12.0                      # waveguide y in [-0.5, 0.5] -> rows 35:45

# %% [markdown]
# ## Source and PML
# A CW Ez line source at x=-7 spanning the guide core, ramped over 50 time
# units so the turn-on transient does not pollute the steady state.

# %%
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
# CW Ez source at x=-7 -> ix=10, on the waveguide rows
src = LineSource(grid, ix=10, j0=35, j1=45, profile=CWRamp(0.15, t_ramp=50.0))

# %% [markdown]
# ## Run
# 200 meep time units = 4000 steps; enough for the CW field to fill the cell.

# %%
state = init_tm_state(grid)
n_steps = int(round(200.0 / grid.dt))     # 200 time units / dt(=0.05) = 4000
for t in range(n_steps):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

# %% [markdown]
# ## Plot
# Permittivity in grayscale with the Ez snapshot overlaid.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
v = float(state.Ez.abs().max()) or 1.0
ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(state.Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.set_title("Straight waveguide -- CW Ez at f=0.15 (meep straight-waveguide.py)")
ax.set_xlabel("x (cells, dx=0.1 um)")
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "straight-waveguide.png", dpi=150)
plt.close(fig)
print("straight-waveguide.png")
