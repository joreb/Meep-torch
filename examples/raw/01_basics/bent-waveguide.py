# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Bent waveguide (CW Ez)
#
# meep-torch port of meep/python/examples/bent-waveguide.py — permittivity + Ez
# field of a 90-degree bent waveguide driven by a CW source. Expect the guided
# wave to turn the corner with visible radiation loss at the bend; saved to
# `output/bent-waveguide.png`.
#
# meep -> meep-torch mapping:
# ```
#   mp.Vector3(16,16,0) cell, resolution 10  ->  Grid(nx=160, ny=160, dx=0.1)
#   mp.Block(12,1,inf) eps=12 @ (-2.5,-3.5)  ->  eps[0:115, 40:50] = 12
#   mp.Block(1,12,inf) eps=12 @ (3.5, 2)      ->  eps[110:120, 40:160] = 12
#   mp.PML(1.0)                              ->  npml=10
#   mp.ContinuousSource(wavelength=2*sqrt(11), width=20) Ez @ (-7,-3.5)
#     ->  LineSource(ix=10, j0=40, j1=50, CWRamp(fcen=1/(2*sqrt(11)), t_ramp=20))
#   sim.run(until=200)                       ->  200 / dt = 4000 steps (reduced to
#                                                3000 to keep runtime < 90s; field
#                                                is fully developed by t=150)
# ```
# Run: `uv run python examples/01_basics/bent-waveguide.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp, LineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Geometry
# Two overlapping eps=12 Blocks form the L-shaped guide: a horizontal arm along
# y=-3.5 and a vertical arm at x=3.5.

# %%
# cell 16x16 um, centered at origin; dx=0.1 -> 160x160, x,y in [-8,8)
grid = Grid(nx=160, ny=160, dx=0.1)
eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64, device=grid.device)
# horizontal arm: size (12,1) center (-2.5,-3.5) -> x in [-8,3.5], y in [-4,-3]
eps[0:115, 40:50] = 12.0
# vertical arm: size (1,12) center (3.5,2) -> x in [3,4], y in [-4,8]
eps[110:120, 40:160] = 12.0

# %% [markdown]
# ## Source and PML
# A CW Ez line source across the horizontal arm at x=-7, at the wavelength
# 2*sqrt(11) of the meep tutorial.

# %%
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
# CW Ez source at x=-7 -> ix=10, on the waveguide rows y in [-4,-3] -> j in [40,50]
fcen = 1.0 / (2.0 * math.sqrt(11.0))   # meep wavelength=2*sqrt(11)
src = LineSource(grid, ix=10, j0=40, j1=50, profile=CWRamp(fcen, t_ramp=20.0))

# %% [markdown]
# ## Run
# 3000 steps (150 time units) — the CW field is fully developed by then.

# %%
state = init_tm_state(grid)
n_steps = 3000                          # meep until=200 -> 4000; reduced to keep <90s
for t in range(n_steps):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

# %% [markdown]
# ## Plot
# Permittivity in grayscale with the Ez snapshot overlaid; the console prints
# the peak |Ez|.

# %%
fig, ax = plt.subplots(figsize=(6, 6))
v = float(state.Ez.abs().max()) or 1.0
ax.imshow(eps.cpu().T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(state.Ez.cpu().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.set_title("Bent waveguide -- CW Ez at f=%.4f (meep bent-waveguide.py)" % fcen)
ax.set_xlabel("x (cells, dx=0.1 um)")
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "bent-waveguide.png", dpi=150)
plt.close(fig)
print("bent-waveguide.png")
print("peak |Ez| = %.4g" % float(state.Ez.abs().max()))
