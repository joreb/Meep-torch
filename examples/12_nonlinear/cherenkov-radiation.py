# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Cherenkov radiation from a superluminal point charge
#
# A point charge moving at v = 0.7c through an n = 1.5 medium exceeds the local
# phase velocity c/n = 0.667c, radiating a Cherenkov cone whose half-angle
# satisfies cos(theta) = 1/(n v).

# %%
"""meep-torch port of meep/python/examples/cherenkov-radiation.py.

Cherenkov cone of a point charge moving at v = 0.7c through an n = 1.5
medium (v > c/n = 0.667): a soft Ez current whose injection cell moves
along x each step. The cone half-angle satisfies cos(theta) = 1/(n v).

SUBSTITUTIONS: meep's change_sources moving source -> per-step moving
mask in a plain run loop (core stepper).

Run: uv run python examples/12_nonlinear/cherenkov-radiation.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Moving source in a uniform n = 1.5 medium
#
# meep's `change_sources` trick becomes a per-step moving injection mask in a
# plain stepping loop: each step the single-cell Ez current advances by
# v * dt along x.

# %%
V, N_MED, RES = 0.7, 1.5, 20

dx = 1.0 / RES
nx, ny = int(20.0 * RES), int(10.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
coeffs = build_tm_coeffs(grid, eps_inf=torch.full((nx, ny), N_MED ** 2,
                                                  dtype=torch.float64), npml=RES)
state = init_tm_state(grid)
jy = ny // 2
x0 = 2.0
n_steps = int(round((16.0 / V) / grid.dt))
for t in range(n_steps):
    x = x0 + V * t * grid.dt
    ix = int(round(x / dx))
    if ix >= nx - RES:
        break
    mask = grid.zeros(nx, ny)
    mask[ix, jy] = 1.0
    state = step_tm(state, coeffs, 1.0, mask)

# %% [markdown]
# ## Plot
#
# The final Ez snapshot shows the characteristic cone; the title quotes the
# analytic half-angle for comparison.

# %%
ez = state.Ez.numpy()
fig, ax = plt.subplots(figsize=(8, 4.2))
v = float(np.abs(ez).max()) or 1.0
ax.imshow(ez.T, origin="lower", cmap="RdBu", vmin=-v / 3, vmax=v / 3,
          extent=[0, 20, 0, 10])
theta = math.degrees(math.acos(1.0 / (N_MED * V)))
ax.set_title(f"Cherenkov cone, v=0.7c, n=1.5 "
             f"(analytic half-angle {theta:.1f} deg)")
ax.set_xlabel("x (um)")
fig.tight_layout()
fig.savefig(OUT / "cherenkov-radiation.png", dpi=150)
plt.close(fig)
print(f"cherenkov-radiation.png  cos(theta) = 1/(n v) -> theta = {theta:.1f} deg")
