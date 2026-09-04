# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Cylinder with an ellipsoidal air hole (meep smoke test)
#
# The classic meep sanity check: a dielectric cylinder (r=3, n=3.5) with a
# 1 x 2 air ellipsoid carved from its center, excited by an Hz (TE) Gaussian
# pulse at the origin. We snapshot the field after the pulse transits.

# %%
"""meep-torch port of meep/python/examples/cyl-ellipsoid.py.

The classic meep smoke test: a dielectric cylinder (r=3, n=3.5) with an
air ellipsoid (1 x 2) carved from its center, Hz (TE) Gaussian pulse at
the origin; field snapshot after the pulse transits.

SUBSTITUTIONS: TE via build_te_coeffs/run_te; no Mirror symmetries
(full cell); supersampled eps.

Run: uv run python examples/09_cylindrical/cyl-ellipsoid.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_te
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_te import init_te_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Supersampled geometry and TE run
#
# The eps profile is rasterized on a 4x finer grid and box-averaged down,
# a cheap stand-in for meep's subpixel smoothing. A single-cell Ey source at
# the center drives the TE (Hz) polarization for 23 time units.

# %%
RES = 10

dx = 1.0 / RES
n = int(10.0 * RES)
grid = Grid(nx=n, ny=n, dx=dx)
ss = 4
xi = (torch.arange(n * ss, dtype=torch.float64, device=grid.device) + 0.5) / (RES * ss) - 5.0
II, JJ = torch.meshgrid(xi, xi, indexing="ij")
fine = torch.ones(n * ss, n * ss, dtype=torch.float64, device=grid.device)
fine[II ** 2 + JJ ** 2 < 3.0 ** 2] = 3.5 ** 2
fine[(II / 0.5) ** 2 + (JJ / 1.0) ** 2 < 1.0] = 1.0     # ellipsoid 1 x 2
eps = fine.reshape(n, ss, n, ss).mean(dim=(1, 3))

coeffs = build_te_coeffs(grid, eps_inf=eps, npml=RES)
src = LineSource(grid, ix=n // 2, j0=n // 2, j1=n // 2 + 1,
                 profile=GaussianPulse(1.0, 0.1))
src.mask = src.mask[:, :-1]                              # Ey grid
state = init_te_state(grid)
state, _ = run_te(state, coeffs, int(round(23.0 / grid.dt)), source=src,
                  monitors=[])

# %% [markdown]
# ## Plot
#
# Hz snapshot at t=23 over the eps outline: the pulse is still ringing inside
# the high-index cylinder.

# %%
hz = state.Hz.cpu().numpy()
fig, ax = plt.subplots(figsize=(5.4, 5))
v = float(np.abs(hz).max()) or 1.0
ax.imshow(eps.cpu().numpy().T, origin="lower", cmap="binary", alpha=0.85)
ax.imshow(hz.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.75)
ax.set_title("cyl-ellipsoid: Hz at t=23 (TE)")
fig.tight_layout()
fig.savefig(OUT / "cyl-ellipsoid.png", dpi=150)
plt.close(fig)
print(f"cyl-ellipsoid.png  |Hz|max = {v:.3e} (pulse inside the cylinder)")
