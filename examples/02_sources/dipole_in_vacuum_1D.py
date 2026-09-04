# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Point dipole in vacuum (1D)
#
# meep-torch port of meep/python/examples/dipole_in_vacuum_1D.py. The meep
# original computes a full 3D radiation pattern of a dipole in vacuum via
# Brillouin-zone / k-point integration -- that requires oblique-planewave
# (k_point) sources and DFT flux, which meep-torch does not support. Per the
# port instructions, this file ports the simpler underlying physics: a point
# dipole (LineSource, 1 cell in y) radiating into vacuum, with the field
# sampled at a probe and plotted vs time. Output:
# `output/dipole_in_vacuum_1D.png`.
#
# meep -> meep-torch mapping (1D problem modeled as thin 2D, x = propagation):
# ```
#   dimensions=1, cell z=12 um, resolution 50   ->  Grid(nx=600, ny=3, dx=0.02)
#   mp.PML(1, Z)                                ->  npml=50, y_pmc=True (PML on x only)
#   mp.GaussianSource(1.0, fwidth=0.1) Ex @ z=0 ->  LineSource(ix=300, j0=1, j1=2)
#   field probe @ z=+5 um                      ->  PointProbe(ix=550, iy=1)
# ```
# Run: `uv run python examples/02_sources/dipole_in_vacuum_1D.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Setup and run
# Thin 2D strip (PMC on y) = 1D vacuum; a narrow Gaussian pulse at the center,
# probe 5 um away.

# %%
# cell 12 um, pml 1 um each end, resolution 50 -> dx=0.02, nx=600, npml=50
grid = Grid(nx=600, ny=3, dx=0.02)
coeffs = build_tm_coeffs(grid, eps_inf=1.0, npml=50, y_pmc=True)

# point dipole: 1-cell LineSource at cell center (x=6 um -> ix=300), middle row
src = LineSource(grid, ix=300, j0=1, j1=2, profile=GaussianPulse(fcen=1.0, fwidth=0.1))

# probe at z=+5 um relative to center -> x=11 um -> ix=550
probe = PointProbe(ix=550, iy=1)

state = init_tm_state(grid)
n_steps = int(round(60.0 / grid.dt))  # 60 time units / dt(=0.01) = 6000
state, (ez_t,) = run_tm(state, coeffs, n_steps, source=src, monitors=(probe,))

# %% [markdown]
# ## Plot
# The probe records the pulse arriving after the 5 um flight time.

# %%
ez = ez_t.detach().cpu().numpy()
t = (torch.arange(n_steps).numpy() + 0.5) * grid.dt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, ez, lw=0.8)
ax.set_title("Dipole in vacuum -- Ez at probe z=+5 um (meep dipole_in_vacuum_1D)")
ax.set_xlabel("time (meep units)")
ax.set_ylabel("Ez")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "dipole_in_vacuum_1D.png", dpi=150)
plt.close(fig)

peak = float(ez.max() - ez.min())
print(f"dipole_in_vacuum_1D.png  peak-to-peak Ez at probe = {peak:.6e}")
