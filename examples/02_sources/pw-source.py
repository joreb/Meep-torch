# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Planewave at 45 degrees from two phased line sources
#
# meep-torch port of meep/python/examples/pw-source.py. Approximate
# Ez-polarized planewave in vacuum propagating at 45 degrees, built from two
# line sources whose spatial amplitude follows exp(ik.x). Meep supplies the
# phase via amp_func; meep-torch real fields take the real part cos(k.x) on
# each source line, with a CW temporal carrier (CWRamp). The two lines (left
# and bottom) together launch the 45-degree planewave. Output:
# `output/pw-source.png`.
#
# meep -> meep-torch mapping:
# ```
#   s=11, dpml=1, sxy=13, resolution 10    ->  Grid(nx=130, ny=130, dx=0.1), npml=10
#   left source:  center (-5.5, 0), size (0, 11)  ->  ix=10, j in [10,120]
#   bottom source: center (0, -5.5), size (11, 0) ->  iy=10, i in [10,120]
#   amp_func exp(ik.(x+x0))                 ->  mask = cos(k.x0 + k.r) (real part)
#   ContinuousSource(fcen=0.8, fwidth=0.02) ->  CWRamp(0.8, t_ramp=1/df=50)
#   sim.run(until=400)                      ->  4000 steps (reduced from 8000 to keep
#                                               runtime < 90s; the 13 um domain fills
#                                               in ~13 time units so t=200 is ample)
# ```
# Run: `uv run python examples/02_sources/pw-source.py`

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
from meep_torch.core.stepper import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Source construction
# One combined soft-source mask carries both line sources; each line gets the
# real part of exp(i k.x) evaluated along it.

# %%
s, dpml = 11, 1
sxy = s + 2 * dpml                       # 13
grid = Grid(nx=int(sxy / 0.1), ny=int(sxy / 0.1), dx=0.1)   # 130 x 130
# origin x,y in [-6.5, 6.5); cell index i -> x = -6.5 + i*dx
x0 = -0.5 * sxy                          # -6.5

fcen, df = 0.8, 0.02
n = 1
kdir = (1.0, 1.0)
knorm = math.sqrt(kdir[0] ** 2 + kdir[1] ** 2)
kx = kdir[0] / knorm * (2 * math.pi * fcen * n)
ky = kdir[1] / knorm * (2 * math.pi * fcen * n)

# Build a combined 2D soft-source mask holding both line sources. The meep
# amp_func is exp(i k.(x_rel + x0)); we take the real part on each line.
# left line: x=-5.5 -> ix=10, y varies; phase = kx*(-5.5) + ky*y  (x0=(-5.5,0))
# bottom line: y=-5.5 -> iy=10, x varies; phase = kx*x + ky*(-5.5) (x0=(0,-5.5))
mask = torch.zeros(grid.nx, grid.ny, dtype=torch.float64)
ix_left = int(round((-0.5 * s - x0) / grid.dx))    # 10
iy_bot = int(round((-0.5 * s - x0) / grid.dx))    # 10
j_lo = int(round((-0.5 * s - x0) / grid.dx))      # 10
j_hi = int(round(( 0.5 * s - x0) / grid.dx))      # 120
# left line (x = -5.5): cos(kx*(-5.5) + ky*y)
y_left = torch.tensor([-x0 + (j + 0.5) * grid.dx for j in range(j_lo, j_hi)])
mask[ix_left, j_lo:j_hi] = torch.cos(kx * (-0.5 * s) + ky * y_left)
# bottom line (y = -5.5): cos(kx*x + ky*(-5.5))
x_bot = torch.tensor([-x0 + (i + 0.5) * grid.dx for i in range(j_lo, j_hi)])
mask[j_lo:j_hi, iy_bot] = torch.cos(kx * x_bot + ky * (-0.5 * s))

# %% [markdown]
# ## Run
# CW drive ramped over 50 time units; 4000 steps reach a clean steady state.

# %%
eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
# wrap the mask in a LineSource shell so step_tm's source protocol is satisfied;
# the profile provides the temporal carrier.
src = LineSource(grid, ix=ix_left, j0=j_lo, j1=j_hi, profile=CWRamp(fcen, t_ramp=50.0))
src.mask = mask

state = init_tm_state(grid)
n_steps = 4000                          # meep until=400 -> 8000; reduced to keep <90s
for t in range(n_steps):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

# %% [markdown]
# ## Plot
# Phase fronts perpendicular to the (1,1) diagonal confirm the 45-degree
# planewave.

# %%
fig, ax = plt.subplots(figsize=(6, 6))
v = float(state.Ez.abs().max()) or 1.0
ax.imshow(state.Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v)
ax.set_title("45-degree planewave (meep pw-source.py) fcen=%.2f" % fcen)
ax.set_xlabel("x (cells, dx=0.1 um)")
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "pw-source.png", dpi=150)
plt.close(fig)
print("pw-source.png")
# 45-degree planewave: |Ez| along the propagation diagonal (x=y) should be
# nearly constant phase; report the peak amplitude.
print("peak |Ez| = %.4g (45-degree planewave, k=(%.3f,%.3f))" %
      (float(state.Ez.abs().max()), kx, ky))
