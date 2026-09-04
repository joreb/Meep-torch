# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Gaussian beam from a line source
#
# meep-torch port of meep/python/examples/gaussian-beam.py. Meep uses
# mp.GaussianBeamSource (a Bessel-Gaussian beam, 3D-aware); meep-torch has no
# GaussianBeamSource, so per the porting rules we approximate it with a
# LineSource whose spatial mask is a Gaussian across the transverse span and a
# GaussianPulse temporal envelope. The beam is launched along +x (the
# LineSource is a vertical line on the left, Gaussian profile in y), which is
# the meep-torch LineSource's native orientation and is physically equivalent
# to the meep +y beam rotated by 90 degrees. Expect a traveling Gaussian packet
# with transverse FWHM ~ w0=0.8; output: `output/gaussian-beam.png`.
#
# meep -> meep-torch mapping:
# ```
#   s=14, resolution 50, dpml=2            ->  resolution reduced to 20 (dx=0.05) so
#                                              a 280x280 grid runs in < 90s; the
#                                              meep res=50 grid (700x700) is infeasible
#   PML(2.0)                               ->  npml=40
#   GaussianBeamSource @ y=-4, size x=14    ->  vertical LineSource at x=-4, gaussian
#                                              mask in y (waist w0=0.8), GaussianPulse(1.0)
#   ContinuousSource(fcen=1)               ->  GaussianPulse(fcen=1, fwidth=0.2)
#   sim.run(until=20)                      ->  20/dt = 800 steps
# ```
# Run: `uv run python examples/02_sources/gaussian-beam.py`

# %%
import math
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
# ## Grid and beam source
# The LineSource's uniform mask is overwritten with a Gaussian transverse
# profile of waist w0 centered at y=0 — that is the whole beam approximation.

# %%
s, dpml = 14, 2
res = 20                                 # meep res=50; reduced to 20 for runtime
dx = 1.0 / res
grid = Grid(nx=int(s / dx), ny=int(s / dx), dx=dx)   # 280 x 280
npml = int(round(dpml / dx))             # 40
# origin x,y in [-7, 7); index i -> x = -7 + i*dx
x0 = -0.5 * s

fcen, fwidth = 1.0, 0.3
w0 = 0.8                                  # beam waist (meep beam_w0)

# source line at x = -0.5*s + dpml + 1 = -4 -> ix = 60; spans full y (gaussian)
ix_src = int(round((-0.5 * s + dpml + 1.0 - x0) / dx))   # 60
src = LineSource(grid, ix=ix_src, j0=0, j1=grid.ny, profile=GaussianPulse(fcen, fwidth))
# overwrite the uniform mask with a Gaussian transverse profile centered at y=0
y = torch.tensor([x0 + (j + 0.5) * dx for j in range(grid.ny)])
src.mask = torch.zeros(grid.nx, grid.ny, dtype=torch.float64)
src.mask[ix_src, :] = torch.exp(-(y ** 2) / (w0 ** 2))

# %% [markdown]
# ## Run
# 1000 steps (t=25) catches the pulsed packet mid-domain so the transverse
# Gaussian profile is visible.

# %%
eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
state = init_tm_state(grid)
# meep until=20 with a CW source fills the domain steadily. A GaussianPulse
# (fwidth=0.3 -> w=3.33, peak at t0=16.7, t_end=33) emits a traveling packet;
# snapshot at t=25 (1000 steps) catches the packet mid-domain (source at x=-4,
# peak reaches x~3 by t=25), so the transverse Gaussian profile is visible.
n_steps = 1000                           # t = 1000*dt = 25 time units
for t in range(n_steps):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

# %% [markdown]
# ## Plot and beam-width check
# The transverse FWHM at the peak column should be close to the source waist
# w0 (the packet has propagated less than the Rayleigh range zR ~ 2 um).

# %%
fig, ax = plt.subplots(figsize=(6, 6))
v = float(state.Ez.abs().max()) or 1.0
ax.imshow(state.Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v)
ax.set_title("Gaussian beam (meep gaussian-beam.py) fcen=%.1f, w0=%.2f" % (fcen, w0))
ax.set_xlabel("x (cells, dx=0.05 um)")
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "gaussian-beam.png", dpi=150)
plt.close(fig)
print("gaussian-beam.png")
# The beam packet is mid-domain at t=25; report the peak |Ez| and the transverse
# FWHM at the column of peak field, which should be ~ the source waist w0 (the
# packet has only propagated ~7 um, less than the Rayleigh range
# zR = pi*w0^2/lambda = pi*0.64/1 = 2.0 um, so it has spread but is still Gaussian).
peak = float(state.Ez.abs().max())
peak_idx = int(state.Ez.abs().argmax())
col = peak_idx // grid.ny
row = peak_idx % grid.ny
col_prof = state.Ez[col, :].abs().detach().numpy()
half = peak / 2.0
above = np.where(col_prof >= half)[0]
if len(above) > 1:
    fwhm_um = (above[-1] - above[0]) * dx
else:
    fwhm_um = float("nan")
print("peak |Ez| = %.4g, transverse FWHM at peak column = %.3f um (source w0=0.8)" %
      (peak, fwhm_um))
