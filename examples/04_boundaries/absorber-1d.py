# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Absorber vs PML in a lossy background (1D)
#
# meep-torch port of meep/python/examples/absorber-1d.py. 1D ringdown of a
# Gaussian pulse in a lossy background, terminated by either mp.Absorber
# (matched sigma_d = sigma_b ramp, W5) or PML; the center-probe ringdown
# envelopes are overlaid — the point is that a plain PML misbehaves in a
# conductive background while the matched absorber rings down cleanly.
# Output: `output/absorber-1d.png`.
#
# SUBSTITUTIONS:
# * meep.materials.Al -> conductive medium (eps=1, D_conductivity=1). Al's
#   Drude pole (w0=0) has no meep-torch ADE representation; the example's
#   point (an Absorber terminating a lossy medium) survives with any lossy
#   background, and the meep-oracle reference (absorber_1d.json) uses the
#   same substitute so the comparison stays apples-to-apples.
# * Core-level run (Grid + run_tm, y_pmc 1D): the mp shim run() re-inits
#   state per call, and this example needs one continuous probe series.
#
# Run: `uv run python examples/04_boundaries/absorber-1d.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.pml import absorber_sigma
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

RES = 40
FCEN, DF = 1.0 / 0.803, 0.1
SIGMA_D = 1.0
T_END = 240.0

grid = Grid(nx=10 * RES, ny=8, dx=1.0 / RES)
nlayer = RES                                     # 1.0 um boundary layer

# %% [markdown]
# ## Ringdown driver
# One function runs the same lossy cell twice: with a matched-sigma absorber
# ramp (npml=0) or with a standard PML layer, recording |Ez| at the center.

# %%
def ringdown(use_absorber):
    sd = grid.zeros(grid.nx, grid.ny) + SIGMA_D  # lossy background everywhere
    kwargs = dict(sigma_d=sd, y_pmc=True)
    if use_absorber:
        ramp = absorber_sigma(grid.nx, nlayer, grid.dx,
                              dtype=grid.dtype, device=grid.device)[:, None]
        kwargs["sigma_d"] = sd + ramp
        kwargs["sigma_b"] = ramp.expand(grid.nx, grid.ny).clone()
        npml = 0
    else:
        npml = nlayer
    coeffs = build_tm_coeffs(grid, eps_inf=grid.zeros(grid.nx, grid.ny) + 1.0,
                             npml=npml, **kwargs)
    src = LineSource(grid, ix=grid.nx // 2, j0=0, j1=grid.ny,
                     profile=GaussianPulse(FCEN, DF))
    probe = PointProbe(grid.nx // 2, grid.ny // 2)
    n_steps = int(round(T_END / grid.dt))
    _, (series,) = run_tm(init_tm_state(grid, y_pmc=True), coeffs, n_steps,
                          source=src, monitors=[probe])
    return np.abs(series.detach().numpy())

# %% [markdown]
# ## Run both terminations and plot
# Semilog envelopes; the printed residual/peak ratio quantifies how completely
# each boundary swallowed the pulse.

# %%
s_abs = ringdown(True)
s_pml = ringdown(False)
t = np.arange(len(s_abs)) * grid.dt

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.semilogy(t, np.maximum(s_abs, 1e-25), lw=0.8, label="Absorber(1.0)")
ax.semilogy(t, np.maximum(s_pml, 1e-25), lw=0.8, label="PML(1.0)")
ax.set_xlabel("time (meep units)")
ax.set_ylabel("|Ez| at center")
ax.set_title(f"absorber-1d ringdown (conductive bg sigma_d={SIGMA_D})")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "absorber-1d.png", dpi=150)
plt.close(fig)

tail = slice(2 * len(s_abs) // 3, None)          # after t = 160
r_abs = float(s_abs[tail].max() / s_abs.max())
r_pml = float(s_pml[tail].max() / s_pml.max())
print(f"absorber-1d.png  residual/peak: absorber={r_abs:.2e}  pml={r_pml:.2e}")
