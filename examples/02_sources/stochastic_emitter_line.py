# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Coherent line vs incoherent point emitters
#
# meep-torch port of meep/python/examples/stochastic_emitter_line.py.
# Line-emitter variant of the stochastic-emission family: COHERENT line drive
# (one random temporal signal on the whole line, stochastic_emitter machinery)
# vs the INCOHERENT point ensemble (reciprocity port) in the same textured
# structure -- the two differ by the emitters' mutual coherence, printed side
# by side. Output: `output/stochastic_emitter_line.png`.
#
# Run: `uv run python examples/02_sources/stochastic_emitter_line.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource, RandomSource
from meep_torch.core.stepper_tm import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, N_HI, RES, N_SEEDS = 1.0, 3.5, 20, 8

# %% [markdown]
# ## Shared scene
# The same textured/flat high-index layer used by the whole stochastic-emitter
# family.

# %%
def scene(textured):
    dx = 1.0 / RES
    nx, ny = int(6.0 * RES), int(8.0 * RES)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    eps = torch.ones(nx, ny, dtype=torch.float64)
    i0, i1 = int(2.0 * RES), int(3.0 * RES)
    eps[i0:i1, :] = N_HI ** 2
    if textured:
        for jc in range(0, ny, RES):
            eps[i1 - RES // 4: i1, jc: jc + RES // 2] = 1.0
    return grid, build_tm_coeffs(grid, eps_inf=eps, npml=RES), (i0 + i1) // 2

# %% [markdown]
# ## Two drive models
# `line_coherent` puts ONE random temporal signal on the whole line (mutually
# coherent emitters); `point_ensemble` averages independent single-point
# emitters (fully incoherent).

# %%
def line_coherent(textured):
    grid, coeffs, ix = scene(textured)
    ny = grid.ny
    mon = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=int(4.5 * RES),
                        lo=RES, hi=ny - RES)
    tot = 0.0
    for seed in range(N_SEEDS):
        src = RandomSource(grid, ix=ix, j0=RES, j1=ny - RES, seed=seed)
        _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                           int(round(60.0 / grid.dt)), source=src, monitors=[mon])
        tot += float(mon.flux(acc, grid.dx)[0])
    return tot / N_SEEDS


def point_ensemble(textured):
    grid, coeffs, ix = scene(textured)
    ny = grid.ny
    mon = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=int(4.5 * RES),
                        lo=RES, hi=ny - RES)
    tot = 0.0
    js = np.linspace(RES, ny - RES - 1, N_SEEDS).astype(int)
    for j in js:
        src = LineSource(grid, ix=ix, j0=int(j), j1=int(j) + 1,
                         profile=GaussianPulse(FCEN, 0.2))
        _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                           int(round(60.0 / grid.dt)), source=src, monitors=[mon])
        tot += float(mon.flux(acc, grid.dx)[0])
    return tot / N_SEEDS

# %% [markdown]
# ## Run and plot
# The textured/flat enhancement ratio differs between the two coherence
# models — that difference is the point of the example.

# %%
ratios = {
    "coherent line": line_coherent(True) / line_coherent(False),
    "incoherent points": point_ensemble(True) / point_ensemble(False),
}
fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.bar(list(ratios), list(ratios.values()), color=["tab:blue", "tab:orange"])
ax.set_ylabel("textured / flat flux enhancement")
ax.set_title("emitter coherence changes the answer")
fig.tight_layout()
fig.savefig(OUT / "stochastic_emitter_line.png", dpi=150)
plt.close(fig)
for k, r in ratios.items():
    print(f"stochastic_emitter_line  {k}: enhancement = {r:.3f}")
