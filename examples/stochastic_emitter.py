# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/stochastic_emitter.py.

Ensemble-averaged emission of stochastic (incoherent) line emitters inside
a textured high-index layer: seed-averaged radiated flux above the
structure via RandomSource (W5.5), textured vs flat comparison.

SUBSTITUTIONS:
  * meep CustomSource random arrays + reciprocity cross-check -> direct
    ensemble only (RandomSource, N seeds); reduced N=8, res 20, single
    frequency (the meep example averages a spectrum; at one frequency
    thin-film interference can go either way -- the port demonstrates the
    incoherent-ensemble machinery, not a broadband enhancement claim).

Run: uv run python examples/stochastic_emitter.py
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
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import RandomSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent
FCEN, N_HI, RES = 1.0, 3.5, 20
N_SEEDS = 8


def ensemble_flux(textured):
    dx = 1.0 / RES
    nx, ny = int(6.0 * RES), int(8.0 * RES)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    eps = torch.ones(nx, ny, dtype=torch.float64)
    i0, i1 = int(2.0 * RES), int(3.0 * RES)       # high-index layer x in [2, 3]
    eps[i0:i1, :] = N_HI ** 2
    if textured:                                   # etched grooves, period 1
        for jc in range(0, ny, RES):
            eps[i1 - RES // 4: i1, jc: jc + RES // 2] = 1.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES)
    mon = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=int(4.5 * RES),
                        lo=RES, hi=ny - RES)
    n_steps = int(round(60.0 / grid.dt))
    total = 0.0
    for seed in range(N_SEEDS):
        src = RandomSource(grid, ix=(i0 + i1) // 2, j0=RES, j1=ny - RES,
                           seed=seed)
        _, (acc,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                           source=src, monitors=[mon])
        total += float(mon.flux(acc, dx)[0])
    return total / N_SEEDS


f_tex = ensemble_flux(True)
f_flat = ensemble_flux(False)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.bar(["flat", "textured"], [f_flat, f_tex], color=["gray", "tab:blue"])
ax.set_ylabel(f"ensemble flux above the layer ({N_SEEDS} seeds)")
ax.set_title(f"stochastic emitters in n={N_HI} layer")
fig.tight_layout()
fig.savefig(OUT / "stochastic_emitter.png", dpi=150)
plt.close(fig)
print(f"stochastic_emitter.png  flux flat={f_flat:.3e}  textured={f_tex:.3e}  "
      f"enhancement x{f_tex / f_flat:.2f}")
