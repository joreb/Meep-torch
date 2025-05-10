# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/stochastic_emitter_reciprocity.py.

Reciprocity check for incoherent emission: the ensemble-averaged flux
radiated FROM emitters in a layer to a monitor plane equals (up to a
constant) the |E|^2 summed over the emitter region when a source is
placed AT the monitor plane (reciprocal run). Gate printed: the
textured/flat enhancement ratio computed both ways agrees.

SUBSTITUTIONS: single frequency, 8-seed forward ensemble, TM; the meep
example's broadband multi-polarization sweep is reduced.

Run: uv run python examples/stochastic_emitter_reciprocity.py
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
from meep_torch.core.monitors import DFTFieldsMonitor, FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource, RandomSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent
FCEN, N_HI, RES, N_SEEDS = 1.0, 3.5, 20, 8


def _scene(textured):
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


def forward(textured):
    # incoherent ensemble = average over SINGLE-point emitters (a coherent
    # line drive is NOT the reciprocal partner of the |E|^2 sum)
    grid, coeffs, ix_src = _scene(textured)
    ny = grid.ny
    mon = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=int(4.5 * RES),
                        lo=RES, hi=ny - RES)
    total = 0.0
    js = np.linspace(RES, ny - RES - 1, N_SEEDS).astype(int)
    for j in js:
        src = LineSource(grid, ix=ix_src, j0=int(j), j1=int(j) + 1,
                         profile=GaussianPulse(FCEN, 0.2))
        _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                           int(round(60.0 / grid.dt)), source=src,
                           monitors=[mon])
        total += float(mon.flux(acc, grid.dx)[0])
    return total / N_SEEDS


def reciprocal(textured):
    grid, coeffs, ix_src = _scene(textured)
    ny = grid.ny
    src = LineSource(grid, ix=int(4.5 * RES), j0=RES, j1=ny - RES,
                     profile=GaussianPulse(FCEN, 0.2))
    mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(60.0 / grid.dt)), source=src, monitors=[mon])
    ez = (acc[0, 0] + 1j * acc[0, 1]).detach().numpy()
    js = np.linspace(RES, ny - RES - 1, N_SEEDS).astype(int)
    return float(np.sum(np.abs(ez[ix_src, js]) ** 2))


f_tex, f_flat = forward(True), forward(False)
r_tex, r_flat = reciprocal(True), reciprocal(False)
ratio_fwd = f_tex / f_flat
ratio_rec = r_tex / r_flat
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.bar(["forward\n(point ensemble)", "reciprocal\n(1 run)"],
       [ratio_fwd, ratio_rec], color=["tab:blue", "tab:orange"])
ax.set_ylabel("textured / flat enhancement")
ax.set_title("emission reciprocity check")
fig.tight_layout()
fig.savefig(OUT / "stochastic_emitter_reciprocity.png", dpi=150)
plt.close(fig)
dev = abs(ratio_fwd - ratio_rec) / ratio_rec
print(f"stochastic_emitter_reciprocity  fwd {ratio_fwd:.3f} vs rec "
      f"{ratio_rec:.3f} (dev {dev:.1%}, statistical at {N_SEEDS} seeds)")
