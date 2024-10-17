# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/chirped_pulse.py.

A linearly-chirped Gaussian pulse (down-chirp: higher frequencies at the
front) launched from the left and propagating in +x. The meep original uses a
2D cell with Y-mirror symmetry and a CustomSource; meep-torch has no mirror
symmetry or arbitrary CustomSource, so per the port instructions we model it
as 1D (thin 2D, x = propagation, PMC on y) and supply the chirp via a custom
profile class with .amplitude(t).

meep -> meep-torch mapping:
  resolution 40, sx=40, dpml=2, sy=6   ->  Grid(nx=880, ny=3, dx=0.05), npml=40
                                          (resolution reduced 40->20 for runtime)
  mp.CustomSource(chirp), Ez @ x=-sx/2  ->  LineSource(ix=40, j0=0, j1=3, ChirpedPulse)
  mp.PML(2, X)                          ->  npml=40, y_pmc=True
  run until t0+50 = 65                  ->  n_steps = 65 / dt
Run: uv run python examples/chirped_pulse.py
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
from meep_torch.core.run import run_tm
from meep_torch.core.sources import LineSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent


class ChirpedPulse:
    """Linearly-chirped Gaussian pulse.

    E(t) = exp(-a (t-t0)^2) * cos(2 pi v0 (t-t0) + b (t-t0)^2)
    which is Re[ exp(i 2 pi v0 (t-t0)) * exp(-a (t-t0)^2 + i b (t-t0)^2) ],
    matching the meep original (b<0 -> down-chirp: higher freq at the front).
    """

    def __init__(self, v0: float, a: float, b: float, t0: float):
        self.v0, self.a, self.b, self.t0 = v0, a, b, t0

    def amplitude(self, t: float) -> float:
        dt = t - self.t0
        return math.exp(-self.a * dt * dt) * math.cos(
            2 * math.pi * self.v0 * dt + self.b * dt * dt
        )


# parameters from the meep original
v0, a, b, t0 = 1.0, 0.2, -0.5, 15.0
profile = ChirpedPulse(v0, a, b, t0)

# resolution reduced 40 -> 20 for runtime (dx 0.025 -> 0.05); 1D via thin 2D + PMC on y
grid = Grid(nx=880, ny=3, dx=0.05)
coeffs = build_tm_coeffs(grid, eps_inf=1.0, npml=40, y_pmc=True)

# source at left edge (x=2 um inside left PML boundary -> ix=40), spanning full y
src = LineSource(grid, ix=40, j0=0, j1=3, profile=profile)

state = init_tm_state(grid)
n_steps = int(round((t0 + 50.0) / grid.dt))  # meep: until=t0+50
state, _ = run_tm(state, coeffs, n_steps, source=src)

# snapshot of Ez along x at final time (middle row)
ez = state.Ez[:, 1].detach().cpu().numpy()
x = (torch.arange(grid.nx).numpy() + 0.5) * grid.dx

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, ez, lw=0.8)
ax.set_title("Chirped pulse -- Ez(x) at final time (meep chirped_pulse, down-chirp)")
ax.set_xlabel("x (um)")
ax.set_ylabel("Ez")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "chirped_pulse.png", dpi=150)
plt.close(fig)

peak = float(np.max(np.abs(ez)))
print(f"chirped_pulse.png  max|Ez| in snapshot = {peak:.6e}")