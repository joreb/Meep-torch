# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/faraday-rotation.py.

Faraday rotation of a linearly polarized CW plane wave in a gyrotropic
Lorentzian medium (bias along the propagation axis), via the W5.5 1D
full-vector stepper (Ey, Ez coupled through the gyrotropic ADE).

SUBSTITUTIONS:
  * meep z-axis -> meep-torch x-axis; meep (Ex, Ey) -> (Ey, Ez).
  * Source matches meep's ContinuousSource carrier (cos, abrupt turn-on)
    with the -1 polarity of meep's current convention.

Analytic overlay (meep tutorial): Ey ~ cos(kappa (x - x_src)),
Ez ~ sin(kappa (x - x_src)) modulating the carrier, kappa = (k- - k+)/2.

Run: uv run python examples/faraday-rotation.py
"""
import cmath
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.grid import Grid
from meep_torch.core.stepper_1d import build_1d_coeffs, init_1d_state, step_1d

OUT = pathlib.Path(__file__).parent

EPSN, F0, GAMMA, SN, B0 = 1.5, 1.0, 1e-6, 0.1, 0.15
FSRC, SRC_X, L, TMAX = 0.8, 1.5, 20.0, 100.0
RES = 50

grid = Grid(nx=int(L * RES), ny=4, dx=1.0 / RES)
coeffs = build_1d_coeffs(grid, eps=EPSN, npml=RES, gyro=(F0, GAMMA, SN, B0))
state = init_1d_state(grid)
src_ix = int(round(SRC_X / grid.dx))
for t in range(int(round(TMAX / grid.dt))):
    amp = -math.cos(2 * math.pi * FSRC * (t + 0.5) * grid.dt)
    state = step_1d(state, coeffs, amp, src_ix)

x = np.arange(grid.nx) * grid.dx
ey = state.Ey.numpy()
ez = state.Ez.numpy()

# analytic rotation wavenumber (tutorial formulas)
dfsq = F0 ** 2 - 1j * FSRC * GAMMA - FSRC ** 2
eperp = EPSN + SN * F0 ** 2 * dfsq / (dfsq ** 2 - (FSRC * B0) ** 2)
eta = SN * F0 ** 2 * FSRC * B0 / (dfsq ** 2 - (FSRC * B0) ** 2)
kp = 2 * math.pi * FSRC * cmath.sqrt(eperp - eta).real
km = 2 * math.pi * FSRC * cmath.sqrt(eperp + eta).real
kappa = (km - kp) / 2
amp0 = max(np.abs(ey).max(), np.abs(ez).max())

fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True)
for ax, field, name, env in (
    (axes[0], ey, "Ey (meep Ex)", np.cos(kappa * (x - SRC_X))),
    (axes[1], ez, "Ez (meep Ey)", np.sin(kappa * (x - SRC_X))),
):
    ax.plot(x, field / amp0, lw=0.7, label=name)
    ax.plot(x, env, "r--", lw=1.0, label="rotation envelope (analytic)")
    ax.plot(x, -env, "r--", lw=1.0)
    ax.set_ylabel(name)
    ax.legend(fontsize=8, loc="upper right")
axes[1].set_xlabel("x (um)")
axes[0].set_title(f"Faraday rotation, kappa={kappa:.4f} (analytic)")
fig.tight_layout()
fig.savefig(OUT / "faraday-rotation.png", dpi=150)
plt.close(fig)
print(f"faraday-rotation.png  kappa_analytic={kappa:.5f}  "
      f"(rotation period {math.pi / kappa:.2f} um)")
