# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/material-dispersion.py.

omega(k) of a uniform DISPERSIVE medium (single Lorentz pole): run the
Bloch TM stepper (which carries the W1 Lorentz ADE) per k, harminv the
ringdown -> the two polariton branches and the polaritonic gap, overlaid
with the analytic roots of k^2 = omega^2 eps(omega).

SUBSTITUTIONS: mp.Simulation run_k_points -> Bloch stepper + harminv.

Run: uv run python examples/material-dispersion.py
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
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch import init_bloch_tm_state, step_bloch_tm
from meep_torch.modes.harminv import harminv

OUT = pathlib.Path(__file__).parent
EPS_INF, F0, GAMMA, SIGMA = 2.25, 1.1, 1e-5, 0.5


def eps_w(w):
    return EPS_INF + SIGMA * F0 ** 2 / (F0 ** 2 - w ** 2)


def bands_at(kf):
    grid = Grid(nx=40, ny=8, dx=0.1)
    coeffs = build_tm_coeffs(grid, eps_inf=torch.full((40, 8), EPS_INF, dtype=torch.float64),
                             poles=[(F0, GAMMA, SIGMA)], npml=10, y_pmc=True)
    st = init_bloch_tm_state(grid, n_poles=1)
    mask = grid.zeros(40, 8)
    mask[20, 3] = 1.0
    pulse = GaussianPulse(0.6, 1.2)
    ky = 2 * math.pi * kf
    n_steps = int(round(300.0 / grid.dt))
    series = np.empty(n_steps, dtype=complex)
    for t in range(n_steps):
        a = pulse.amplitude((t + 0.5) * grid.dt)
        st = step_bloch_tm(st, coeffs, ky, src_c=a, src_s=0.0, src_mask=mask)
        series[t] = complex(st.Ez_c[20, 3].item(), st.Ez_s[20, 3].item())
    tail = series[int(round(40.0 / grid.dt)):]
    return [m.freq for m in harminv(tail, grid.dt, 0.05, 1.6)
            if m.Q > 200 and m.amp > 1e-8]


ks = np.arange(0.1, 1.01, 0.1)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
for kf in ks:
    for f in bands_at(float(kf)):
        ax.plot(kf, f, "b.", ms=5)
# analytic branches: solve k = w sqrt(eps(w)) on each side of the pole
ws = np.linspace(0.02, 1.6, 800)
valid = eps_w(ws) > 0
ax.plot(ws[valid] * np.sqrt(eps_w(ws[valid])), ws[valid], "r,", ms=1,
        label="analytic $k=\\omega\\sqrt{\\epsilon(\\omega)}$")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.6)
ax.set_xlabel("$k$ ($2\\pi$/a)")
ax.set_ylabel("frequency")
ax.set_title("polariton dispersion of a Lorentz medium")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "material-dispersion.png", dpi=150)
plt.close(fig)
# gate: at k=0.5 the lower branch must satisfy the dispersion relation
fs = bands_at(0.5)
low = min(fs) if fs else 0
k_an = low * math.sqrt(eps_w(low))
print(f"material-dispersion.png  lower branch at k=0.5: f={low:.4f}, "
      f"k_analytic(f)={k_an:.4f} (dev {abs(k_an - 0.5) / 0.5:.1%})")
