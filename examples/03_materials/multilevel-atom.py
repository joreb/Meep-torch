# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Two-level-atom Fabry-Perot laser
#
# meep-torch port of meep/python/examples/multilevel-atom.py. 1D one-sided
# Fabry-Perot laser (Cerjan et al., Opt. Express 20, 474 (2012), Fig. 2):
# two-level gain medium (pump Rp, non-radiative rate_21, radiative transition
# at omega_a) against a PEC mirror, PML on the open side. The field grows from
# a seed and saturates; the steady-state spectrum peaks at the lasing mode
# near omega_a/2pi. Output: `output/multilevel-atom.png`.
#
# SUBSTITUTIONS:
# * mp.MultilevelAtom/mp.Transition -> core stepper_1d mla ADE (two-level
#   subset; the mp shim lands if a second example needs it).
# * meep z-axis -> x-axis, Ex -> Ey. Resolution 100 (meep's 400; the
#   reduced-resolution meep-oracle uses the same, gate in
#   tests/examples/test_w575_multilevel.py).
#
# Run: `uv run python examples/03_materials/multilevel-atom.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.stepper_1d import build_1d_coeffs, init_1d_state, step_1d

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Cavity, gain medium and seed
# n=1.5 cavity against the PEC left wall (no PML there), the two-level gain
# ADE active only inside the cavity, and a delta-function Ey seed to start the
# lasing buildup.

# %%
RES = 100
NCAV, LCAV, DPAD, DPML = 1.5, 1.0, 1.0, 1.0
OMEGA_A = 40
GAMMA_21 = (2 * 4) / (2 * math.pi)     # FWHM (SALT gamma_perp = 4)
SIGMA_21 = 2 * 1 * 1 * OMEGA_A         # theta = 1
RATE_21 = 0.005
N0 = 37
RP = 0.0051
ENDT = 2000.0
TAIL_T = 250.0

sz = LCAV + DPAD + DPML
grid = Grid(nx=int(sz * RES), ny=4, dx=1.0 / RES)
ncav_cells = int(LCAV * RES)
mask = torch.zeros(grid.nx, dtype=grid.dtype)
mask[:ncav_cells] = 1.0
eps = torch.ones(grid.nx, dtype=grid.dtype)
eps[:ncav_cells] = NCAV ** 2
coeffs = build_1d_coeffs(grid, eps=eps, npml=(0, int(DPML * RES)),
                         mla=(OMEGA_A, GAMMA_21, SIGMA_21, RP, RATE_21, mask))
state = init_1d_state(grid, n0=N0, n0_mask=mask)
state.Ey[ncav_cells // 2] = 1.0

# %% [markdown]
# ## Run
# 2000 time units: record the |Ey| envelope during buildup and the last 250
# time units at a probe for the steady-state spectrum.

# %%
probe_ix = int((LCAV + 0.5 * DPAD) * RES)
n_steps = int(round(ENDT / grid.dt))
tail_n = int(round(TAIL_T / grid.dt))
envelope_t, envelope = [], []
tail = np.empty(tail_n)
k = 0
for t in range(n_steps):
    state = step_1d(state, coeffs)
    if t % 2000 == 0:
        envelope_t.append(t * grid.dt)
        envelope.append(float(state.Ey.abs().max()))
    if t >= n_steps - tail_n:
        tail[k] = state.Ey[probe_ix].item()
        k += 1

# %% [markdown]
# ## Spectrum and plot
# The rfft of the saturated tail peaks at the lasing mode.

# %%
spec = np.abs(np.fft.rfft(tail))
freqs = np.fft.rfftfreq(len(tail), grid.dt)
band = (freqs > 5.0) & (freqs < 8.0)
f_peak = float(freqs[band][spec[band].argmax()])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(envelope_t, envelope, lw=0.9)
ax1.set_xlabel("time")
ax1.set_ylabel("max |Ey|")
ax1.set_title("lasing buildup + saturation")
ax2.plot(freqs[band], spec[band] / spec[band].max(), lw=0.9)
ax2.axvline(OMEGA_A / (2 * math.pi), color="g", ls=":", label="omega_a/2pi")
ax2.axvline(f_peak, color="r", ls="--", label=f"peak {f_peak:.3f}")
ax2.set_xlabel("frequency")
ax2.set_title("steady-state spectrum")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "multilevel-atom.png", dpi=150)
plt.close(fig)
print(f"multilevel-atom.png  lasing peak f={f_peak:.3f} "
      f"(omega_a/2pi={OMEGA_A / (2 * math.pi):.3f})")
