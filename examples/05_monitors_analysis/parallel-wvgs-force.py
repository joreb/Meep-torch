# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Optical force between parallel waveguides
#
# meep-torch port of meep/python/examples/parallel-wvgs-force.py. Optical
# force between two parallel dielectric waveguides (n=3.45, w=0.5) driven in
# their even / odd supermode: the even mode pulls the guides together, the
# odd mode pushes them apart, and both forces decay with separation. Force
# from the W6 Maxwell-stress-tensor contour around ONE waveguide; the
# supermode is launched with an fd1d cross-section profile as the source's
# spatial mask. Output: `output/parallel-wvgs-force.png`.
#
# SUBSTITUTIONS:
# * mp.EigenModeSource + get_force -> fd1d supermode-profile LineSource
#   (launches the target parity; this was the example's true blocker,
#   closed core-level) + core.forces.mst_force_tm on DFT planes.
# * TM (Ez) polarization; reduced res 20, two separations.
#
# Run: `uv run python examples/05_monitors_analysis/parallel-wvgs-force.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.forces import mst_force_tm
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, RES, N_WG, W_WG = 0.2, 20, 3.45, 0.5

# %% [markdown]
# ## Force per configuration
# Build the two-guide cell, launch the parity-selected supermode (fd1d
# profile as the source mask), record DFT planes, and evaluate the MST
# contour force on the upper guide.

# %%
def force_per_length(sep, parity):
    dx = 1.0 / RES
    sx, sy = 12.0, 8.0
    nx, ny = int(sx * RES), int(sy * RES)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    eps = torch.ones(nx, ny, dtype=torch.float64)
    cy = sy / 2
    y1 = int((cy - sep / 2 - W_WG) * RES), int((cy - sep / 2) * RES)
    y2 = int((cy + sep / 2) * RES), int((cy + sep / 2 + W_WG) * RES)
    eps[:, y1[0]:y1[1]] = N_WG ** 2
    eps[:, y2[0]:y2[1]] = N_WG ** 2
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES)

    # supermode profile from the 1D cross-section (parity-selected)
    modes = solve_1d(eps[0, :].numpy(), dx, FCEN, "tm", k=4)
    field = None
    for i in range(modes.profiles.shape[0]):
        v = modes.profiles[i].numpy()
        sym = v[::-1]
        even = np.linalg.norm(v - sym) < np.linalg.norm(v + sym)
        if even == (parity == "even"):
            field = v / np.abs(v).max()
            break
    src = LineSource(grid, ix=int(2.0 * RES), j0=0, j1=ny,
                     profile=GaussianPulse(FCEN, 0.05))
    src.mask = torch.zeros(nx, ny, dtype=torch.float64)
    src.mask[int(2.0 * RES), :] = torch.from_numpy(field)

    mons = [DFTFieldsMonitor(grid, [FCEN], c) for c in ("Ez", "Hx", "Hy")]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(120.0 / grid.dt)),
                     source=src, monitors=mons)
    planes = {c: (a[:, 0] + 1j * a[:, 1]).detach().numpy()
              for c, a in zip(("Ez", "Hx", "Hy"), accs)}
    # MST box around the UPPER waveguide, x-span inside the PML margins
    fx, fy = mst_force_tm(planes, grid,
                          ix0=int(4.0 * RES), ix1=int(8.0 * RES),
                          iy0=int((cy + sep / 4) * RES),
                          iy1=int((cy + sep / 2 + W_WG + 0.5) * RES))
    return float(fy[0])                            # +y = repulsive

# %% [markdown]
# ## Sweep and plot
# Even should come out attractive (negative Fy on the upper guide), odd
# repulsive.

# %%
seps = [0.2, 0.4]
rows = []
for parity in ("even", "odd"):
    for sep in seps:
        rows.append((parity, sep, force_per_length(sep, parity)))
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for parity, marker in (("even", "bo-"), ("odd", "rs-")):
    vals = [r[2] for r in rows if r[0] == parity]
    ax.plot(seps, vals, marker, label=parity)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("separation (um)")
ax.set_ylabel("$F_y$ on the upper guide (a.u.)")
ax.set_title("parallel waveguides: even attracts, odd repels")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "parallel-wvgs-force.png", dpi=150)
plt.close(fig)
for parity, sep, f in rows:
    print(f"parallel-wvgs-force  {parity} sep={sep}: Fy = {f:+.3e}")
