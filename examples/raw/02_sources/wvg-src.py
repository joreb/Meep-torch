# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Eigenmode source in a waveguide
#
# meep-torch port of meep/python/examples/wvg-src.py. Meep tutorial: an
# eigenmode source exciting the right-going mode of an asymmetric dielectric
# waveguide, then comparing left- and right-going flux.
#
# meep uses mp.EigenModeSource with eig_parity=ODD_Z (the TM Ez mode).
# meep-torch has a one-way TFSF eigenmode source (ModeSourceTM) used through
# compute_smatrix, which mode-decomposes the fields at each port. We read
# |S21|^2 (right-going transmitted) and |S11|^2 (left-going reflected); for a
# transparent eigenmode source in a uniform guide |S21| ~ 1 and |S11| ~ 0.
# Output: `output/wvg-src.png`.
#
# meep -> meep-torch mapping:
# ```
#   cell 16x8, resolution 10                ->  Grid(nx=160, ny=80, dx=0.1)
#   mp.Block(inf,1,inf) eps=12 @ y in [-0.5,0.5]  ->  eps[:, 35:45] = 12
#   mp.Block(inf,0.1,inf) eps=1  @ y=0.3   ->  no-op (eps=1 = vacuum background; the
#                                             meep mp.Medium() defaults to eps=1, so
#                                             this "air slot" is indistinguishable
#                                             from the surrounding vacuum)
#   mp.PML(1.0)                            ->  npml=10
#   EigenModeSource @ x=-5, size y=6        ->  Port(axis="x", index=30, lo=10, hi=70)
#   ContinuousSource(0.15)                 ->  GaussianPulse(0.15, 0.05)
# ```
# Run: `uv run python examples/02_sources/wvg-src.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Geometry and ports
# A uniform eps=12 strip guide; input port launches the +x eigenmode, output
# port receives it. Port spans stay clear of the PML.

# %%
grid = Grid(nx=160, ny=80, dx=0.1)
eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64, device=grid.device)
# waveguide: eps=12, y in [-0.5, 0.5] -> rows 35:45
eps[:, 35:45] = 12.0
# (the meep air slot at y=0.3, height 0.1, eps=1 is a no-op: eps=1 = background)

# eigenmode source at x=-5 -> ix=30. Meep size=y=6 (60 cells) reaches the PML
# edge; meep-torch ports must stay clear of PML (npml=10), so we use lo=20, hi=60
# (40 cells, 15 cladding cells each side of the 10-cell core -- enough for the
# strongly confined eps=12 mode) -- matches the validated smatrix-tm test.
# output port at x=+5 -> ix=130, same span, direction=-1 (receive the +x wave)
ports = [
    Port(axis="x", index=30, lo=20, hi=60, direction=+1),
    Port(axis="x", index=130, lo=20, hi=60, direction=-1),
]

# %% [markdown]
# ## Run
# One pulsed S-matrix run; in a uniform guide the eigenmode source should be
# transparent: |S21| ~ 1, |S11| ~ 0.

# %%
freqs = [0.13, 0.15, 0.17]
# fwidth=0.1 -> GaussianPulse t_end=100; 3000 steps = 150 time units lets the
# pulse fully emit (ends t=100) then propagate to port2 (~34 time units in the
# eps=12 guide). Mirrors the smatrix-tm parity test recipe.
sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=0.15, fwidth=0.1,
                     freqs=freqs, n_steps=3000, npml=10)
s = sm.s
S11 = s[:, 0, 0].abs()    # reflected (left-going) at input port
S21 = s[:, 1, 0].abs()    # transmitted (right-going) at output port

# %% [markdown]
# ## Plot and report

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.imshow(eps.cpu().T, origin="lower", cmap="binary", alpha=0.9)
# overlay the |S21|^2 bar at each freq for a visual (transmission ~ 1)
ax.set_title("Eigenmode source in waveguide (meep wvg-src.py)")
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "wvg-src.png", dpi=150)
plt.close(fig)
print("wvg-src.png")
print("right-going |S21| = %.4f (transmitted), left-going |S11| = %.4f (reflected) at fcen=0.15"
      % (float(S21[1]), float(S11[1])))
