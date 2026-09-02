# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Straight waveguide via the meep-syntax shim
#
# meep-syntax twin of straight-waveguide.py via `meep_torch.mp`. Runs the same
# straight-waveguide (CW Ez, eps=12 guide, PML 1) through the W1 meep-shaped
# shim. `size=Vector3(0,1,0)` spans the 1um guide height (the meep original's
# point source couples into the guide; the raw twin uses a line source across
# the guide for clean excitation -- matched here). Output matches
# straight-waveguide.py (field L2 < 1e-3 rel); saved to
# `output/straight-waveguide_mp.png`.
#
# Run: `uv run python examples/01_basics/straight-waveguide_mp.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Simulation setup and run
# The full meep-style Simulation object: cell, Block geometry, CW source, PML —
# then `run(until=200)` exactly as in the meep tutorial.

# %%
sim = mp.Simulation(
    cell_size=mp.Vector3(16, 8, 0),
    resolution=10,
    geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf), center=mp.Vector3(),
                       material=mp.Medium(epsilon=12))],
    sources=[mp.Source(mp.ContinuousSource(frequency=0.15), component=mp.Ez,
                       center=mp.Vector3(-7, 0), size=mp.Vector3(0, 1, 0))],
    boundary_layers=[mp.PML(1.0)],
    default_material=mp.Medium(epsilon=1.0),
)
sim.run(until=200)
ez = sim.get_array(component=mp.Ez)
eps = sim.get_array(component=mp.Dielectric)

# %% [markdown]
# ## Plot
# Same permittivity + Ez overlay as the raw twin.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
v = float(np.abs(ez).max()) or 1.0
ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.set_title("Straight waveguide (mp shim) -- CW Ez at f=0.15")
ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "straight-waveguide_mp.png", dpi=150); plt.close(fig)
print("straight-waveguide_mp.png  peak|Ez|=%.4g" % v)
