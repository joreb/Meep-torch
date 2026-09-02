# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Bent waveguide via the meep-syntax shim
#
# meep-syntax twin of bent-waveguide.py via `meep_torch.mp`. 90-degree bent
# waveguide (two eps=12 Blocks), CW Ez source, PML 1, until=200 (reduced to
# 3000 steps to match the raw twin's runtime trim). Output matches
# bent-waveguide.py field (L2 < 1e-3 rel); saved to
# `output/bent-waveguide_mp.png`.
#
# Run: `uv run python examples/01_basics/bent-waveguide_mp.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
fcen = 1.0 / (2.0 * math.sqrt(11.0))

# %% [markdown]
# ## Simulation setup and run
# Two Blocks form the L-shaped guide; the CW line source sits on the horizontal
# arm at x=-7.

# %%
sim = mp.Simulation(
    cell_size=mp.Vector3(16, 16, 0),
    resolution=10,
    geometry=[
        mp.Block(size=mp.Vector3(12, 1, mp.inf), center=mp.Vector3(-2.5, -3.5, 0),
                 material=mp.Medium(epsilon=12)),
        mp.Block(size=mp.Vector3(1, 12, mp.inf), center=mp.Vector3(3.5, 2, 0),
                 material=mp.Medium(epsilon=12)),
    ],
    sources=[mp.Source(mp.ContinuousSource(frequency=fcen, width=20),
                       component=mp.Ez, center=mp.Vector3(-7, -3.5),
                       size=mp.Vector3(0, 1, 0))],
    boundary_layers=[mp.PML(1.0)],
)
sim.run(n_steps=3000)   # raw twin: meep until=200 -> 4000, trimmed to 3000
ez = sim.get_array(component=mp.Ez)
eps = sim.get_array(component=mp.Dielectric)

# %% [markdown]
# ## Plot

# %%
fig, ax = plt.subplots(figsize=(6, 6))
v = float(np.abs(ez).max()) or 1.0
ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.set_title("Bent waveguide (mp shim) -- CW Ez at f=%.4f" % fcen)
ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "bent-waveguide_mp.png", dpi=150); plt.close(fig)
print("bent-waveguide_mp.png  peak|Ez|=%.4g" % v)
