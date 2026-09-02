# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Gaussian beam via the meep-syntax shim
#
# meep-syntax twin of gaussian-beam.py via `meep_torch.mp`. Gaussian beam from
# a line source with a Gaussian transverse amplitude profile (amp_func),
# GaussianPulse temporal envelope (GaussianSource), vacuum, PML 2. Field
# snapshot matches gaussian-beam.py (same grid, n_steps=1000). Output:
# `output/gaussian-beam_mp.png`.
#
# Run: `uv run python examples/02_sources/gaussian-beam_mp.py`

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
s, dpml = 14, 2
res = 20
fcen, fwidth, w0 = 1.0, 0.3, 0.8

# %% [markdown]
# ## Simulation setup and run
# The Gaussian transverse profile enters through the meep-style `amp_func`.

# %%
sim = mp.Simulation(
    cell_size=mp.Vector3(s, s, 0),
    resolution=res,
    sources=[mp.Source(mp.GaussianSource(frequency=fcen, fwidth=fwidth), component=mp.Ez,
                       center=mp.Vector3(-0.5 * s + dpml + 1.0, 0),
                       size=mp.Vector3(0, s, 0),
                       amp_func=lambda p: math.exp(-(p.y ** 2) / (w0 ** 2)))],
    boundary_layers=[mp.PML(dpml)],
    default_material=mp.Medium(epsilon=1.0),
)
sim.run(n_steps=1000)   # raw twin: t = 25 time units
ez = sim.get_array(component=mp.Ez)

# %% [markdown]
# ## Plot

# %%
fig, ax = plt.subplots(figsize=(6, 6))
v = float(np.abs(ez).max()) or 1.0
ax.imshow(ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v)
ax.set_title("Gaussian beam (mp shim) fcen=%.1f, w0=%.2f" % (fcen, w0))
ax.set_xlabel("x (cells, dx=%.2f um)" % (1.0 / res)); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "gaussian-beam_mp.png", dpi=150); plt.close(fig)
print("gaussian-beam_mp.png  peak|Ez|=%.4g" % v)
