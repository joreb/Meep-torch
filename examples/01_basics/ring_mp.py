# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Ring resonator via the meep-syntax shim
#
# meep-syntax twin of ring.py via `meep_torch.mp`. 2D ring resonator (two
# overlapping Cylinders: outer index n, inner air), a GaussianSource point
# excitation, PML 2. Field snapshot matches ring.py (same 64-pt N-gon via
# Cylinder(n_pts=64), same n_steps). Resonant-frequency extraction (Harminv)
# is W3; this twin reproduces the field. Output: `output/ring_mp.png`.
#
# Run: `uv run python examples/01_basics/ring_mp.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
n, w, r, pad, dpml = 3.4, 1.0, 1.0, 4.0, 2.0
sxy = 2 * (r + w + pad + dpml)
fcen, df = 0.15, 0.1

# %% [markdown]
# ## Simulation setup and run
# Two overlapping Cylinders (painter's order) form the annulus; a Gaussian
# pulse just outside the ring edge excites the modes, then rings down.

# %%
sim = mp.Simulation(
    cell_size=mp.Vector3(sxy, sxy, 0),
    resolution=10,
    geometry=[
        mp.Cylinder(radius=r + w, center=mp.Vector3(), material=mp.Medium(index=n), n_pts=64),
        mp.Cylinder(radius=r, center=mp.Vector3(), material=mp.Medium(epsilon=1.0), n_pts=64),
    ],
    sources=[mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df), component=mp.Ez,
                       center=mp.Vector3(r + 0.1, 0))],
    boundary_layers=[mp.PML(dpml)],
    default_material=mp.Medium(epsilon=1.0),
)
sim.run(n_steps=8000)   # raw twin: (t_end=100 + 300 ringdown)/dt = 8000
ez = sim.get_array(component=mp.Ez)
eps = sim.get_array(component=mp.Dielectric)

# %% [markdown]
# ## Plot

# %%
fig, ax = plt.subplots(figsize=(6, 6))
v = float(np.abs(ez).max()) or 1.0
ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.5)
ax.imshow(ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.set_title("Ring resonator (mp shim) -- Ez ringdown")
ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "ring_mp.png", dpi=150); plt.close(fig)
print("ring_mp.png  peak|Ez|=%.4g" % v)
