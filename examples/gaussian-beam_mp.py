# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-syntax twin of examples/gaussian-beam.py via `meep_torch.mp`.

Gaussian beam from a line source with a Gaussian transverse amplitude profile
(amp_func), GaussianPulse temporal envelope (GaussianSource), vacuum, PML 2.
Field snapshot matches examples/gaussian-beam.py (same grid, n_steps=1000).
Run: uv run python examples/gaussian-beam_mp.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent
s, dpml = 14, 2
res = 20
fcen, fwidth, w0 = 1.0, 0.3, 0.8

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

fig, ax = plt.subplots(figsize=(6, 6))
v = float(np.abs(ez).max()) or 1.0
ax.imshow(ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v)
ax.set_title("Gaussian beam (mp shim) fcen=%.1f, w0=%.2f" % (fcen, w0))
ax.set_xlabel("x (cells, dx=%.2f um)" % (1.0 / res)); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "gaussian-beam_mp.png", dpi=150); plt.close(fig)
print("gaussian-beam_mp.png  peak|Ez|=%.4g" % v)