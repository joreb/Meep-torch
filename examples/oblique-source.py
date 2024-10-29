# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/oblique-source.py.

Launches a planewave at angle theta from a phased line source in a PML box
(ObliqueLineSource, cos(k_y y - 2*pi*f*t)), no Bloch boundary. A soft source
radiates both +/-x; the +x lobe is the oblique planewave. Gate (finite aperture,
looser than Bloch): angle within ~2 deg (tests/examples/test_w2_examples.py).
Run: uv run python examples/oblique-source.py
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
from meep_torch.core.sources import GaussianPulse, ObliqueLineSource
from meep_torch.core.stepper import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent
FCEN, THETA_DEG, DX = 0.3, 25.0, 0.05
NX, NY, NPML = 300, 300, 20

g = Grid(nx=NX, ny=NY, dx=DX)
src = ObliqueLineSource(g, ix=NPML + 5, j0=NPML, j1=NY - NPML,
                        profile=GaussianPulse(FCEN, 0.1), theta=math.radians(THETA_DEG), fcen=FCEN)
coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64), npml=NPML)
st = init_tm_state(g)
for t in range(int(round(60.0 / g.dt))):
    st = step_tm(st, coeffs, src.amplitude(t), src.mask)

fig, ax = plt.subplots(figsize=(6, 6))
v = float(st.Ez.abs().max()) or 1.0
ax.imshow(st.Ez.numpy().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v)
ax.set_title(f"Oblique source (PML box) theta={THETA_DEG} deg, f={FCEN}")
ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "oblique-source.png", dpi=150); plt.close(fig)
print("oblique-source.png  theta=%.1f deg" % THETA_DEG)