# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/oblique-planewave.py.

A planewave at angle theta in a homogeneous Bloch-periodic cell (y-periodic with
k_y = 2*pi*f*sin(theta)), launched as the envelope e = exp(-i w t) (two real
quadratures). The physical field E = e*exp(i k_y y) propagates at theta. Gate:
propagation angle matches theta within 1 deg (tests/examples/test_w2_bloch.py).
Run: uv run python examples/oblique-planewave.py
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
from meep_torch.core.stepper_bloch import init_bloch_tm_state, step_bloch_tm

OUT = pathlib.Path(__file__).parent
FCEN, THETA_DEG, DX = 0.3, 20.0, 0.05
K0 = 2 * math.pi * FCEN
KY = K0 * math.sin(math.radians(THETA_DEG))
LY = 2 * math.pi / KY
NX, NY, NPML = 320, int(round(LY / DX)), 20

g = Grid(nx=NX, ny=NY, dx=DX)
coeffs = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64), npml=NPML, y_pmc=True)
st = init_bloch_tm_state(g)
ix = NPML + 5
mask = g.zeros(g.nx, g.ny); mask[ix, :] = 1.0
omega = K0
t_ramp = 5.0 / FCEN
for t in range(int(round(40.0 / g.dt))):
    r = min(1.0, (t + 0.5) * g.dt / t_ramp)
    st = step_bloch_tm(st, coeffs, KY, src_c=math.cos(omega * (t + 0.5) * g.dt) * r,
                       src_s=-math.sin(omega * (t + 0.5) * g.dt) * r, src_mask=mask)

y = (torch.arange(NY, dtype=torch.float64) + 0.5) * DX
Ephys = (st.Ez_c + 1j * st.Ez_s).numpy() * np.exp(1j * KY * y.numpy())[None, :]
fig, ax = plt.subplots(figsize=(7, 4))
ax.imshow(np.real(Ephys).T, origin="lower", cmap="RdBu")
ax.set_title(f"Oblique planewave (Bloch) theta={THETA_DEG} deg, f={FCEN}")
ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")
fig.tight_layout(); fig.savefig(OUT / "oblique-planewave.png", dpi=150); plt.close(fig)
print("oblique-planewave.png  theta=%.1f deg" % THETA_DEG)