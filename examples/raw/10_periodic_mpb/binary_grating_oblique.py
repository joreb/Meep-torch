# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Binary grating at oblique incidence: diffraction orders
#
# A planewave hits the glass binary grating at theta = 10.7 deg *inside the
# glass*, handled properly with a Bloch-periodic cell of one grating period
# (transverse wavevector ky = k0 n sin(theta)). The transmitted field on a
# DFT line is projected onto planewave diffraction orders, whose angles
# follow the grating equation.

# %%
"""meep-torch port of meep/python/examples/binary_grating_oblique.py.

Glass (n=1.5) substrate + binary grating (gp=10, gh=0.5, gdc=0.5), oblique
incidence theta=10.7deg in glass (Bloch cell = one period), transmitted into
air. Diffraction orders via planewave projection (ports/orders). Resolution
reduced 50 -> 25 for runtime. Gate: transmitted orders vs meep-oracle JSON
(tests/examples/test_w2_binary_grating.py).
Run: uv run python examples/10_periodic_mpb/binary_grating_oblique.py
"""
import json
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
from meep_torch.ports.orders import BlochDFTLine, diffraction_orders

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
# %% [markdown]
# ## Bloch cell, CW ramp, and order projection
#
# The complex Bloch stepper carries ky through the periodic y boundary; a
# ramped CW source at f = 2 runs 120 time units, then `diffraction_orders`
# converts the accumulated DFT line into per-order powers and angles.

# %%
NG, GP, GH, GDC = 1.5, 10.0, 0.5, 0.5
DPML, DSUB, DPAD = 1.0, 3.0, 3.0
THETA = math.radians(10.7)
FCEN = 1.0 / 0.5
DX = 1.0 / 25
K0 = 2 * math.pi * FCEN
KY = K0 * NG * math.sin(THETA)
G = 2 * math.pi / GP
SX = DPML + DSUB + GH + DPAD + DPML

nx, ny = int(round(SX / DX)), int(round(GP / DX))
g = Grid(nx=nx, ny=ny, dx=DX, courant=0.3)
npml = int(round(DPML / DX))
eps = torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device)
sx1 = int(round((DPML + DSUB) / DX))
eps[:sx1, :] = NG * NG
gr0 = sx1
gr1 = int(round((DPML + DSUB + GH) / DX))
ty0 = int(round((GP * (1 - GDC) / 2) / DX))
ty1 = int(round((GP * (1 + GDC) / 2) / DX))
eps[gr0:gr1, ty0:ty1] = NG * NG
c = build_tm_coeffs(g, eps_inf=eps, npml=npml, y_pmc=True)
st = init_bloch_tm_state(g)
ix_src = int(round((DPML + 0.3 * DSUB) / DX))
ix_mon = nx - npml - int(round(0.5 * DPAD / DX))
mask = g.zeros(g.nx, g.ny); mask[ix_src, :] = 1.0
mon = BlochDFTLine(g, ix_mon, K0)
t_ramp = 8.0 / FCEN
n_steps = int(round(120.0 / g.dt))
for t in range(n_steps):
    tt = (t + 0.5) * g.dt
    r = min(1.0, tt / t_ramp)
    st = step_bloch_tm(st, c, KY, src_c=math.cos(K0 * tt) * r,
                       src_s=-math.sin(K0 * tt) * r, src_mask=mask)
    mon.sample(st, t)

orders, angles, powers = diffraction_orders(mon.acc, KY, G, K0, 1.0, DX, n_modes=12)
ps = powers / max(powers.sum(), 1e-30)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(angles, ps, width=1.5)
ax.set_xlabel("transmitted order angle (deg)")
ax.set_ylabel("normalized power")
ax.set_title("Binary grating oblique orders (theta=10.7deg, gp=10)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "binary_grating_oblique.png", dpi=150); plt.close(fig)
dom = sorted(zip(angles, ps), key=lambda kv: -kv[1])[:4]
print("binary_grating_oblique.png  dominant orders:", [(round(a, 1), round(p, 3)) for a, p in dom])