# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Launching a single diffraction order
#
# Instead of a planewave, the source *is* one diffraction order: a complex
# spatial mask e^{i m G y} on a Bloch-periodic line injects only order m = 1.
# A vacuum run verifies launch purity, then the same order is sent through a
# binary grating and the transmitted field decomposed back into orders.

# %%
"""meep-torch port of meep/python/examples/diffracted_planewave.py.

Launch a SINGLE diffraction order as the source: a Bloch-periodic cell
with a complex spatial source mask e^{i m G y} injects only order m; the
transmitted field through a binary grating is decomposed back into
orders (W2 bloch_diffraction_orders).

SUBSTITUTIONS: mp.DiffractedPlanewave -> complex source mask in
step_bloch_tm (src_mask_s); grating period 3, lambda 1 (orders 0, +-1,
+-2 propagating); vacuum purity check replaces meep's eigenmode
normalization.

Run: uv run python examples/10_periodic_mpb/diffracted_planewave.py
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
from meep_torch.core.stepper_bloch import (bloch_diffraction_orders,
                                           init_bloch_tm_state, step_bloch_tm)

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, RES, GP = 1.0, 20, 3.0
M_LAUNCH = 1
G = 2 * math.pi / GP


# %% [markdown]
# ## One run per case (vacuum, grating)
#
# The cos/sin mask pair feeds the complex Bloch stepper's src_mask /
# src_mask_s channels; after the CW ramp the field at the output line is
# projected onto orders m = -1, 0, +1.

# %%
def run(with_grating):
    dx = 1.0 / RES
    sx = 12.0
    nx, ny = int(sx * RES), int(GP * RES)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    eps = torch.ones(nx, ny, dtype=torch.float64, device=grid.device)
    if with_grating:
        i0 = int(6.0 * RES)
        eps[i0:i0 + RES // 2, : ny // 2] = 2.25   # binary glass grating
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES, y_pmc=True)
    st = init_bloch_tm_state(grid)
    ys = (torch.arange(ny, dtype=torch.float64, device=grid.device) + 0.5) * dx
    line = torch.zeros(nx, ny, dtype=torch.float64, device=grid.device)
    line[int(2.0 * RES), :] = 1.0
    mask_c = line * torch.cos(M_LAUNCH * G * ys)[None, :]
    mask_s = line * torch.sin(M_LAUNCH * G * ys)[None, :]
    omega = 2 * math.pi * FCEN
    t_ramp = 5.0
    for t in range(int(round(80.0 / grid.dt))):
        tt = (t + 0.5) * grid.dt
        ramp = min(1.0, tt / t_ramp)
        st = step_bloch_tm(st, coeffs, 0.0,
                           src_c=math.cos(omega * tt) * ramp,
                           src_s=-math.sin(omega * tt) * ramp,
                           src_mask=mask_c, src_mask_s=mask_s)
    orders, powers = bloch_diffraction_orders(
        st, nx - int(2.0 * RES), 0.0, G, 2 * math.pi * FCEN, dx, n_modes=3)
    return orders, powers / powers.sum()


orders, p_vac = run(False)
_, p_grat = run(True)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
w = 0.35
ax.bar(np.array(orders) - w / 2, p_vac, w, label="vacuum (launch purity)")
ax.bar(np.array(orders) + w / 2, p_grat, w, label="through grating")
ax.set_xlabel("diffraction order m")
ax.set_ylabel("relative power")
ax.set_title(f"single-order launch (m={M_LAUNCH}), period {GP}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "diffracted_planewave.png", dpi=150)
plt.close(fig)
purity = p_vac[orders.index(M_LAUNCH)]
print(f"diffracted_planewave.png  launch purity m={M_LAUNCH}: {purity:.3f}; "
      f"grating orders: {dict(zip(orders, np.round(p_grat, 3)))}")
