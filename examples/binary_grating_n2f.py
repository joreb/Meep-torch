# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/binary_grating_n2f.py.

Far-field diffraction of a binary glass grating (period 10, height 0.5,
duty 0.5) under normal incidence: one periodic unit cell (Bloch TM,
ky = 0), near-field line above the grating, W4.5 tiling to nperiods
copies, W4 far fields -> angular intensity with lobes at the grating
orders sin(theta) = m lambda / period.

SUBSTITUTIONS:
  * meep unit-cell add_near2far(nperiods=10) -> near2far.tile_faces.
  * Single frequency (fcen) instead of the meep spectrum sweep.

Run: uv run python examples/binary_grating_n2f.py
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
from meep_torch.core.near2far import Face, farfield_tm, tile_faces
from meep_torch.core.stepper_bloch import init_bloch_tm_state, step_bloch_tm

OUT = pathlib.Path(__file__).parent

GP, GH, GDC = 10.0, 0.5, 0.5      # grating period / height / duty cycle
NPER = 10
FCEN = 0.2                        # lambda = 5
N_GLASS = 1.5
RES = 10
DPML, DSUB, DPAD = 1.0, 2.0, 2.0

dx = 1.0 / RES
sx = DPML + DSUB + GH + DPAD + DPML
nx, ny = int(round(sx * RES)), int(round(GP * RES))
grid = Grid(nx=nx, ny=ny, dx=dx)

eps = torch.ones(nx, ny, dtype=torch.float64)
i0 = int((DPML + DSUB) * RES)
eps[:i0, :] = N_GLASS ** 2                        # substrate
j_lo = int((0.5 - GDC / 2) * ny)
j_hi = int((0.5 + GDC / 2) * ny)
eps[i0:i0 + int(GH * RES), j_lo:j_hi] = N_GLASS ** 2

coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=int(DPML * RES), y_pmc=True)
st = init_bloch_tm_state(grid)
mask = grid.zeros(nx, ny)
mask[int((DPML + 0.5 * DSUB) * RES), :] = 1.0     # planewave line in the substrate

omega = 2 * math.pi * FCEN
t_ramp = 5.0 / FCEN
ix_n2f = nx - int((DPML + 0.5 * DPAD) * RES)      # line above the grating
n_steps = int(round(120.0 / grid.dt))
acc_E = np.zeros(ny, dtype=complex)
acc_H = np.zeros(ny, dtype=complex)
for t in range(n_steps):
    tt = (t + 0.5) * grid.dt
    ramp = min(1.0, tt / t_ramp)
    st = step_bloch_tm(st, coeffs, 0.0,
                       src_c=math.cos(omega * tt) * ramp,
                       src_s=-math.sin(omega * tt) * ramp, src_mask=mask)
    if t >= n_steps // 2:                          # steady state only
        ph = np.exp(-2j * math.pi * FCEN * (t + 1.0) * grid.dt) * grid.dt
        e = (st.Ez_c[ix_n2f, :] + 1j * st.Ez_s[ix_n2f, :]).numpy()
        hyc = 0.5 * (st.Hy_c[ix_n2f - 1, :] + st.Hy_c[ix_n2f, :]).numpy()
        hys = 0.5 * (st.Hy_s[ix_n2f - 1, :] + st.Hy_s[ix_n2f, :]).numpy()
        acc_E += e * ph
        acc_H += (hyc + 1j * hys) * ph * np.exp(1j * math.pi * FCEN * grid.dt)

# one unit-cell face (x-normal, +x outward), then tile NPER periods
ys = (np.arange(ny) + 0.5) * dx
pts = np.stack([np.full(ny, ix_n2f * dx), ys], axis=1)
face = Face(pts=pts, normal=(+1, 0), dl=dx,
            E=acc_E[None, :], H=acc_H[None, :])
faces = tile_faces([face], NPER, GP, ky=0.0)

thetas = np.radians(np.linspace(-60, 60, 961))
R_FAR = 2000.0
cx = ix_n2f * dx
obs = np.stack([cx + R_FAR * np.cos(thetas), GP / 2 + R_FAR * np.sin(thetas)],
               axis=1)
E, H = farfield_tm(faces, [2 * math.pi * FCEN], obs)
inten = np.abs(E[0, :, 2]) ** 2
inten /= inten.max()

fig, ax = plt.subplots(figsize=(7.4, 4))
ax.semilogy(np.degrees(thetas), np.maximum(inten, 1e-8), lw=0.8)
lam = 1.0 / FCEN
for m in range(-2, 3):
    s = m * lam / GP
    if abs(s) < math.sin(math.radians(60)):
        ax.axvline(math.degrees(math.asin(s)), color="r", ls=":", lw=0.7)
ax.set_xlabel("far-field angle (deg)")
ax.set_ylabel("|Ez|^2 (normalized)")
ax.set_title(f"binary grating far field, {NPER} periods (orders dotted)")
fig.tight_layout()
fig.savefig(OUT / "binary_grating_n2f.png", dpi=150)
plt.close(fig)
pk = float(np.degrees(thetas[inten.argmax()]))
print(f"binary_grating_n2f.png  strongest lobe at {pk:.2f} deg "
      f"(orders at {[round(math.degrees(math.asin(m * lam / GP)), 1) for m in range(-2, 3) if abs(m * lam / GP) < 1]})")
