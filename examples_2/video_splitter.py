# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2c): videos of the inverse-designed Y splitter.

1. c_splitter_propagation.mp4 -- 2D CW TE0 mode drive, Ez turn-on
   transient through the optimized design (lateral effective index).
2. c_splitter_3d.mp4 -- TRUE-3D CW run (SiN n=1.9963, 150 nm slab),
   |Ey| in the core plane while the mode splits.

Rebuilds the committed design from c_y_splitter_rho.npy; geometry is
the gdsfactory mmi1x2_with_sbend import throughout.

Run: uv run python examples_2/video_splitter.py
"""
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(16)

sys.path.insert(0, str(pathlib.Path(__file__).parent))

OUT = pathlib.Path(__file__).parent

# reuse the exact rebuild path of the 3D validation script
src = open(OUT / "c_y_splitter_3d.py").read()
head = src.split("PULSE = GaussianPulse")[0]
g = {"__file__": str(OUT / "c_y_splitter_3d.py")}
exec(compile(head, "rebuild", "exec"), g)
eps2d, eps3, grid2 = g["eps2d"], g["eps3"], g["grid2"]
o1x, o1y = g["o1x"], g["o1y"]
RES2, RES3, FCEN = g["RES2"], g["RES3"], g["FCEN"]
NPML3 = g["NPML"]

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.monitors_3d import run_3d
from meep_torch.core.run import run_tm
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port
from meep_torch.ports.validate3d import ModePlaneSourceX

# --- 1: 2D propagation ---
nx, ny = grid2.nx, grid2.ny
dx = 1.0 / RES2
NPML2 = RES2
SPAN = int(1.8 * RES2)
p_in = Port(axis="x", index=max(o1x + 6, NPML2 + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
coeffs2 = build_tm_coeffs(grid2, eps_inf=eps2d, npml=NPML2)
src2 = ModeSourceTM(grid2, coeffs2, p_in, ContinuousSource(FCEN, width=8.0),
                    FCEN)
st = init_tm_state(grid2)
frames = []
n_steps = int(round(150.0 / grid2.dt))
every = max(1, n_steps // 180)
for t in range(n_steps):
    h = getattr(src2, "h_masks", None)
    if h:
        st = step_tm(st, coeffs2, src2.amplitude(t), src2.mask,
                     src2.h_amplitude(t), **h)
    else:
        st = step_tm(st, coeffs2, src2.amplitude(t), src2.mask)
    if t % every == 0:
        frames.append(st.Ez.numpy().copy())
vmax = max(abs(f).max() for f in frames[len(frames) // 2:]) * 0.6
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(frames[0].T, origin="lower", cmap="RdBu", vmin=-vmax,
               vmax=vmax, extent=[0, nx * dx, 0, ny * dx])
core = (eps2d > 0.5 * (float(eps2d.min()) + float(eps2d.max()))).numpy()
ax.imshow(np.ma.masked_where(~core, core).T, origin="lower", cmap="gray",
          alpha=0.22, extent=[0, nx * dx, 0, ny * dx])   # waveguides visible
ax.contour(np.linspace(0, nx * dx, nx), np.linspace(0, ny * dx, ny),
           core.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
ax.set_title("Y splitter, CW 1550 nm (2D eff. index): $E_z$")
ax.set_xlabel("x [um]")


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


anim.FuncAnimation(fig, upd, frames=len(frames), interval=40).save(
    OUT / "c_splitter_propagation.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("c_splitter_propagation.mp4 written")

# --- 2: 3D core-plane video ---
n3x, n3y, n3z = eps3.shape
dx3 = 1.0 / RES3
c3 = build_3d_coeffs(n3x, n3y, n3z, dx3, eps=eps3, npml=NPML3)
IX_SRC = max(int(o1x * RES3 / RES2) + 6, NPML3 + 4)
src3 = ModePlaneSourceX(eps3, dx3, IX_SRC, FCEN,
                        ContinuousSource(FCEN, width=8.0), c3.dt)
st3 = init_3d_state(n3x, n3y, n3z, npml=NPML3)
kmid = n3z // 2
frames3 = []
n_steps3 = int(round(110.0 / c3.dt))
every3 = max(1, n_steps3 // 150)
comp, mask = src3.src
for t in range(n_steps3):
    a = src3.amplitude(t)
    st3 = step_3d(st3, c3, a, (comp, mask))
    if t % every3 == 0:
        frames3.append(st3.Ey[:, :, kmid].numpy().copy())
vmax3 = max(abs(f).max() for f in frames3[len(frames3) // 2:]) * 0.6
fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(frames3[0].T, origin="lower", cmap="RdBu", vmin=-vmax3,
               vmax=vmax3, extent=[0, n3x * dx3, 0, n3y * dx3])
core3 = (eps3[:, :, kmid] > 0.5 * (float(eps3.min()) + float(eps3.max()))).numpy()
ax.imshow(np.ma.masked_where(~core3, core3).T, origin="lower", cmap="gray",
          alpha=0.22, extent=[0, n3x * dx3, 0, n3y * dx3])
ax.contour(np.linspace(0, n3x * dx3, n3x), np.linspace(0, n3y * dx3, n3y),
           core3.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
ax.set_title("Y splitter TRUE 3D (SiN 150 nm): core-plane $E_y$")
ax.set_xlabel("x [um]")


def upd3(i):
    im.set_data(frames3[i].T)
    return (im,)


anim.FuncAnimation(fig, upd3, frames=len(frames3), interval=40).save(
    OUT / "c_splitter_3d.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("c_splitter_3d.mp4 written")
