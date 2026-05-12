# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Example (a3): video of light propagating in the SiN ring.

Full gdsfactory racetrack ring (r=50, W=1, LC=4, gap 0.2) at 674 nm,
CW TE0 mode drive on the bus AT the resonance found by the half-ring
extraction (674.54 nm): the ring charges over many round trips.
GPU (RES 25 for the video; the S-matrix numbers come from RES 50 runs).

Run: uv run python examples_2/video_ring.py
"""
import math
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

LAM_RES = 0.67454                                   # resonance from extraction
N_SIN = Si3N4.n(1 / 0.674).real
N_CLAD = SiO2.n(1 / 0.674).real
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 0.674, "te", k=1).neff[0])
RES = 25
dx = 1.0 / RES
DEV = "cuda" if torch.cuda.is_available() else "cpu"

xs = gf.get_cross_section("strip").copy(width=1.0)
c = gf.components.ring_single(radius=50.0, gap=0.2, length_x=4.0,
                              cross_section=xs)
grid_c, eps_c, ports = component_to_eps(c, res=RES, pad=2.0,
                                        n_core=NEFF, n_clad=N_CLAD)
nx, ny = grid_c.nx, grid_c.ny
grid = Grid(nx=nx, ny=ny, dx=dx, device=DEV)
eps = eps_c.to(DEV)
NPML = RES
print(f"video_ring  grid {nx}x{ny} on {DEV}, drive {LAM_RES*1e3:.2f} nm")

o1x, o1y, _ = ports["o1"]
SPAN = int(1.5 * RES)
p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(1 / LAM_RES, width=30.0),
                   1 / LAM_RES)
st = init_tm_state(grid)
frames = []
n_steps = int(round(1200.0 / grid.dt))              # several ring lifetimes
every = max(1, n_steps // 200)
sub = max(1, nx // 900)
for t in range(n_steps):
    h = getattr(src, "h_masks", None)
    if h:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask,
                     src.h_amplitude(t), **h)
    else:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask)
    if t % every == 0:
        frames.append(st.Ez[::sub, ::sub].abs().cpu().numpy())
vmax = max(f.max() for f in frames[len(frames) // 2:]) * 0.5
core = (eps_c[::sub, ::sub] > 0.5 * (float(eps_c.min()) + float(eps_c.max()))).numpy()
fig, ax = plt.subplots(figsize=(7.2, 7.0))
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
               extent=[0, nx * dx, 0, ny * dx])
ax.imshow(np.ma.masked_where(~core, core).T, origin="lower", cmap="gray",
          alpha=0.25, extent=[0, nx * dx, 0, ny * dx])
ax.set_title(f"SiN racetrack ring at resonance ({LAM_RES*1e3:.1f} nm): |Ez|")
ax.set_xlabel("x [um]")


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


anim.FuncAnimation(fig, upd, frames=len(frames), interval=40).save(
    OUT / "a_ring_propagation.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("a_ring_propagation.mp4 written")
