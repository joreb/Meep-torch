# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a): video of the wave propagating in the ring -- CW drive at
the deepest resonance, Ez frames -> mp4.

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
from meep_torch.core.sources import ContinuousSource, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()
RES = 20
F0 = 1 / 1.5846                                    # deepest resonance from (a) redo (RES 30 mode-overlap scan)

c = gf.components.ring_single(radius=5.0, gap=0.2)
grid, eps, ports = component_to_eps(c, res=RES, pad=1.5, n_core=2.0, n_clad=1.44)
jbus = ports["o1"][1]
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES)
src = LineSource(grid, ix=ports["o1"][0] + 8, j0=jbus - 8, j1=jbus + 8,
                 profile=ContinuousSource(F0, width=8.0))
state = init_tm_state(grid)

frames = []
n_steps = int(round(180.0 / grid.dt))
every = max(1, n_steps // 240)                     # ~240 frames
for t in range(n_steps):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)
    if t % every == 0:
        frames.append(state.Ez.numpy().copy())

vmax = max(np.abs(f).max() for f in frames[len(frames) // 2:]) * 0.6
fig, ax = plt.subplots(figsize=(7.6, 5))
bg = ax.imshow(eps.numpy().T, origin="lower", cmap="binary", alpha=0.9)
im = ax.imshow(frames[0].T, origin="lower", cmap="RdBu",
               vmin=-vmax, vmax=vmax, alpha=0.75)
ax.set_title(f"SiN ring at resonance ({1e3/F0:.0f} nm): $E_z$")
ax.set_xticks([]); ax.set_yticks([])


def update(i):
    im.set_data(frames[i].T)
    return (im,)


ani = anim.FuncAnimation(fig, update, frames=len(frames), interval=33)
ani.save(OUT / "a_ring_propagation.mp4", writer=anim.FFMpegWriter(fps=30))
plt.close(fig)
print(f"a_ring_propagation.mp4 written ({len(frames)} frames)")
