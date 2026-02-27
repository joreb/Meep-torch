# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2c): video of the v6 Y splitter (674 nm, full domain).

Rebuilds the optimized FULL-domain design from c_y_splitter.py (exec of
its setup head + c_y_splitter_rho.npy), drives the TE0 mode CW and
animates |Ez| over the whole device. Waveguide cores shaded + outlined
so the input/output guides stay visible under the field.

Run after c_y_splitter.py:  uv run python examples_2/video_splitter.py
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent

src_txt = open(OUT / "c_y_splitter.py").read()
head = src_txt.split("def s_params")[0]
g = {"__file__": str(OUT / "c_y_splitter.py")}
exec(compile(head, "splitter_head", "exec"), g)

from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.ports.mode_source import ModeSourceTM

grid, eps_base, region = g["grid"], g["eps_base"], g["region"]
dx, FCEN, NPML = g["dx"], g["FCEN"], g["NPML"]
p_in = g["p_in"]
nx, ny = grid.nx, grid.ny
rho = torch.from_numpy(np.load(OUT / "c_y_splitter_rho.npy")).to(eps_base.dtype)
r = tanh_projection(conic_filter(rho, 0.12, dx), 64.0)
eps_fin = region.paint(eps_base, r).detach().to(grid.device)

coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(FCEN, width=10.0), FCEN)
st = init_tm_state(grid)
frames = []
n_steps = int(round(120.0 / grid.dt))
every = max(1, n_steps // 180)
for t in range(n_steps):
    h = getattr(src, "h_masks", None)
    if h:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask,
                     src.h_amplitude(t), **h)
    else:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask)
    if t % every == 0:
        frames.append(st.Ez.abs().cpu().numpy())
vmax = max(f.max() for f in frames[len(frames) // 2:]) * 0.6
core = (eps_fin > 0.5 * (float(eps_fin.min()) + float(eps_fin.max()))).cpu().numpy()
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
               extent=[0, nx * dx, 0, ny * dx])
ax.imshow(np.ma.masked_where(~core, core).T, origin="lower", cmap="gray",
          alpha=0.22, extent=[0, nx * dx, 0, ny * dx])
ax.contour(np.linspace(0, nx * dx, nx), np.linspace(0, ny * dx, ny),
           core.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
ax.set_title("v6 Y splitter at 674 nm (full domain): $|E_z|$")
ax.set_xlabel("x [um]")


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


anim.FuncAnimation(fig, upd, frames=len(frames), interval=40).save(
    OUT / "c_splitter_propagation.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("c_splitter_propagation.mp4 written")
