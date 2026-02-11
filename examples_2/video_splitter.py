# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2c): video of the Y splitter (674 nm, W=4, v5 design).

Rebuilds the optimized half-domain design from c_y_splitter.py (exec of
its setup head + c_y_splitter_rho.npy), drives the TE0 mode CW through
the PMC-mirror half domain, and renders the FULL device by mirroring
|Ez| about the symmetry plane (labeled). Waveguide cores shaded.

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
dx, RES, FCEN, NPML = g["dx"], g["RES"], g["FCEN"], g["NPML"]
p_in = g["p_in"]
nx, ny = grid.nx, grid.ny
rho = torch.from_numpy(np.load(OUT / "c_y_splitter_rho.npy")).to(eps_base.dtype)
r = tanh_projection(conic_filter(rho, 0.15, dx), 64.0)
eps_fin = region.paint(eps_base, r).detach().to(grid.device)

coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML, y_mirror=+1)
src = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(FCEN, width=10.0),
                   FCEN, bc="pmc_lo")
st = init_tm_state(grid, y_mirror=+1)
frames = []
n_steps = int(round(180.0 / grid.dt))
every = max(1, n_steps // 180)
for t in range(n_steps):
    h = getattr(src, "h_masks", None)
    if h:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask,
                     src.h_amplitude(t), **h)
    else:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask)
    if t % every == 0:
        half = st.Ez.abs().cpu().numpy()
        frames.append(np.concatenate([half[:, ::-1], half], axis=1))  # mirror
vmax = max(f.max() for f in frames[len(frames) // 2:]) * 0.6
eps_half = eps_fin.cpu().numpy()
eps_full = np.concatenate([eps_half[:, ::-1], eps_half], axis=1)
core = eps_full > 0.5 * (eps_full.min() + eps_full.max())
ext = [0, nx * dx, -ny * dx, ny * dx]
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
               extent=ext, aspect="auto")
ax.imshow(np.ma.masked_where(~core, core).T, origin="lower", cmap="gray",
          alpha=0.25, extent=ext, aspect="auto")
ax.set_title("Y splitter 674 nm, CW TE0: |Ez| "
             "(half-domain sim, mirrored about the symmetry plane)")
ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


anim.FuncAnimation(fig, upd, frames=len(frames), interval=40).save(
    OUT / "c_splitter_propagation.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("c_splitter_propagation.mp4 written")
