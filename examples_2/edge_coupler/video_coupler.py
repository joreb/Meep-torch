# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Edge coupler propagation video
#
# Example (c1b): video of the astigmatic laser beam coupling into the
# inverse-designed edge coupler.
#
# Rebuilds the final design from c_edge_coupler.py's saved state
# (`output/c_edge_coupler_rho.npy` + the script's geometry constants, imported
# by executing its setup head), then drives the CW astigmatic beam
# (4.0 um lateral waist) and animates Ez -> `output/c_coupler_propagation.mp4`.
#
# Run after c_edge_coupler.py:
# `uv run python examples_2/edge_coupler/video_coupler.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(16)
HERE = pathlib.Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Rebuild the optimized design
# Exec c_edge_coupler.py's setup head (everything before its split marker) to
# get the grid, base eps, design region, and fiber source exactly as
# optimized, then repaint the saved rho.

# %%
# execute the coupler script's setup (everything before the normalization
# runs) to get grid/eps_base/region/filters exactly as optimized
src = open(HERE / "c_edge_coupler.py").read()
marker = "# --- normalization" if "# --- normalization" in src else "def objective"
head = src.split(marker)[0]
g = {"__file__": str(HERE / "c_edge_coupler.py")}
exec(compile(head, "coupler_head", "exec"), g)

from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm

grid, eps_base, region = g["grid"], g["base"], g["region"]
dx, RES, FCEN, NPML = g["dx"], g["RES"], g["FCEN"], g["NPML"]
rho = torch.from_numpy(np.load(OUT / "c_edge_coupler_rho.npy"))
r = tanh_projection(conic_filter(rho, g.get("FILTER_R", 0.12), dx), 64.0)
eps_fin = region.paint(eps_base, r).detach()
nx, ny = grid.nx, grid.ny

# %% [markdown]
# ## Drive the CW beam and collect frames
# CW astigmatic source at FCEN, stepping the TM fields and snapshotting |Ez|
# every few steps for ~170 frames.

# %%
coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
cw = ContinuousSource(FCEN, width=8.0)
fib = g["fiber_source"](coeffs)
st = init_tm_state(grid)
frames = []
n_steps = int(round(140.0 / grid.dt))
every = max(1, n_steps // 170)
for t in range(n_steps):
    a = cw.amplitude((t + 0.5) * grid.dt)
    st = step_tm(st, coeffs, a, fib.mask)
    if t % every == 0:
        frames.append(st.Ez.abs().cpu().numpy())

# %% [markdown]
# ## Animate
# |Ez| frames over the design outline, saved as an mp4 (ffmpeg writer).

# %%
vmax = max(f.max() for f in frames[len(frames) // 2:]) * 0.6
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=0,
               vmax=vmax, extent=[0, nx * dx, 0, ny * dx])
core = (eps_fin > 0.5 * (float(eps_fin.min()) + float(eps_fin.max()))).cpu().numpy()
ax.imshow(np.ma.masked_where(~core, core).T, origin="lower", cmap="gray",
          alpha=0.22, extent=[0, nx * dx, 0, ny * dx])   # in/out guides visible
ax.contour(np.linspace(0, nx * dx, nx), np.linspace(0, ny * dx, ny),
           core.T.astype(float), levels=[0.5], colors="k", linewidths=0.8)
ax.set_title("edge coupler: astigmatic beam (4.0 um) -> SiN guide, $E_z$")
ax.set_xlabel("x [um]")


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


anim.FuncAnimation(fig, upd, frames=len(frames), interval=40).save(
    OUT / "c_coupler_propagation.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("c_coupler_propagation.mp4 written")
