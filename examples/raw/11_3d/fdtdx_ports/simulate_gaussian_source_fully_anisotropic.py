# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Full anisotropic tensor: beam walk-off (fdtdx port)
#
# The block's optic axis is rotated 45 deg in-plane, putting a genuine
# off-diagonal exy into the permittivity tensor: a beam entering the block
# walks off sideways because the Poynting vector is no longer parallel to k —
# the visual signature only a full tensor produces.

# %%
"""meep-torch rebuild of fdtdx simulate_gaussian_source_fully_anisotropic.py.

The anisotropic block's optic axis is rotated 45 deg in-plane, putting a
full off-diagonal exy into the tensor: a beam entering the block walks
off sideways (Poynting vector no longer parallel to k) -- the visual
signature only a FULL tensor produces. Video + printed walk-off angle.

Run: uv run python examples/11_3d/fdtdx_ports/simulate_gaussian_source_fully_anisotropic.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper_te import init_te_state, step_te

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
WL, RES = 1.55, 25
dx = 1.0 / RES
nx, ny = int(12.0 * RES), int(12.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)

no2, ne2 = 1.5 ** 2, 2.5 ** 2                     # strong birefringence
exx_v = (ne2 + no2) / 2
exy_v = (ne2 - no2) / 2                           # 45 deg optic axis
one = torch.ones(nx, ny, dtype=torch.float64, device=grid.device)
exx = one.clone(); eyy = one.clone(); exy = torch.zeros_like(one)
j_blk = int(7.0 * RES)
exx[:, :j_blk] = exx_v
eyy[:, :j_blk] = exx_v
exy[:, :j_blk] = exy_v
coeffs = build_te_coeffs(grid, eps_inf=one, npml=RES,
                         eps_tensor=(exx, exy, eyy))

# %% [markdown]
# ## Gaussian beam drive and walk-off measurement
#
# A Gaussian-profile CW line source launches the beam downward into the
# block; the walk-off gate compares the beam centroid at two depths inside
# the block.

# %%
st = init_te_state(grid)
prof = ContinuousSource(1 / WL, width=6.0)
j_src = ny - int(2.0 * RES)
sl = slice(nx // 2 - 2 * RES, nx // 2 + 2 * RES)
ys = torch.arange(sl.start, sl.stop, dtype=torch.float64, device=grid.device)
beam = torch.exp(-((ys - nx // 2) / (1.2 * RES)) ** 2)
frames = []
n_steps = int(round(30.0 / grid.dt))
every = max(1, n_steps // 150)
for t in range(n_steps):
    a = prof.amplitude((t + 0.5) * grid.dt)
    st = step_te(st, coeffs)
    st.Dx[sl, j_src] += grid.dt * a * beam
    st.Ex[sl, j_src] += grid.dt * a * beam
    if t % every == 0:
        frames.append(np.sqrt(st.Ex.cpu().numpy() ** 2))

vmax = max(f.max() for f in frames[len(frames) // 2:]) * 0.5
fig, ax = plt.subplots(figsize=(6, 5.6))
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
ax.axhline(j_blk, color="w", lw=0.8)
ax.set_title("full tensor (optic axis 45 deg): beam walk-off in the block")
ax.set_xticks([]); ax.set_yticks([])


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


ani = anim.FuncAnimation(fig, upd, frames=len(frames), interval=40)
ani.save(OUT / "simulate_gaussian_source_fully_anisotropic.mp4",
         writer=anim.FFMpegWriter(fps=25))
plt.close(fig)

# walk-off gate: beam centroid drift between two depths inside the block
f = frames[-1]
c1 = float(np.average(np.arange(f.shape[0]), weights=f[:, j_blk - 2 * RES] ** 2))
c2 = float(np.average(np.arange(f.shape[0]), weights=f[:, j_blk - 4 * RES] ** 2))
walk = math.degrees(math.atan2((c2 - c1) * dx, 2.0))
print(f"simulate_gaussian_source_fully_anisotropic  walk-off angle = "
      f"{walk:.1f} deg (nonzero = S not parallel to k);  mp4 written")
