# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Anisotropic block: polarization-dependent refraction (fdtdx port)
#
# A CW 1550 nm wave hits a block with refractive index (n_x, n_y) = (2, 1):
# the x-polarized run sees n = 2 (wavelength halves inside), the y-polarized
# run sees n = 1 (nothing changes). Side-by-side video panels; the measured
# in-block wavelength is the printed gate.

# %%
"""meep-torch rebuild of fdtdx simulate_gaussian_source_anisotropic.py.

CW 1550 nm source above an anisotropic block with refractive index
(n_x, n_y) = (2, 1): the x-polarized run sees n=2 inside the block
(wavelength halves), the y-polarized run sees n=1 (nothing changes).
Side-by-side video panels; the wavelength ratio inside the block is
measured and printed as the gate.

Run: uv run python examples/11_3d/fdtdx_ports/simulate_gaussian_source_anisotropic.py
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
from meep_torch.core.run import run_te
from meep_torch.core.grid import Grid
from meep_torch.core.sources import ContinuousSource, LineSource
from meep_torch.core.stepper_te import init_te_state, step_te

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
WL, RES = 1.55, 25
dx = 1.0 / RES
nx, ny = int(10.0 * RES), int(12.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
NPML = RES

one = torch.ones(nx, ny, dtype=torch.float64, device=grid.device)
exx = one.clone(); eyy = one.clone()
j_blk = int(6.0 * RES)
exx[:, :j_blk] = 4.0                              # n_x = 2 in the block
eyy[:, :j_blk] = 1.0                              # n_y = 1
eps6 = (exx, torch.zeros_like(one), eyy)


# %% [markdown]
# ## One TE run per polarization
#
# The diagonal eps tensor feeds `build_te_coeffs(eps_tensor=...)`; the line
# source injects Dx/Ex or Dy/Ey by hand so each run drives a single
# polarization.

# %%
def run_pol(pol):
    coeffs = build_te_coeffs(grid, eps_inf=one, npml=NPML,
                             eps_tensor=eps6)
    st = init_te_state(grid)
    prof = ContinuousSource(1 / WL, width=6.0)
    j_src = ny - int(2.0 * RES)
    frames = []
    n_steps = int(round(26.0 / grid.dt))
    every = max(1, n_steps // 130)
    for t in range(n_steps):
        a = prof.amplitude((t + 0.5) * grid.dt)
        st = step_te(st, coeffs)
        # manual polarized line injection (Ex or Ey)
        sl = slice(nx // 2 - 3 * RES, nx // 2 + 3 * RES)
        if pol == "y":
            st.Dy[sl, j_src] += grid.dt * a
            st.Ey[sl, j_src] += grid.dt * a / eyy[sl, j_src]
        else:
            # inject D (persistent) AND E (seen by this step's Hz curl)
            st.Dx[sl, j_src] += grid.dt * a
            st.Ex[sl, j_src] += grid.dt * a / exx[sl, j_src]
        if t % every == 0:
            f = st.Ey if pol == "y" else st.Ex
            frames.append(f.cpu().numpy().copy())
    return frames


# %% [markdown]
# ## Side-by-side video and wavelength gate

# %%
fx = run_pol("x")
fy = run_pol("y")
n_frames = min(len(fx), len(fy))
vmx = max(np.abs(f).max() for f in fx[n_frames // 2:]) * 0.4
vmy = max(np.abs(f).max() for f in fy[n_frames // 2:]) * 0.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.6))
ims = []
for ax, fr, vm, ttl in ((ax1, fx, vmx, "x-pol: sees $n_x$=2"),
                        (ax2, fy, vmy, "y-pol: sees $n_y$=1")):
    im = ax.imshow(fr[0].T, origin="lower", cmap="RdBu", vmin=-vm, vmax=vm)
    ax.axhline(j_blk, color="k", lw=0.8)
    ax.set_title(ttl)
    ax.set_xticks([]); ax.set_yticks([])
    ims.append(im)


def upd(i):
    ims[0].set_data(fx[i].T)
    ims[1].set_data(fy[i].T)
    return ims


ani = anim.FuncAnimation(fig, upd, frames=n_frames, interval=40)
ani.save(OUT / "simulate_gaussian_source_anisotropic.mp4",
         writer=anim.FFMpegWriter(fps=25))
plt.close(fig)

# gate: wavelength inside the block (x-pol) ~ WL/2
prof_line = fx[-1][nx // 2, RES: j_blk - RES]
zc = np.where(np.diff(np.sign(prof_line)))[0]
lam_meas = 2 * np.mean(np.diff(zc)) * dx if len(zc) > 3 else float("nan")
print(f"simulate_gaussian_source_anisotropic  lambda(x-pol, in block) = "
      f"{lam_meas * 1e3:.0f} nm (expect {WL / 2 * 1e3:.0f});  mp4 written")
