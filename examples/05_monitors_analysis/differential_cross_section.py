# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Differential scattering cross-section of a cylinder
#
# meep-torch port of meep/python/examples/differential_cross_section.py.
# Differential scattering cross-section dsigma/dtheta of a dielectric
# cylinder (2D TM substitute for meep's 3D sphere): scattered-field
# DIFFERENCE contour -> W4 near2far -> |Ez(theta)|^2, vs the analytic 2D Mie
# |T(theta)|^2 (core/mie2d), at f=0.6. The Gaussian illumination (waist 4)
# carries ~0.13 rad angular spread, smearing the side lobes -- the forward
# lobe matches; normalized rms ~0.14 (plane-wave TFSF is the sharp-lobe
# future route). Output: `output/differential_cross_section.png`.
#
# Run: `uv run python examples/05_monitors_analysis/differential_cross_section.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.mie2d import dsigma_dtheta
from meep_torch.core.monitors import DFTFieldsMonitor
from meep_torch.core.near2far import Face, farfield_tm
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
N_CYL, R_CYL, RES, F = 2.0, 0.7, 20, 0.6
W_BEAM = 4.0

# %% [markdown]
# ## DFT field planes per run
# Gaussian-beam illumination (waist 4) of the cell with and without the
# cylinder; whole-plane DFT monitors record Ez, Hx, Hy at f=0.6.

# %%
def planes(with_cyl):
    dx = 1.0 / RES
    nx = ny = int(14.0 * RES)
    grid = Grid(nx=nx, ny=ny, dx=dx)
    ss = 4
    xi = (torch.arange(nx * ss, dtype=torch.float64) + 0.5) / (RES * ss)
    II, JJ = torch.meshgrid(xi, xi, indexing="ij")
    fine = torch.ones(nx * ss, nx * ss, dtype=torch.float64)
    if with_cyl:
        fine[(II - 7.0) ** 2 + (JJ - 7.0) ** 2 < R_CYL ** 2] = N_CYL ** 2
    eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES)
    src = LineSource(grid, ix=int(2.0 * RES), j0=0, j1=ny,
                     profile=GaussianPulse(F, 0.2))
    ys = (torch.arange(ny, dtype=torch.float64) + 0.5) * dx
    src.mask = torch.zeros(nx, ny, dtype=torch.float64)
    src.mask[int(2.0 * RES), :] = torch.exp(-((ys - 7.0) / W_BEAM) ** 2)
    mons = [DFTFieldsMonitor(grid, [F], c) for c in ("Ez", "Hx", "Hy")]
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(70.0 / grid.dt)),
                     source=src, monitors=mons)
    return {c: (a[0, 0] + 1j * a[0, 1]).detach().numpy()
            for c, a in zip(("Ez", "Hx", "Hy"), accs)}, grid

# %% [markdown]
# ## Scattered-field contour and far-field
# The difference fields on a closed box around the cylinder feed the W4
# near2far transform, sampled around a full circle at r=1000.

# %%
p1, grid = planes(True)
p0, _ = planes(False)
sc = {c: p1[c] - p0[c] for c in p1}
dx = grid.dx
cx = int(7.0 * RES)
b = int(1.5 * RES)
faces = []
for idx, normal, axis in ((cx + b, (+1, 0), "x"), (cx - b, (-1, 0), "x"),
                          (cx + b, (0, +1), "y"), (cx - b, (0, -1), "y")):
    rng = np.arange(cx - b, cx + b + 1)
    w = np.ones(len(rng)); w[0] = w[-1] = 0.5
    if axis == "x":
        pts = np.stack([np.full(len(rng), idx * dx), rng * dx], axis=1)
        E = sc["Ez"][idx, rng]
        H = 0.5 * (sc["Hy"][idx - 1, rng] + sc["Hy"][idx, rng])
    else:
        pts = np.stack([rng * dx, np.full(len(rng), idx * dx)], axis=1)
        E = sc["Ez"][rng, idx]
        H = 0.5 * (sc["Hx"][rng, idx - 1] + sc["Hx"][rng, idx])
    faces.append(Face(pts=pts, normal=normal, dl=dx, E=E[None, :],
                      H=H[None, :], w=w))

th = np.linspace(-math.pi, math.pi, 361)
obs = np.stack([7.0 + 1000 * np.cos(th), 7.0 + 1000 * np.sin(th)], axis=1)
E, _ = farfield_tm(faces, [2 * math.pi * F], obs)
got = np.abs(E[0, :, 2]) ** 2
got /= got.max()
an = dsigma_dtheta(N_CYL, 2 * math.pi * F * R_CYL, th)
an /= an.max()

# %% [markdown]
# ## Plot
# Polar dsigma/dtheta vs the analytic Mie pattern; console prints the
# normalized rms.

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.4, 5))
ax.plot(th, got, "b-", lw=1.1, label="meep-torch (scattered n2f)")
ax.plot(th, an, "r--", lw=1.0, label="2D Mie $|T(\\theta)|^2$")
ax.set_title(f"d$\\sigma$/d$\\theta$, cylinder n={N_CYL} r={R_CYL}, f={F}")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "differential_cross_section.png", dpi=150)
plt.close(fig)
rms = float(np.sqrt(np.mean((got - an) ** 2)))
print(f"differential_cross_section.png  normalized rms vs Mie = {rms:.3f}")
