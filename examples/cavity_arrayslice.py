# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/cavity_arrayslice.py.

2D PhC waveguide cavity (same geometry as holey-wvg-cavity). A Gaussian-pulse Ez
source at the defect excites the cavity mode. After the source decays, the Ez
field array is snapshotted and 1D (along x at the waveguide centerline) and 2D
sub-window slices are plotted -- the meep-torch equivalent of meep's
get_array + output_array_slices.

SUBSTITUTIONS:
  * meep uses Hz polarization; ported to Ez (TM) -- meep-torch primary API.
  * meep uses Harminv/source-termination hooks; here we run a fixed pulse length
    and then read state.Ez directly (the field array).
  * No mirror symmetry; full cell.

Resolution reduced from 20 to 10 to keep runtime <90s; noted.

Run: uv run python examples/cavity_arrayslice.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent

# --- geometry (same as holey-wvg-cavity) ---
eps = 13.0
w = 1.2
r_hole = 0.36
d = 1.4
N = 3
sy = 6.0
pad = 2.0
dpml = 1.0
sx = 2 * (pad + dpml + N) + d - 1
fcen = 0.25
df = 0.2

resolution = 10
dx = 1.0 / resolution
nx = int(round(sx / dx))
ny = int(round(sy / dx))
npml = int(round(dpml / dx))
grid = Grid(nx=nx, ny=ny, dx=dx)

def circle(cx, cy, radius, n_pts=48):
    th = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    return Polygon(np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1))

cx, cy = sx / 2.0, sy / 2.0
slab = Polygon(np.array(
    [[0.0, cy - w / 2], [sx, cy - w / 2], [sx, cy + w / 2], [0.0, cy + w / 2]], dtype=float))
items = [(slab, eps)]
for i in range(N):
    items.append((circle(cx + (d / 2 + i), cy, r_hole), 1.0))
    items.append((circle(cx - (d / 2 + i), cy, r_hole), 1.0))
rm = rasterize_scene(items, grid, background=1.0)
coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)

# source: GaussianPulse Ez at defect center
src_ix = int(round(cx / dx))
src_iy = int(round(cy / dx))
src = LineSource(grid, ix=src_ix, j0=src_iy, j1=src_iy + 1, profile=GaussianPulse(fcen, df))

# run until the source has decayed (t_end = 10/df = 50) plus a short ringdown
t_end = 2 * 5.0 / df
n_steps = int(round((t_end + 20.0) / grid.dt))
state = init_tm_state(grid)
for t in range(n_steps):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

# --- array slices of Ez (meep get_array substitute: direct tensor slice) ---
Ez = state.Ez.numpy()
# 1D slice along x at the waveguide centerline (y = cy), restricted to central x window
xMin_i = int(round((cx - 0.25 * sx) / dx))
xMax_i = int(round((cx + 0.25 * sx) / dx))
slice1d = Ez[xMin_i:xMax_i, src_iy]
x1d = np.linspace(-0.25 * sx, 0.25 * sx, len(slice1d))

# 2D slice: central sub-window in both x and y
yMin_j = int(round((cy - 0.15 * sy) / dx))
yMax_j = int(round((cy + 0.15 * sy) / dx))
slice2d = Ez[xMin_i:xMax_i, yMin_j:yMax_j]

# --- plot 1D + 2D slices (mirror of the meep figure layout) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(x1d, slice1d, lw=1.0)
ax1.set_xlabel("x (um)")
ax1.set_ylabel("Ez")
ax1.set_title("1D Ez slice along waveguide centerline")
v = float(np.abs(slice2d).max()) or 1.0
ax2.contourf(np.linspace(-0.25 * sx, 0.25 * sx, slice2d.shape[0]),
             np.linspace(-0.15 * sy, 0.15 * sy, slice2d.shape[1]),
             slice2d.T, levels=21, cmap="RdBu", vmin=-v, vmax=v)
ax2.set_xlabel("x (um)")
ax2.set_ylabel("y (um)")
ax2.set_title("2D Ez field slice")
fig.tight_layout()
fig.savefig(OUT / "cavity_arrayslice.png", dpi=150)
plt.close(fig)
print(f"cavity_arrayslice.png  (Ez polarization, res={resolution}, n_steps={n_steps})")