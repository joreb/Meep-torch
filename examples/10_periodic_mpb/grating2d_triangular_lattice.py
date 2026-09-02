# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Triangular-lattice photonic crystal with a point defect
#
# A 3x3 supercell of the triangular air-hole lattice in glass, with the
# center hole removed to form a defect cavity. An Ez Gaussian pulse launched
# at the defect images the near-field pattern (the meep original's 3D
# diffraction-order computation is out of scope; only the 2D lattice/field
# part is kept).

# %%
"""meep-torch port of meep/python/examples/grating2d_triangular_lattice.py.

2D photonic crystal: a triangular lattice of air holes in a glass slab, with a
point defect (missing center hole). An Ez Gaussian pulse is launched near the
defect and the resulting near-field Ez pattern is imaged.

The meep original is a 3D binary *diffraction grating* (glass cylinders on a
glass substrate, periodic in x/y via k_point, PML in z) that computes the
diffraction-order transmission coefficients via eigenmode expansion
(`get_eigenmode_coefficients`). 3D simulation, Bloch periodic boundaries, and
eigenmode/diffraction-order extraction are all UNSUPPORTED in meep-torch, so
this port keeps only the 2D triangular-lattice geometry / field part and
images the near field. The diffraction-order computation is omitted.

meep -> meep-torch mapping:
  triangular lattice, sx=1.0, sy=sqrt(3)       ->  same lattice, 3x3 supercell, N=3
  glass n=1.5 (eps=2.25) substrate + cylinders  ->  background eps=2.25 (glass), air holes (eps=1.0)
  mp.Cylinder radius=0.1                        ->  24-gon Polygon, r=0.1 um
  defect = missing center hole                  ->  skip the center lattice site
  mp.GaussianSource(1/0.5, fwidth=0.1*fc) Ex   ->  GaussianPulse(fcen, fwidth) Ez LineSource

Run: uv run python examples/10_periodic_mpb/grating2d_triangular_lattice.py
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

OUT = pathlib.Path(__file__).parent / "output"

# --- parameters (meep natural units, um) ---
RES = 30                        # pixels/um (reduced from 100 for runtime)
DX = 1.0 / RES
NG = 1.5                         # glass index
WVL = 0.5
FCEN = 1.0 / WVL
FWIDTH = 0.1 * FCEN
SX = 1.0                        # lattice period (x)
SY = np.sqrt(3)                 # lattice period (y)
R_HOLE = 0.1                    # hole radius
N_SUPER = 3                     # 3x3 supercell
PML_UM = 0.5
DEFECT = True                   # remove center hole to form a defect cavity

CELL_X = N_SUPER * SX
CELL_Y = N_SUPER * SY
NX = int(round(CELL_X / DX))
NY = int(round(CELL_Y / DX))
NPML = int(round(PML_UM / DX))

grid = Grid(nx=NX, ny=NY, dx=DX)


def circle_polygon(cx: float, cy: float, r: float, n: int = 24) -> Polygon:
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    verts = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)
    return Polygon(vertices=verts)


# %% [markdown]
# ## Lattice construction and run
#
# Each sx x sqrt(3) rectangle carries two lattice sites (corner + body
# center); skipping the central body-center site creates the defect. The
# pulse rings for ~80 time units before the snapshot.

# %%
# Rectangular-supercell convention (meep original): each sx x sy rectangle holds
# two triangular-lattice sites -- site A at the corner (i*sx, j*sy) and site B at
# the body center (i*sx + sx/2, j*sy + sy/2). Tiling the N_SUPER x N_SUPER grid of
# rectangles fills the rectangular cell with a triangular hole pattern (rows
# alternate between x=0,1,2,... and x=0.5,1.5,... at y spacing sqrt(3)/2). The
# central body-center site is removed to form a defect cavity.
items = []
defect_i = N_SUPER // 2
defect_j = N_SUPER // 2
for i in range(N_SUPER):
    for j in range(N_SUPER):
        for (px, py), is_body in (
            ((i * SX, j * SY), False),                       # corner site A
            ((i * SX + 0.5 * SX, j * SY + 0.5 * SY), True),  # body-center site B
        ):
            if px >= CELL_X or py >= CELL_Y:
                continue
            if DEFECT and i == defect_i and j == defect_j and is_body:
                continue
            items.append((circle_polygon(px, py, R_HOLE), 1.0))   # air hole (eps=1.0)

rm = rasterize_scene(items, grid, background=NG * NG)          # glass background
eps = rm.eps_tm
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

# Ez point source near the defect center.
ix_src = int(round(CELL_X / (2 * DX)))
iy_src = int(round(CELL_Y / (2 * DX)))
src = LineSource(grid, ix=ix_src, j0=iy_src, j1=iy_src + 1, profile=GaussianPulse(FCEN, FWIDTH))

state = init_tm_state(grid)
N_STEPS = int(round(80.0 / grid.dt))               # ~80 time units / dt(=0.0167) = 4800
for t in range(N_STEPS):
    state = step_tm(state, coeffs, src.amplitude(t), src.mask)

v = float(state.Ez.abs().max()) or 1.0
fig, ax = plt.subplots(figsize=(7, 6))
ax.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9)
ax.imshow(state.Ez.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax.plot(ix_src, iy_src, "y*", ms=10)
ax.set_title("2D triangular photonic-crystal lattice (air holes in glass)\nwith point defect -- near-field Ez")
ax.set_xlabel("x (cells, dx=%.4g um)" % DX)
ax.set_ylabel("y (cells)")
fig.tight_layout()
fig.savefig(OUT / "grating2d_triangular_lattice.png", dpi=150)
plt.close(fig)

n_holes = len(items)
print("grating2d_triangular_lattice.png")
print(f"max|Ez|={v:.4e}  ({n_holes} holes, {N_SUPER}x{N_SUPER} supercell, defect={DEFECT})")