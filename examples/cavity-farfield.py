# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/cavity-farfield.py.

Far fields of the holey-waveguide cavity (the W3 holey-wvg-cavity
geometry): near fields on a line above the waveguide -> W4 2D near2far
-> |E| along a far-field line above the structure.

SUBSTITUTIONS:
  * meep add_near2far one-sided line -> W4 box-contour machinery used
    one-sided (single top face; the cavity radiates predominantly up/down
    and the far line sits directly above).
  * dpad reduced 32 -> 8 (runtime); resolution 20.

Run: uv run python examples/cavity-farfield.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.near2far import faces_from_accs, farfield_tm
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent

RES = 20
FCEN, DF = 0.25, 0.2
EPS, W, R_HOLE, D, N = 13.0, 1.2, 0.36, 1.4, 3
DPAD, DPML = 8.0, 1.0
D1, D2 = 0.2, 2.0

dx = 1.0 / RES
sx = 2 * (DPAD + DPML + N) + D - 1
sy = W + 2 * (D1 + D2 + DPML)
nx, ny = int(round(sx * RES)), int(round(sy * RES))
grid = Grid(nx=nx, ny=ny, dx=dx)
cx, cy = sx / 2, sy / 2


def circle(x0, y0, rad, n_pts=48):
    th = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    return Polygon(np.stack([x0 + rad * np.cos(th), y0 + rad * np.sin(th)], axis=1))


slab = Polygon(np.array([[0, cy - W / 2], [sx, cy - W / 2],
                         [sx, cy + W / 2], [0, cy + W / 2]], dtype=float))
items = [(slab, EPS)]
for i in range(N):
    items.append((circle(cx + (D / 2 + i), cy, R_HOLE), 1.0))
    items.append((circle(cx - (D / 2 + i), cy, R_HOLE), 1.0))
rm = rasterize_scene(items, grid, background=1.0)
coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=int(DPML * RES))

src = LineSource(grid, ix=nx // 2, j0=ny // 2, j1=ny // 2 + 1,
                 profile=GaussianPulse(FCEN, DF))
# one-sided near2far line at y = cy + W/2 + D1
jy = int(round((cy + W / 2 + D1) * RES))
mon = FluxMonitorTM(grid, [FCEN], x_normal=False, idx=jy,
                    lo=int(DPML * RES), hi=nx - int(DPML * RES))
n_steps = int(round(300.0 / grid.dt))
_, (acc,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                   source=src, monitors=[mon])
faces = faces_from_accs(grid, [(mon, (0, +1))], [acc])

# far-field line at y = cy + W/2 + D1 + D2, spanning x
xs = np.linspace(DPML, sx - DPML, 200)
obs = np.stack([xs, np.full_like(xs, cy + W / 2 + D1 + D2)], axis=1)
E, H = farfield_tm(faces, [2 * math.pi * FCEN], obs)
absE = np.abs(E[0, :, 2])

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(xs - cx, absE / absE.max(), lw=1.0)
ax.set_xlabel("x - x_cavity (um)")
ax.set_ylabel("|Ez| (normalized)")
ax.set_title(f"holey-wvg-cavity far field at y = +{D1 + D2} um")
fig.tight_layout()
fig.savefig(OUT / "cavity-farfield.png", dpi=150)
plt.close(fig)
peak_x = float(xs[absE.argmax()] - cx)
print(f"cavity-farfield.png  far-field peak at x - x_cav = {peak_x:.2f} um "
      f"(cavity-centered: {'yes' if abs(peak_x) < 1.0 else 'no'})")
