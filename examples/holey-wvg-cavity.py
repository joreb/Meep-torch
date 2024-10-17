# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/holey-wvg-cavity.py (resonant-mode branch).

2D PhC waveguide cavity: a dielectric slab (eps=13) perforated by a periodic
sequence of air holes, with a defect (larger spacing d) in the middle forming a
cavity. A Gaussian-pulse Ez source at the defect excites a resonant cavity mode;
a probe records Ez vs time and an FFT extracts the resonant frequency.

SUBSTITUTIONS:
  * meep uses Hz (TE-like, out-of-plane H) polarization with odd Y/X mirror
    symmetry. meep-torch exposes the TM (out-of-plane Ez) stepper as the primary
    API; we port to Ez polarization. The cavity still supports an Ez-polarized
    monopole mode; the resonant frequency differs from the meep Hz value.
  * meep uses Harminv for f/Q extraction -> replaced by probe + rfft peak-find.
  * No mirror symmetry (meep-torch has no Mirror); full cell is simulated.

Resolution reduced from 20 to 10 (dx=0.05 -> 0.1) to keep runtime <90s; noted.

Run: uv run python examples/holey-wvg-cavity.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent

# --- geometry (meep units) ---
eps = 13.0
w = 1.2
r_hole = 0.36
d = 1.4                 # defect spacing (ordinary spacing = 1)
N = 3                   # number of holes on either side of defect
sy = 6.0
pad = 2.0
dpml = 1.0
sx = 2 * (pad + dpml + N) + d - 1        # 12.4
fcen = 0.25
df = 0.2

# reduced from meep resolution=20 to 10 to keep runtime <90s
resolution = 10
dx = 1.0 / resolution
nx = int(round(sx / dx))                 # 124
ny = int(round(sy / dx))                 # 60
npml = int(round(dpml / dx))             # 10
grid = Grid(nx=nx, ny=ny, dx=dx)

# circle helper (grid coords; grid (0,0) = meep (-sx/2,-sy/2))
def circle(cx, cy, radius, n_pts=48):
    th = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    return Polygon(np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1))

cx = sx / 2.0
cy = sy / 2.0

# slab: rectangle spanning full x, y in [cy-w/2, cy+w/2], eps=13
slab = Polygon(np.array(
    [[0.0, cy - w / 2], [sx, cy - w / 2], [sx, cy + w / 2], [0.0, cy + w / 2]], dtype=float))
items = [(slab, eps)]
# holes (air, eps=1) overwrite the slab; painter's order: slab first, holes second
for i in range(N):
    items.append((circle(cx=cx + (d / 2 + i), cy=cy, radius=r_hole), 1.0))
    items.append((circle(cx=cx - (d / 2 + i), cy=cy, radius=r_hole), 1.0))
rm = rasterize_scene(items, grid, background=1.0)
coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)

# source: GaussianPulse Ez at defect center (meep (0,0) -> grid (cx, cy))
src_ix = int(round(cx / dx))
src_iy = int(round(cy / dx))
src = LineSource(grid, ix=src_ix, j0=src_iy, j1=src_iy + 1, profile=GaussianPulse(fcen, df))

# probe at center (cavity monopole mode peaks at defect); use tail after source ends
probe = PointProbe(src_ix, src_iy)

# --- run ---
# t_end = 10/df = 50; allow long ringdown for the (high-Q) cavity mode: ~350 total
t_end = 2 * 5.0 / df
n_steps = int(round((t_end + 350.0) / grid.dt))   # ~8000
state = init_tm_state(grid)
state, (series,) = run_tm(state, coeffs, n_steps, source=src, monitors=[probe])
series = series.detach().numpy()

# --- FFT peak-find (Harminv substitute) ---
steady_start = int(round(t_end / grid.dt))
steady = series[steady_start:]
spec = np.abs(np.fft.rfft(steady))
freqs = np.fft.rfftfreq(len(steady), grid.dt)
peak_idx = int(spec.argmax())
f0 = float(freqs[peak_idx])
half = spec.max() * 0.5
left = np.where(spec[:peak_idx] <= half)[0]
right = np.where(spec[peak_idx:] <= half)[0]
if len(left) and len(right):
    dfwhm = max(freqs[peak_idx + right[0]] - freqs[left[-1]], freqs[1] - freqs[0])
    Q = f0 / dfwhm if dfwhm > 0 else float("nan")
else:
    Q = float("nan")

# --- plot: eps+Ez field, and spectrum ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
v = float(np.abs(state.Ez.numpy()).max()) or 1.0
ax1.imshow(rm.eps_tm.numpy().T, origin="lower", cmap="binary", alpha=0.9)
ax1.imshow(state.Ez.numpy().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax1.set_title(f"holey-wvg-cavity Ez (res={resolution}, N={N})")
ax1.set_xlabel("x (cells, dx=%.2f um)" % dx)
ax1.set_ylabel("y (cells)")
ax2.plot(freqs, spec, lw=0.8)
ax2.axvline(f0, color="r", ls="--", label=f"f0={f0:.4f}")
ax2.set_xlabel("frequency")
ax2.set_ylabel("|Ez(fft)|")
ax2.set_title("cavity ringdown spectrum (Harminv substitute)")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "holey-wvg-cavity.png", dpi=150)
plt.close(fig)
print(f"holey-wvg-cavity.png  f0={f0:.5f}  Q~{Q:.1f}  (Ez polarization; meep Hz mode differs)")