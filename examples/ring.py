# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/ring.py.

2D ring resonator: two overlapping cylinders (outer eps=n^2, inner air) form an
annulus. A Gaussian-pulse Ez source excites the ring modes; a probe inside the
ring records Ez vs time, and an FFT extracts the resonant frequency.

SUBSTITUTION: meep uses Harminv to extract resonant frequency/Q. meep-torch has
NO harminv; we record Ez at a PointProbe through run_tm, then numpy/torch rfft +
peak-find on the ringdown tail. Q is estimated as f0/dfwhm when resolvable.

meep -> meep-torch mapping:
  cell sxy=16 (2*(r+w+pad+dpml)=2*8), resolution 10  ->  Grid(160,160,dx=0.1)
  mp.Cylinder(r+w, n=3.4) + mp.Cylinder(r, vacuum)   ->  two Polygon circles, painter's order
  mp.PML(2.0)                                        ->  npml=20
  mp.GaussianSource(0.15, fwidth=0.1) Ez @ (r+0.1,0)  ->  GaussianPulse(0.15,0.1), LineSource single cell
  Harminv @ (r+0.1,0)                                ->  PointProbe + rfft peak-find
  until_after_sources=300                            ->  n_steps=round((t_end+300)/dt)

Run: uv run python examples/ring.py
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

# --- geometry (meep units: um) ---
n = 3.4
w = 1.0
r = 1.0
pad = 4.0
dpml = 2.0
sxy = 2 * (r + w + pad + dpml)          # 16.0
fcen = 0.15
df = 0.1

# resolution 10 -> dx=0.1; grid 160x160. Reduced from meep's mirror-halved cell,
# no symmetry exploited here (meep-torch has no Mirror). Runtime <90s.
resolution = 10
dx = 1.0 / resolution
nx = int(round(sxy / dx))               # 160
ny = nx
npml = int(round(dpml / dx))            # 20
grid = Grid(nx=nx, ny=ny, dx=dx)

# circle approximated by polygon, in grid coords (grid (0,0) = meep (-sxy/2,-sxy/2))
def circle(cx, cy, radius, n_pts=64):
    th = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    return Polygon(np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1))

cx = cy = sxy / 2.0                      # meep origin -> grid center
outer = circle(cx, cy, r + w)            # eps = n^2
inner = circle(cx, cy, r)                # air (eps=1) overwrites center -> annulus
rm = rasterize_scene([(outer, n ** 2), (inner, 1.0)], grid, background=1.0)
coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)

# source: GaussianPulse Ez at meep (r+0.1, 0) -> grid (cx + r+0.1, cy)
src_ix = int(round((cx + r + 0.1) / dx))
src_iy = int(round(cy / dx))
src = LineSource(grid, ix=src_ix, j0=src_iy, j1=src_iy + 1, profile=GaussianPulse(fcen, df))

# probe inside the ring at meep (0, r+0.1) -> grid (cx, cy + r+0.1)
probe_ix = int(round(cx / dx))
probe_iy = int(round((cy + r + 0.1) / dx))
probe = PointProbe(probe_ix, probe_iy)

# --- run ---
# GaussianPulse t_end = 2*cutoff/fwidth = 10/df = 100; plus 300 ringdown.
t_end = 2 * 5.0 / df
n_steps = int(round((t_end + 300.0) / grid.dt))
state = init_tm_state(grid)
state, (series,) = run_tm(state, coeffs, n_steps, source=src, monitors=[probe])
series = series.detach().numpy()

# --- FFT peak-find on ringdown tail (substitute for Harminv) ---
steady_start = int(round(t_end / grid.dt))
steady = series[steady_start:]
spec = np.abs(np.fft.rfft(steady))
freqs = np.fft.rfftfreq(len(steady), grid.dt)
peak_idx = int(spec.argmax())
f0 = float(freqs[peak_idx])
# Q ~ f0 / FWHM; estimate half-width via half-power points around the peak
half = spec.max() * 0.5
left = np.where(spec[:peak_idx] <= half)[0]
right = np.where(spec[peak_idx:] <= half)[0]
if len(left) and len(right):
    f_lo = freqs[left[-1]]
    f_hi = freqs[peak_idx + right[0]]
    dfwhm = max(f_hi - f_lo, freqs[1] - freqs[0])
    Q = f0 / dfwhm if dfwhm > 0 else float("nan")
else:
    Q = float("nan")

# --- plot: field + spectrum ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
v = float(np.abs(state.Ez.numpy()).max()) or 1.0
ax1.imshow(rm.eps_tm.numpy().T, origin="lower", cmap="binary", alpha=0.9)
ax1.imshow(state.Ez.numpy().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax1.set_title(f"ring Ez (t={n_steps} steps, res={resolution})")
ax1.set_xlabel("x (cells, dx=%.2f um)" % dx)
ax1.set_ylabel("y (cells)")
ax2.plot(freqs, spec, lw=0.8)
ax2.axvline(f0, color="r", ls="--", label=f"f0={f0:.4f}")
ax2.set_xlabel("frequency")
ax2.set_ylabel("|Ez(fft)|")
ax2.set_title("ringdown spectrum (Harminv substitute)")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "ring.png", dpi=150)
plt.close(fig)
print(f"ring.png  f0={f0:.5f}  Q~{Q:.1f}  (meep tutorial ~0.15-0.2)")