# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/solve-cw.py.

meep's solve_cw tutorial drives a CW frequency-domain solver at the ring
resonator's resonant mode (fcen=0.118) and visualizes the steady-state fields
in time and frequency (DFT) domains.

SUBSTITUTION:
  * meep-torch is time-domain only; there is no solve_cw / CW solver. We
    substitute a long CWRamp run at fcen=0.118 (two odd-sign Ez sources, the
    meep odd mode), let the resonant field build up, then:
      - snapshot the time-domain Ez field (the "CW" field at the last step),
      - rfft a probe time series to confirm the spectral peak sits at fcen
        (the DFT-field substitute: a single-frequency DFT bin = the CW field).
  * meep's tolerance-convergence sweep (solve_cw with decreasing tol) has no
    time-domain analog and is dropped.

Resolution reduced from 20 to 10 to keep runtime <90s; noted.

Run: uv run python examples/solve-cw.py
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
from meep_torch.core.sources import CWRamp, LineSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent

n = 3.4
w = 1.0
r = 1.0
pad = 4.0
dpml = 2.0
sxy = 2 * (r + w + pad + dpml)
fcen = 0.118

resolution = 10   # reduced from meep's 20 -> runtime <90s
dx = 1.0 / resolution
nx = int(round(sxy / dx))
ny = nx
npml = int(round(dpml / dx))
grid = Grid(nx=nx, ny=ny, dx=dx)

def circle(cx, cy, radius, n_pts=64):
    th = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    return Polygon(np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1))

cx = cy = sxy / 2.0
outer = circle(cx, cy, r + w)
inner = circle(cx, cy, r)
rm = rasterize_scene([(outer, n ** 2), (inner, 1.0)], grid, background=1.0)
coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)

# two odd-sign CWRamp Ez sources at (r+0.1, 0) and (-(r+0.1), 0) -> meep odd mode.
# meep-torch has no Mirror symmetry, so both sources are placed explicitly.
ix_pos = int(round((cx + r + 0.1) / dx))
ix_neg = int(round((cx - (r + 0.1)) / dx))
iy = int(round(cy / dx))
src_pos = LineSource(grid, ix=ix_pos, j0=iy, j1=iy + 1, profile=CWRamp(fcen, t_ramp=50.0))
src_neg = LineSource(grid, ix=ix_neg, j0=iy, j1=iy + 1, profile=CWRamp(fcen, t_ramp=50.0))

probe = PointProbe(int(round(cx / dx)), int(round((cy + r + 0.1) / dx)))

# Buildup time ~ Q/f0; budget ~700 time units (dt=0.05 -> 14000 steps).
n_steps = int(round(700.0 / grid.dt))
state = init_tm_state(grid)
samples = np.empty(n_steps, dtype=np.float64)
for t in range(n_steps):
    a_pos = src_pos.amplitude(t)
    a_neg = -src_neg.amplitude(t)   # odd-sign source (amplitude negated)
    state = step_tm(state, coeffs, a_pos, src_pos.mask)
    state = state._replace(Dz=state.Dz + grid.dt * a_neg * src_neg.mask)
    samples[t] = state.Ez[probe.ix, probe.iy].item()

# --- DFT substitute: rfft of the probe time series; peak should sit at fcen ---
spec = np.abs(np.fft.rfft(samples))
freqs = np.fft.rfftfreq(len(samples), grid.dt)
peak_idx = int(spec.argmax())
f_peak = float(freqs[peak_idx])

# --- plot: time-domain Ez field + DFT spectrum ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
v = float(np.abs(state.Ez.numpy()).max()) or 1.0
ax1.imshow(rm.eps_tm.numpy().T, origin="lower", cmap="binary", alpha=0.9)
ax1.imshow(state.Ez.numpy().T, origin="lower", cmap="RdBu", vmin=-v, vmax=v, alpha=0.8)
ax1.set_title(f"solve-cw substitute: Ez after CWRamp@{fcen} (res={resolution})")
ax1.set_xlabel("x (cells, dx=%.2f um)" % dx); ax1.set_ylabel("y (cells)")
ax2.plot(freqs, spec, lw=0.8)
ax2.axvline(fcen, color="g", ls="--", label=f"drive fcen={fcen}")
ax2.axvline(f_peak, color="r", ls=":", label=f"FFT peak={f_peak:.4f}")
ax2.set_xlabel("frequency"); ax2.set_ylabel("|Ez(DFT)|")
ax2.set_title("DFT spectrum (solve_cw -> CWRamp + rfft substitute)")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "solve-cw.png", dpi=150)
plt.close(fig)
print(f"solve-cw.png  FFT peak={f_peak:.5f} (drive fcen={fcen}); "
      "CW solver substituted with long CWRamp time-domain run + rfft.")