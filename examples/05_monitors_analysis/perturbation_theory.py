# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Ring perturbation theory (finite-difference arm)
#
# meep-torch port of meep/python/examples/perturbation_theory.py. meep's
# perturbation-theory tutorial computes the first-order frequency shift dwdR
# of a ring-resonator mode vs inner-radius perturbation, comparing an
# analytic perturbation-theory integral against a finite difference of
# Harminv frequency extractions. Output: `output/perturbation_theory.png`.
#
# BLOCKER / SUBSTITUTION:
# * perturbation_theory.py uses mp.CYLINDRICAL dimensions with angular order
#   m=5 (a 1D radial cell with exp(i*m*phi) dependence). meep-torch is 2D
#   Cartesian FDTD only -- it cannot represent the cylindrical m=5 mode.
# * Per the porting rules, we substitute a 2D Cartesian ring-cavity
#   perturbation sweep: the same ring annulus (outer eps=n^2, inner air),
#   perturb the inner radius by dr, and extract the resonant frequency for
#   each geometry via probe + rfft peak-find (Harminv substitute). The
#   finite-difference dwdR = (f_perturbed - f_unperturbed)/dr is reported.
#   The analytic perturbation-theory integral (needs field-point +
#   energy-in-box sampling not exposed here) is NOT computed; only the
#   finite-difference arm.
# * The meep "parallel" (Ez) source at (r+0.1, 0) is used (default branch).
#
# Resolution reduced from 100 to 10 to keep runtime <90s.
#
# Run: `uv run python examples/05_monitors_analysis/perturbation_theory.py`

# %%
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

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

n = 3.4
w = 1.0
r = 1.0
pad = 4.0
dpml = 2.0
sxy = 2 * (r + w + pad + dpml)
fcen = 0.17
df = 0.2 * fcen
dr = 0.04

resolution = 10   # reduced from meep's 100 -> runtime <90s
dx = 1.0 / resolution
nx = int(round(sxy / dx))
ny = nx
npml = int(round(dpml / dx))

# %% [markdown]
# ## Resonance per geometry
# Build the annulus with a given inner radius, pulse it, and read the
# ringdown-tail spectral peak as the resonant frequency.

# %%
def circle(cx, cy, radius, n_pts=64):
    th = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    return Polygon(np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1))


def ring_freq(inner_radius):
    """Resonant frequency of the 2D ring (Ez mode) via FFT of a probe ringdown."""
    grid = Grid(nx=nx, ny=ny, dx=dx)
    cx = cy = sxy / 2.0
    outer = circle(cx, cy, inner_radius + w)
    inner = circle(cx, cy, inner_radius)
    rm = rasterize_scene([(outer, n ** 2), (inner, 1.0)], grid, background=1.0)
    coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)
    # source just inside the outer ring wall, at meep (r+0.1, 0)
    src_ix = int(round((cx + inner_radius + 0.1) / dx))
    src_iy = int(round(cy / dx))
    src = LineSource(grid, ix=src_ix, j0=src_iy, j1=src_iy + 1,
                     profile=GaussianPulse(fcen, df))
    # probe at (0, r+0.1) -- in the ring, orthogonal to the source
    probe = PointProbe(int(round(cx / dx)), int(round((cy + inner_radius + 0.1) / dx)))
    t_end = 2 * 5.0 / df
    n_steps = int(round((t_end + 300.0) / grid.dt))
    state = init_tm_state(grid)
    _, (series,) = run_tm(state, coeffs, n_steps, source=src, monitors=[probe])
    series = series.detach().numpy()
    steady = series[int(round(t_end / grid.dt)):]
    spec = np.abs(np.fft.rfft(steady))
    freqs = np.fft.rfftfreq(len(steady), grid.dt)
    return float(freqs[spec.argmax()])

# %% [markdown]
# ## Finite difference and spectra plot
# Two geometries (r and r+dr) give dwdR by finite difference; the two
# ringdown spectra are re-run and overlaid for the figure.

# %%
f_unperturbed = ring_freq(r)
f_perturbed = ring_freq(r + dr)
dwdR_finite_diff = (f_perturbed - f_unperturbed) / dr

# --- plot the two spectra for comparison ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, rad, color in [("unperturbed r=%.2f" % r, r, "C0"),
                          ("perturbed r=%.2f" % (r + dr), r + dr, "C3")]:
    grid = Grid(nx=nx, ny=ny, dx=dx)
    cx = cy = sxy / 2.0
    outer = circle(cx, cy, rad + w); inner = circle(cx, cy, rad)
    rm = rasterize_scene([(outer, n ** 2), (inner, 1.0)], grid, background=1.0)
    coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)
    src_ix = int(round((cx + rad + 0.1) / dx)); src_iy = int(round(cy / dx))
    src = LineSource(grid, ix=src_ix, j0=src_iy, j1=src_iy + 1,
                     profile=GaussianPulse(fcen, df))
    probe = PointProbe(int(round(cx / dx)), int(round((cy + rad + 0.1) / dx)))
    t_end = 2 * 5.0 / df
    n_steps = int(round((t_end + 300.0) / grid.dt))
    state = init_tm_state(grid)
    _, (series,) = run_tm(state, coeffs, n_steps, source=src, monitors=[probe])
    steady = series[int(round(t_end / grid.dt)):].detach().numpy()
    spec = np.abs(np.fft.rfft(steady))
    freqs = np.fft.rfftfreq(len(steady), grid.dt)
    ax.plot(freqs, spec / spec.max(), color=color, label=label, lw=0.8)
ax.set_xlabel("frequency"); ax.set_ylabel("|Ez(fft)| (norm)")
ax.set_title("ring perturbation: Ez-mode freq shift (cylindrical m=5 -> 2D Cartesian substitute)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "perturbation_theory.png", dpi=150)
plt.close(fig)

print(f"perturbation_theory.png  f0(unpert)={f_unperturbed:.5f}  "
      f"f0(pert)={f_perturbed:.5f}  dwdR(finite diff)={dwdR_finite_diff:.5f}")
print("NOTE: cylindrical m=5 mode unsupported; ported as 2D Cartesian ring sweep. "
      "Analytic perturbation-theory integral not computed (field-point/energy-box API "
      "not exposed). Only the finite-difference arm is reported.")
