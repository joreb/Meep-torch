# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # 2D ring perturbation theory (finite-difference arm)
#
# meep-torch port of meep/python/examples/perturbation_theory_2d.py. 2D
# ring-resonator perturbation-theory tutorial. meep computes the resonant
# frequency of the unperturbed and perturbed (inner radius + dr) ring via
# Harminv, takes a finite difference dwdR = (f1-f0)/dr, and compares against
# an analytic perturbation-theory integral (field-point +
# electric-energy-in-box). Output: `output/perturbation_theory_2d.png`.
#
# SUBSTITUTIONS:
# * meep Harminv -> probe + rfft peak-find on the ringdown tail.
# * meep uses two odd-sign sources + X/Y mirror symmetries to select the
#   mode; meep-torch has no Mirror, so a single source is used (the odd-mode
#   selection is lost; the dominant Ez ring mode is still excited and
#   detected).
# * The analytic perturbation-theory integral is NOT computed (needs
#   get_field_point / electric_energy_in_box, not exposed by meep-torch).
#   Only the finite-difference dwdR arm is reported, per the porting rules.
#
# Resolution reduced from 30 to 10 to keep runtime <90s.
#
# Run: `uv run python examples/05_monitors_analysis/perturbation_theory_2d.py`

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
from meep_torch.core.stepper_tm import init_tm_state
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

resolution = 10   # reduced from meep's 30 -> runtime <90s
dx = 1.0 / resolution
nx = int(round(sxy / dx))
ny = nx
npml = int(round(dpml / dx))

# %% [markdown]
# ## Resonance per geometry
# Same probe + rfft peak-find as perturbation_theory.py, returning the
# spectrum for plotting.

# %%
def circle(cx, cy, radius, n_pts=64):
    th = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False)
    return Polygon(np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1))


def ring_freq(inner_radius):
    grid = Grid(nx=nx, ny=ny, dx=dx)
    cx = cy = sxy / 2.0
    outer = circle(cx, cy, inner_radius + w)
    inner = circle(cx, cy, inner_radius)
    rm = rasterize_scene([(outer, n ** 2), (inner, 1.0)], grid, background=1.0)
    coeffs = build_tm_coeffs(grid, eps_inf=rm.eps_tm, npml=npml)
    src_ix = int(round((cx + inner_radius + 0.1) / dx))
    src_iy = int(round(cy / dx))
    src = LineSource(grid, ix=src_ix, j0=src_iy, j1=src_iy + 1,
                     profile=GaussianPulse(fcen, df))
    probe = PointProbe(int(round(cx / dx)), int(round((cy + inner_radius + 0.1) / dx)))
    t_end = 2 * 5.0 / df
    n_steps = int(round((t_end + 300.0) / grid.dt))
    state = init_tm_state(grid)
    _, (series,) = run_tm(state, coeffs, n_steps, source=src, monitors=[probe])
    series = series.detach().cpu().numpy()
    steady = series[int(round(t_end / grid.dt)):]
    spec = np.abs(np.fft.rfft(steady))
    freqs = np.fft.rfftfreq(len(steady), grid.dt)
    return float(freqs[spec.argmax()]), freqs, spec

# %% [markdown]
# ## Finite difference and plot

# %%
f0, freqs0, spec0 = ring_freq(r)
f1, freqs1, spec1 = ring_freq(r + dr)
dwdR_finite_diff = (f1 - f0) / dr

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(freqs0, spec0 / spec0.max(), "C0", lw=0.8, label=f"unperturbed r={r:.2f} (f0={f0:.4f})")
ax.plot(freqs1, spec1 / spec1.max(), "C3", lw=0.8, label=f"perturbed r={r+dr:.2f} (f0={f1:.4f})")
ax.set_xlabel("frequency"); ax.set_ylabel("|Ez(fft)| (norm)")
ax.set_title("2D ring perturbation: Ez-mode freq shift (Harminv -> FFT substitute)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "perturbation_theory_2d.png", dpi=150)
plt.close(fig)

print(f"perturbation_theory_2d.png  f0(unpert)={f0:.5f}  f0(pert)={f1:.5f}  "
      f"dwdR(finite diff)={dwdR_finite_diff:.5f}")
print("NOTE: analytic perturbation-theory integral not computed (field-point / "
      "electric_energy_in_box API not exposed). Only finite-difference arm reported.")
