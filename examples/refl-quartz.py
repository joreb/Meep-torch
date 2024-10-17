# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/refl-quartz.py.

Reflectance of a broadband pulse off a fused-quartz slab, compared with the
analytic Fresnel reflectance. The meep original uses a 1D simulation with
load_minus_flux_data to isolate the reflected flux; meep-torch has no
load_minus_flux_data, so per the port instructions we use two runs (empty +
slab) of FluxMonitorTM and take R = -(slab_flux - empty_flux)/empty_flux,
which equals (empty - slab)/empty = reflected/incident (the backward-reflected
component is the drop in net forward flux at the monitor left of the slab).

meep -> meep-torch mapping:
  dimensions=1, sz=12, resolution 200       ->  thin 2D, dx=0.01, nx=1200, ny=3, npml=100
  mp.PML(1)                                ->  npml=100, y_pmc=True (PML on x only)
  fused_quartz slab, z in [0, 0.5*sz]      ->  rasterize_scene with meep_torch SiO2 Material
  mp.GaussianSource(fcen, fwidth=df) Ex    ->  GaussianPulse + LineSource (full y)
  refl flux @ z=-0.25*sz                    ->  FluxMonitorTM x_normal at ix=300
Run: uv run python examples/refl-quartz.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import SiO2, analytic_eps
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent

# wavelength 0.4-0.8 um -> frequency 1.25-2.5 (1/um)
wvl_min, wvl_max = 0.4, 0.8
fmin, fmax = 1.0 / wvl_max, 1.0 / wvl_min
fcen, df = 0.5 * (fmax + fmin), fmax - fmin
nfreq = 50
freqs = np.linspace(fmin, fmax, nfreq)

sz = 12.0  # cell length (10 + 2 pml)
res = 100  # resolution (um^-1); original 200, reduced for runtime
dx = 1.0 / res
grid = Grid(nx=int(sz * res), ny=3, dx=dx)
npml = res  # 1 um PML

# source just inside the left PML (x=1 um -> ix=100), full-y line -> 1D plane wave
src = LineSource(grid, ix=npml, j0=0, j1=grid.ny, profile=GaussianPulse(fcen, df))

# flux monitor at z=-0.25*sz relative to center (center x=6 um) -> x=3 um -> ix=300
mon = FluxMonitorTM(grid, freqs.tolist(), x_normal=True, idx=3 * res, lo=0, hi=grid.ny)


def run(eps_inf, poles=(), pole_masks=None):
    coeffs = build_tm_coeffs(
        grid, eps_inf=eps_inf, poles=poles, pole_masks=pole_masks,
        npml=npml, y_pmc=True,
    )
    state = init_tm_state(grid, n_poles=len(poles))
    n_steps = int(round(80.0 / grid.dt))  # 80 time units (long enough for dispersive slab tail)
    state, (acc,) = run_tm(state, coeffs, n_steps, source=src, monitors=(mon,))
    return mon.flux(acc, grid.dx).detach().cpu().numpy()


# empty run (vacuum) -> incident flux
empty_flux = run(1.0)

# slab run: fused quartz (SiO2) occupying x in [6, 11] um (right half, z in [0, 5] rel. center)
slab = Polygon(np.array([[6.0, 0.0], [11.0, 0.0], [11.0, grid.ny * dx], [6.0, grid.ny * dx]]))
rm = rasterize_scene([(slab, SiO2)], grid, background=1.0)
slab_flux = run(rm.eps_tm, rm.poles, rm.pole_masks)

# R = -(slab - empty)/empty = (empty - slab)/empty  (reflected = drop in net forward flux)
R = -(slab_flux - empty_flux) / empty_flux
wvls = 1.0 / freqs

# analytic Fresnel using the meep_torch SiO2 Lorentzian eps
n_q = np.array([math.sqrt(analytic_eps(SiO2.eps_inf, SiO2.poles, f).real) for f in freqs])
R_analytic = np.abs((1.0 - n_q) / (1.0 + n_q)) ** 2

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wvls, R, "bo-", label="meep-torch")
ax.plot(wvls, R_analytic, "rs-", label="analytic (SiO2 Lorentzian)")
ax.set_xlabel("wavelength (um)")
ax.set_ylabel("reflectance")
ax.set_title("Fused-quartz slab reflectance (meep refl-quartz)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "refl-quartz.png", dpi=150)
plt.close(fig)

i_mid = nfreq // 2
print(f"refl-quartz.png  R@wvl={wvls[i_mid]:.3f}  meep-torch={R[i_mid]:.5f}  analytic={R_analytic[i_mid]:.5f}")