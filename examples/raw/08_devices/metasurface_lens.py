# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Metasurface lens focusing
#
# A phase-imparting metasurface focuses a CW planewave: glass blocks whose
# thickness varies across the aperture impart the hyperboloidal phase
# phi(y) = 2 pi / lambda * (f - sqrt(y^2 + f^2)), the smooth-profile analogue
# of meep's binary-grating duty-cycle metasurface. The focal spot is imaged
# directly in the FDTD domain.

# %%
"""meep-torch port of meep/python/examples/metasurface_lens.py.

Binary-grating metasurface lens focusing. The meep original has two parts:
(1) a single-cell calibration sweep of phase vs grating duty cycle (using
Bloch-periodic eigenmode coefficients), and (2) a multi-cell lens built from
unit cells whose duty cycle is chosen from the calibration to impart the
hyperboloidal phase profile, with the far field imaged via near-to-far transform.

meep-torch has neither Bloch-periodic y (needed for the calibration) nor
near-to-far transform (needed for the far-field image), so this port does the
focusing part directly in the FDTD domain (like zone_plate.py): a phase-imparting
metasurface is built from glass blocks whose *thickness* in x varies across the
aperture to give the required hyperboloidal phase
  phi(y) = 2*pi/lambda * (f - sqrt(y^2 + f^2))            (mod 2*pi)
  h(y)   = phi(y) * lambda / (2*pi*(n_glass - 1))
which is the smooth-profile analogue of the meep binary-grating metasurface
(same hyperboloidal phase, imparted by varying thickness instead of varying
duty cycle). Part 1 (duty-cycle calibration) is omitted; the thickness-to-phase
relation is analytic for a smooth slab. The focal field is imaged directly.

The focal-field envelope |A(x,y)| is recovered from two steady-state Ez
snapshots T/4 apart: |A|^2 = Ez(t)^2 + Ez(t+T/4)^2.

Reduced for runtime: focal_length=6 um (meep: 200), aperture radius 3 um,
N=60 cells across the aperture, resolution 20 (dx=0.05).
Run: uv run python examples/08_devices/metasurface_lens.py
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
from meep_torch.core.sources import CWRamp, LineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.core.run import run_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Lens design: wrapped hyperboloidal thickness profile
#
# Each of the 60 aperture cells gets a glass block of thickness
# h(y) = phi(y) * lambda / (2 pi (n-1)), with the phase wrapped modulo 2 pi so
# h stays below h_wrap = lambda/(n-1) = 1 um (a Fresnel-lens-style wrap).

# %%
# --- lens parameters (meep units: um) ---
pml_um = 1.0
substrate_um = 1.0
focal_length = 6.0              # reduced from meep's 200
aperture_radius = 3.0
N_cells = 60                    # cells across the full aperture
wavelength = 0.5
fcen = 1.0 / wavelength
n_glass = 1.5
eps_glass = n_glass ** 2
h_wrap = wavelength / (n_glass - 1)   # thickness for a 2*pi phase shift = 1.0 um
pad_um = 1.0

# metasurface thickness range (max wrapped thickness < h_wrap)
# h_max_unwrapped = (sqrt(R^2 + f^2) - f) / (n-1); wraps modulo h_wrap.
dy = 2.0 * aperture_radius / N_cells
y_cells = np.linspace(-aperture_radius + 0.5 * dy,
                      aperture_radius - 0.5 * dy, N_cells)
# hyperboloidal phase (negative), wrapped to [0, 2*pi)
phase_raw = (2 * np.pi / wavelength) * (focal_length - np.sqrt(y_cells ** 2 + focal_length ** 2))
phase_mod = phase_raw % (2 * np.pi)
h_cells = phase_mod * wavelength / (2 * np.pi * (n_glass - 1))   # in [0, h_wrap)

# --- cell ---
h_max = float(h_cells.max())
sx = pml_um + substrate_um + h_wrap + focal_length + pad_um + pml_um
sy = 2.0 * (aperture_radius + pad_um + pml_um)
dx = 0.05
nx = int(round(sx / dx))
ny = int(round(sy / dx))
npml = int(round(pml_um / dx))
grid = Grid(nx=nx, ny=ny, dx=dx)

# --- build geometry ---
# glass substrate x in [pml, pml+substrate], full y interior.
x0 = pml_um
x1 = pml_um + substrate_um
items = [(Polygon(np.array([[x0, pml_um], [x1, pml_um],
                            [x1, sy - pml_um], [x0, sy - pml_um]])), eps_glass)]
# metasurface blocks: each at x in [x1, x1+h_j], y in [y_j-dy/2, y_j+dy/2]
yc = 0.5 * sy   # meep y=0 -> grid y=sy/2
for yj, hj in zip(y_cells, h_cells):
    ya = yc + yj - 0.5 * dy
    yb = yc + yj + 0.5 * dy
    items.append((Polygon(np.array([
        [x1, ya], [x1 + hj, ya], [x1 + hj, yb], [x1, yb]])), eps_glass))

eps = rasterize_scene(items, grid, background=1.0).eps_tm
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)

# %% [markdown]
# ## CW planewave drive and steady state
#
# A ramped CW line source inside the substrate illuminates the metasurface;
# after 2200 steps the field has reached steady state.

# %%
# --- source: CW planewave just inside the left PML ---
src_ix = int(round((pml_um + 0.3 * substrate_um) / dx))
src = LineSource(grid, ix=src_ix, j0=npml, j1=ny - npml,
                 profile=CWRamp(fcen, t_ramp=50.0))

# --- run to steady state ---
n_steps = 2200
state = init_tm_state(grid)
state, _ = run_tm(state, coeffs, n_steps, source=src, monitors=[])

# %% [markdown]
# ## Focal-field envelope and focus location
#
# Two Ez snapshots a quarter period apart give the CW envelope,
# |A|^2 = Ez(t)^2 + Ez(t + T/4)^2; the peak is compared against the expected
# focal plane at substrate-top + f.

# %%
# --- focal-field envelope from two snapshots T/4 apart ---
# fcen=2, dt=0.025 -> T/4 = 0.125 = 5 steps
quarter = int(round(0.25 / (fcen * grid.dt)))
Ez1 = state.Ez.clone()
for _ in range(quarter):
    state = step_tm(state, coeffs, src.amplitude(n_steps), src.mask)
    n_steps += 1
Ez2 = state.Ez
envelope = torch.sqrt(Ez1 ** 2 + Ez2 ** 2).detach().cpu().numpy()

# --- focal spot ---
# The metasurface phase is referenced to the substrate top (x1); the blocks are
# thin (h_j < h_wrap ~ 1 um << focal_length), so the focal plane is at x1 + f.
focus_x = x1 + focal_length
ix_focus = int(round(focus_x / dx))
i_peak = np.unravel_index(
    np.argmax(envelope[npml:nx - npml, npml:ny - npml]),
    (nx - 2 * npml, ny - 2 * npml),
)
x_peak = (i_peak[0] + npml) * dx
y_peak = (i_peak[1] + npml) * dx
peak_val = float(envelope[i_peak[0] + npml, i_peak[1] + npml])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(eps.cpu().T, origin="lower", cmap="binary", alpha=0.9,
           extent=[0, sx, 0, sy])
v = float(envelope.max()) or 1.0
ax1.imshow(envelope.T, origin="lower", cmap="hot", alpha=0.55,
           extent=[0, sx, 0, sy], vmin=0, vmax=v)
ax1.axvline(focus_x, color="c", ls="--", lw=0.8, label=f"expected focus x={focus_x:.2f}")
ax1.plot(x_peak, y_peak, "g+", ms=12, mew=2, label=f"peak ({x_peak:.2f},{y_peak:.2f})")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("metasurface lens + focal field envelope |Ez|")
ax1.set_xlabel("x (um)")
ax1.set_ylabel("y (um)")
ax2.plot(np.arange(ny) * dx, envelope[ix_focus, :], "b-", lw=1)
ax2.axvline(y_peak, color="r", ls="--", lw=0.8, label=f"peak y={y_peak:.2f}")
ax2.set_xlabel("y (um)")
ax2.set_ylabel("|Ez| envelope at focal plane")
ax2.set_title(f"transverse focal profile (x={focus_x:.2f} um)")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "metasurface_lens.png", dpi=150)
plt.close(fig)

print("metasurface_lens.png")
print(f"focal peak at x={x_peak:.2f} um (expected {focus_x:.2f}), "
      f"y={y_peak:.2f} um (expected {0.5*sy:.2f}), |Ez|={peak_val:.4e}")