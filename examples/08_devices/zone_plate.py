# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Binary-phase Fresnel zone plate
#
# A zone plate focuses a CW planewave: odd Fresnel zones carry a
# half-wavelength glass bar (~pi phase shift), even zones are open, with radii
# r_n = sqrt(n lambda (f + n lambda / 4)). The focal spot is imaged directly
# in the 2D FDTD domain from the steady-state Ez envelope.

# %%
"""meep-torch port of meep/python/examples/zone_plate.py.

Binary-phase Fresnel zone plate focusing. The meep original is a 3D cylindrical
(mp.CYLINDRICAL, m=-1) simulation using a near-to-far-field transform to image
the focal-plane fields. meep-torch has neither cylindrical coords nor near-to-far,
so this is a 2D Cartesian port: the zone plate is a stack of binary glass bars
along y at the input plane (a 2D slice through the cylindrical zone plate),
illuminated by a CW planewave, and the focal field is imaged *directly* in the
FDTD domain (no n2f) by recording the steady-state Ez envelope.

The zone radii follow the standard Fresnel formula
  r_n = sqrt(n*lambda*(f + n*lambda/4)).
Odd-numbered (1-indexed) zones are glass (n=1.5, ~pi phase shift for the
half-wavelength-thick slab), even zones are open.

The focal field envelope |A(x,y)| is recovered from two steady-state Ez
snapshots T/4 apart: |A|^2 = Ez(t)^2 + Ez(t+T/4)^2 (since Ez=Re[A*e^{iwt}]).

Reduced for runtime (and to fit the focal length in-domain): focal_length=10 um
(meep: 200), num_zones=10 (meep: 25), resolution 20 (dx=0.05, meep: 25). The
focal spot still forms at x ~ substrate+height+focal_length from the left.
Run: uv run python examples/08_devices/zone_plate.py
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
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.core.run import run_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Zone geometry
#
# The substrate slab plus glass bars over the odd-zone y-intervals (mirrored
# about the axis) are rasterized into eps; in this 2D slice each zone n covers
# y in [r_(n-1), r_n].

# %%
# --- zone plate parameters (meep units: um) ---
pml_um = 1.0
substrate_um = 2.0
height_um = 0.5                # glass bar thickness (half-wavelength -> ~pi shift)
focal_length = 10.0            # reduced from meep's 200
wavelength = 0.5
fcen = 1.0 / wavelength
num_zones = 10                 # reduced from meep's 25
n_glass = 1.5
eps_glass = n_glass ** 2
pad_um = 2.0

# Fresnel zone radii
zone_r = np.array([
    math.sqrt(n * wavelength * (focal_length + n * wavelength / 4.0))
    for n in range(1, num_zones + 1)
])
R = zone_r[-1]
sx = pml_um + substrate_um + height_um + focal_length + pad_um + pml_um
sy = 2.0 * (R + pad_um + pml_um)

dx = 0.05                       # resolution 20
nx = int(round(sx / dx))
ny = int(round(sy / dx))
npml = int(round(pml_um / dx))
grid = Grid(nx=nx, ny=ny, dx=dx)

# --- build geometry ---
# substrate: glass slab x in [pml, pml+substrate], full y interior.
# zone bars: at x in [pml+substrate, pml+substrate+height], y in ring intervals.
# 1-indexed odd zones (n=1,3,5,...) -> glass; even zones -> open (vacuum).
# In the 2D slice, each zone n covers y in [r_{n-1}, r_n] and [-r_n, -r_{n-1}]
# (r_0 = 0). Zone n (1-indexed) is glass if n is odd.
items = []
# substrate
x0 = pml_um
x1 = pml_um + substrate_um
items.append((Polygon(np.array([[x0, pml_um], [x1, pml_um],
                                 [x1, sy - pml_um], [x0, sy - pml_um]])),
              eps_glass))
# zone bars (glass for odd 1-indexed zones)
x_zp0 = pml_um + substrate_um
x_zp1 = x_zp0 + height_um
r_prev = 0.0
for n in range(1, num_zones + 1):
    r_n = zone_r[n - 1]
    if n % 2 == 1:   # odd zone -> glass bar (pi phase shift)
        # positive-y ring [r_prev, r_n] and negative-y ring [-r_n, -r_prev]
        yc = 0.5 * sy   # grid center (meep y=0 -> grid y=sy/2)
        for sign in (+1, -1):
            ya = yc + sign * r_prev
            yb = yc + sign * r_n
            ylo, yhi = min(ya, yb), max(ya, yb)
            items.append((Polygon(np.array([
                [x_zp0, ylo], [x_zp1, ylo], [x_zp1, yhi], [x_zp0, yhi]])),
                eps_glass))
    r_prev = r_n

eps = rasterize_scene(items, grid, background=1.0).eps_tm
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)

# %% [markdown]
# ## CW drive to steady state

# %%
# --- source: CW planewave just inside the left PML, full y interior ---
src_ix = int(round((pml_um + 0.3 * substrate_um) / dx))
src = LineSource(grid, ix=src_ix, j0=npml, j1=ny - npml,
                 profile=CWRamp(fcen, t_ramp=50.0))

# --- run to steady state ---
# propagation time ~ sx/c ~ 25 um; ramp ~ 50; total ~ 75 -> 3000 steps
n_steps = 3000
state = init_tm_state(grid)
state, _ = run_tm(state, coeffs, n_steps, source=src, monitors=[])

# %% [markdown]
# ## Envelope, focus location, and plot
#
# Two snapshots T/4 apart give |A|^2 = Ez(t)^2 + Ez(t + T/4)^2; the envelope
# peak is compared with the expected focus at the plate edge + f.

# %%
# --- focal-field envelope from two snapshots T/4 apart ---
# fcen=2, dt=0.025 -> T/4 = 0.125 = 5 steps (integer).
quarter = int(round(0.25 / (fcen * grid.dt)))   # 5
Ez1 = state.Ez.clone()
for _ in range(quarter):
    state = step_tm(state, coeffs, src.amplitude(n_steps), src.mask)
    n_steps += 1
Ez2 = state.Ez
envelope = torch.sqrt(Ez1 ** 2 + Ez2 ** 2).detach().numpy()

# --- focal spot location: peak of envelope in the post-plate region ---
x_zp = x_zp1                       # zone plate right edge
focus_x = x_zp + focal_length
# search window around the expected focal plane
ix_focus = int(round(focus_x / dx))
i_peak = np.unravel_index(
    np.argmax(envelope[npml:nx - npml, npml:ny - npml]),
    (nx - 2 * npml, ny - 2 * npml),
)
x_peak = (i_peak[0] + npml) * dx
y_peak = (i_peak[1] + npml) * dx
peak_val = float(envelope[i_peak[0] + npml, i_peak[1] + npml])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(eps.T, origin="lower", cmap="binary", alpha=0.9,
           extent=[0, sx, 0, sy])
v = float(envelope.max()) or 1.0
ax1.imshow(envelope.T, origin="lower", cmap="hot", alpha=0.55,
           extent=[0, sx, 0, sy], vmin=0, vmax=v)
ax1.axvline(focus_x, color="c", ls="--", lw=0.8, label=f"expected focus x={focus_x:.1f}")
ax1.plot(x_peak, y_peak, "g+", ms=12, mew=2, label=f"peak ({x_peak:.1f},{y_peak:.1f})")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("zone plate + focal field envelope |Ez|")
ax1.set_xlabel("x (um)")
ax1.set_ylabel("y (um)")
# transverse profile at the focal plane
ax2.plot(np.arange(ny) * dx, envelope[ix_focus, :], "b-", lw=1)
ax2.axvline(y_peak, color="r", ls="--", lw=0.8, label=f"peak y={y_peak:.2f}")
ax2.set_xlabel("y (um)")
ax2.set_ylabel("|Ez| envelope at focal plane")
ax2.set_title(f"transverse focal profile (x={focus_x:.1f} um)")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "zone_plate.png", dpi=150)
plt.close(fig)

print("zone_plate.png")
print(f"focal peak at x={x_peak:.2f} um (expected {focus_x:.2f}), "
      f"y={y_peak:.2f} um (expected {0.5*sy:.2f}), |Ez|={peak_val:.4e}")