# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Bend flux: transmission around a 90-degree bend
#
# meep-torch port of meep/python/examples/bend-flux.py. Transmission around a
# 90-degree waveguide bend in 2D. The meep original does two runs (straight
# reference + bend) with flux subtraction; here we use the meep-torch S-matrix
# driver (compute_smatrix), which mode-decomposes the reflected and transmitted
# fields directly, so R = |S11|^2 and T = |S21|^2 come from a single bend run
# with two ports. Expect T ~ 0.9 near fcen with the remainder mostly forward
# radiation; output: `output/bend-flux.png`.
#
# meep -> meep-torch mapping:
# ```
#   cell 16x32, resolution 10              ->  Grid(nx=160, ny=320, dx=0.1)
#   PML(1.0)                                ->  npml=10
#   horiz wvg: Block(sx-pad=12, w=1) @ (-2, wvg_ycen=-11.5)
#                                           ->  eps[0:120, 40:50] = 12
#   vert  wvg: Block(w=1, sy-pad=28) @ (wvg_xcen=3.5, pad*0.5=2)
#                                           ->  eps[110:120, 40:320] = 12
#   port 1 (left,  x-normal @ x=-6.5)       ->  Port(axis="x", index=20, lo=25, hi=65, dir=+1)
#   port 2 (top,   y-normal @ y=14.5)       ->  Port(axis="y", index=280, lo=95, hi=135, dir=-1)
#   GaussianSource(fcen=0.15, fwidth=0.1)   ->  GaussianPulse(0.15, 0.1)
# ```
# Run: `uv run python examples/01_basics/bend-flux.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.grid import Grid
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Geometry
# The bend is built from two polygons rasterized with subpixel (Kottke)
# smoothing — essential at the corner, where a staircased edge would radiate
# most of the guided power.

# %%
sx, sy = 16, 32
dpml = 1.0
pad = 4
w = 1
wvg_xcen = 0.5 * (sx - w - 2 * pad)   # 3.5
wvg_ycen = -0.5 * (sy - w - 2 * pad)  # -11.5

grid = Grid(nx=int(sx / 0.1), ny=int(sy / 0.1), dx=0.1)   # 160 x 320
# meep cell is centered at origin; rasterize_scene uses a (0,0) grid origin, so
# shift meep coords by (+sx/2, +sy/2) = (+8, +16). Subpixel (Kottke) smoothing
# at the bend corner is essential -- a staircase 90-degree corner radiates
# ~99% of the guided power; the smoothed corner transmits ~90% (matching meep).
OX, OY = 0.5 * sx, 0.5 * sy
horiz = Polygon(np.array([[ -8 + OX, -12 + OY], [4 + OX, -12 + OY],
                          [ 4 + OX, -11 + OY], [-8 + OX, -11 + OY]]))   # x[0,12] y[4,5]
vert  = Polygon(np.array([[ 3 + OX, -12 + OY], [4 + OX, -12 + OY],
                          [ 4 + OX,  16 + OY], [3 + OX,  16 + OY]]))   # x[11,12] y[4,32]
rm = rasterize_scene([(horiz, 12.0), (vert, 12.0)], grid, background=1.0)
eps = rm.eps_tm

# %% [markdown]
# ## Ports
# Input on the horizontal arm, output on the vertical arm. The port spans are
# 4 um (40 cells) so the guided-mode profile — near cutoff at fcen=0.15, with a
# long evanescent tail — is captured without truncation.

# %%
# port 1: left, x-normal at x=-6.5 -> ix=15
# port 2: top,  y-normal at y=14.5 -> iy=305
# Meep uses a 2*w = 2um (20-cell) flux region and measures total Poynting flux,
# which is truncation-tolerant. meep-torch mode-decomposes, so the port span must
# capture the full guided-mode profile: at fcen=0.15 the eps=12, w=1um mode is near
# the V=pi/2 cutoff with a long evanescent tail (20-cell span truncates the profile
# at 2.7% of peak; 40-cell span truncates at 0.2% and neff converges). Use 40-cell
# (4um) spans centered on each waveguide core.
ports = [
    Port(axis="x", index=20, lo=25, hi=65, direction=+1),    # input,  +x launch; core [40:50]
    Port(axis="y", index=280, lo=95, hi=135, direction=-1),  # output, +y leaves device; b_out=b_p
]

# %% [markdown]
# ## Run: S-matrix sweep
# One Gaussian-pulse run; compute_smatrix records the DFT fields at both ports
# and mode-decomposes them into S-parameters over 25 frequencies.

# %%
fcen, fwidth = 0.15, 0.1
freqs = np.linspace(0.105, 0.195, 25)   # meep nfreq=100; 25 is enough for a clean spectrum
# n_steps: the bend path is ~36 um (port1 -> corner -> port2). At the eps=12
# group velocity (~c/3.4) that is ~125 time units of travel; the GaussianPulse
# (t_end=100) must finish emitting, then the tail propagates through. 7000 steps
# = 350 time units lets the pulse fully pass both monitors and the DFT converge.
# (Multithreaded run is ~16s.)
sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=fcen, fwidth=fwidth,
                     freqs=list(freqs), n_steps=7000, npml=10)
s = sm.s                                # (nf, nports, nports); col=0 is port-1 input
R = (s[:, 0, 0].abs() ** 2).numpy()      # |S11|^2 reflectance
T = (s[:, 1, 0].abs() ** 2).numpy()      # |S21|^2 transmittance

# %% [markdown]
# ## Plot and interpretation
# R/T here are guided-mode quantities; meep's original measures total Poynting
# flux, so its T is larger — the 1-R-T "loss" is mostly forward radiation, not
# absorption.

# %%
wl = 1.0 / freqs
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wl, R, "bo-", label="reflectance |S11|^2", ms=3)
ax.plot(wl, T, "ro-", label="transmittance |S21|^2", ms=3)
ax.plot(wl, 1 - R - T, "go-", label="loss 1-R-T", ms=3)
ax.set_xlabel("wavelength (um)")
ax.set_ylabel("power")
ax.set_title("90-degree bend R/T (meep bend-flux.py -> compute_smatrix)")
ax.legend(loc="upper right")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(OUT / "bend-flux.png", dpi=150)
plt.close(fig)
print("bend-flux.png")
i_fcen = int(np.argmin(np.abs(freqs - fcen)))
# NOTE: R/T here are guided-mode reflectance/transmittance (|S|^2 from the
# mode-overlap decomposition). meep's original measures TOTAL Poynting flux
# through the port plane (guided + radiation), so its T is larger -- the
# 1-R-T "loss" below is mostly forward radiation that crosses port 2 outside
# the guided mode, not absorption. This is the inherent difference between
# mode-decomposed S-parameters (compute_smatrix) and meep's total-flux sweep.
print("at fcen=%.3f: R=%.3f T=%.3f (guided-mode S-params) loss=%.3f" %
      (fcen, R[i_fcen], T[i_fcen], 1 - R[i_fcen] - T[i_fcen]))
