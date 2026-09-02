# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Binary grating phase map: duty-cycle sweep
#
# The design curve behind duty-cycle metasurfaces: sweep the duty cycle of a
# subwavelength grating (gp = 0.35 um) and map the 0th-order transmittance
# and phase vs wavelength. Phase is measured relative to a flat-substrate
# reference run, which cancels the source-to-monitor propagation phase.

# %%
"""meep-torch port of meep/python/examples/binary_grating_phasemap.py.

Sweep the binary-grating duty cycle and record the 0th-order (PEC-box mode 0)
transmittance and phase vs wavelength. The meep original uses Bloch-periodic y
(eigenmode coefficients of the periodic grating, one period); meep-torch has no
Bloch y, so the y walls here are PMC (y_pmc=True) -- a mirror in H, i.e. even-Ez,
matching the meep original's Mirror(Y) symmetry. The grating period is
subwavelength (gp=0.35 um) and PML-on-y would not fit (dpml=1.0 > gp), so PMC y
(no y PML) is the only viable boundary. The PEC mode solver (solve_1d, Dirichlet)
is boundary-mismatched to the PMC FDTD walls, but PEC mode 0 (sin(pi*y/sy),
even about the cell center) captures the even-symmetry field; the decomposition
is a sine-Fourier basis, an approximation of the Bloch diffraction orders.

Two runs per duty cycle (reference substrate-only + grating), matching the meep
original's two-run input_flux / mode_coeff structure. Phase is the relative
phase of the grating 0th-order coefficient vs the reference 0th-order
coefficient (angle(a_grt[0] / a_ref[0])), which removes the source-to-monitor
propagation phase.

Reduced for runtime: resolution 50 (dx=0.02) vs meep 60, nfreq=5 vs 21,
gdc = 7 values (0.15..0.85) vs 9, n_steps=2500.
Run: uv run python examples/10_periodic_mpb/binary_grating_phasemap.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.ports.mode_monitor import decompose

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Cell with PMC y-walls
#
# The subwavelength period cannot fit a y-PML (dpml > gp), so the y walls are
# PMC — an even-Ez mirror matching meep's Mirror(Y) symmetry. Only the
# fundamental mode propagates across the band, so n_modes = 1.

# %%
# --- geometry (meep units: um) ---
dpml, dsub, dpad = 1.0, 3.0, 3.0
gh, gp = 0.6, 0.35                # meep defaults: gh=0.6, gp=0.35
sx = dpml + dsub + gh + dpad + dpml          # 8.5
sy = gp                                      # 0.35
n_glass = 1.5
eps_glass = n_glass ** 2

dx = 0.02                                     # resolution 50 (meep: 60); gp=0.35 needs dx<<gp
nx = int(round(sx / dx))                       # 425
ny = int(round(sy / dx))                       # 18
npml = int(round(dpml / dx))                   # 50
grid = Grid(nx=nx, ny=ny, dx=dx)

# source / monitor
wvl_min, wvl_max = 0.4, 0.6
fmin, fmax = 1.0 / wvl_max, 1.0 / wvl_min
fcen, df = 0.5 * (fmin + fmax), fmax - fmin
nfreq = 5
freqs = np.linspace(fmin, fmax, nfreq)

src_ix = int(round((dpml + 0.5 * dsub) / dx))
src = LineSource(grid, ix=src_ix, j0=0, j1=ny,
                 profile=GaussianPulse(fcen, df))
out_idx = int(round((sx - dpml - 0.5 * dpad) / dx))
mon = FluxMonitorTM(grid, freqs.tolist(), x_normal=True,
                    idx=out_idx, lo=0, hi=ny)

n_steps = 2500
n_modes = 1  # only the fundamental PEC mode propagates across the full band


def build_eps(gdc: float) -> torch.Tensor:
    sub = Polygon(np.array([[0.0, 0.0], [dpml + dsub, 0.0],
                            [dpml + dsub, sy], [0.0, sy]]))
    ridge = Polygon(np.array([[dpml + dsub, 0.5 * (sy - gdc * gp)],
                              [dpml + dsub + gh, 0.5 * (sy - gdc * gp)],
                              [dpml + dsub + gh, 0.5 * (sy + gdc * gp)],
                              [dpml + dsub, 0.5 * (sy + gdc * gp)]]))
    return rasterize_scene([(sub, eps_glass), (ridge, eps_glass)],
                           grid, background=1.0).eps_tm


def run_decompose(eps: torch.Tensor):
    # y_pmc=True: PMC y walls (even-Ez mirror), PML on x only. Matches meep's
    # Mirror(Y) for the even-Y Ez source; required because dpml=1.0 > gp=0.35.
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml, y_pmc=True)
    state, (acc,) = run_tm(init_tm_state(grid, y_pmc=True), coeffs, n_steps,
                            source=src, monitors=[mon])
    eps_line = eps[out_idx, 0:ny].detach()
    a_p, _ = decompose(acc, eps_line, grid.dx, freqs.tolist(), "tm", n_modes,
                       axis_sign=-1.0)
    return a_p


# %% [markdown]
# ## Duty-cycle sweep
#
# One reference run plus one grating run per duty cycle (7 values,
# 0.15..0.85); T0 = |a0|^2 / P_in and the relative phase
# angle(a_grt0 / a_ref0) is wrapped to [-2 pi, 0] like the meep original.

# %%
gdcs = np.arange(0.15, 0.86, 0.1)              # 7 values
mode_tran = np.empty((len(gdcs), nfreq))
mode_phase = np.empty((len(gdcs), nfreq))
# reference (flat substrate) run once -- ridge-free for all duty cycles
eps_ref = rasterize_scene(
    [(Polygon(np.array([[0.0, 0.0], [dpml + dsub, 0.0],
                        [dpml + dsub, sy], [0.0, sy]])), eps_glass)],
    grid, background=1.0).eps_tm
a_ref = run_decompose(eps_ref)
P_in = (np.abs(a_ref.numpy()) ** 2).sum(axis=1)

for k, gdc in enumerate(gdcs):
    eps_g = build_eps(float(gdc))
    a_g = run_decompose(eps_g)
    g_np = a_g.numpy()
    mode_tran[k, :] = (np.abs(g_np[:, 0]) ** 2) / (P_in + 1e-30)
    # relative phase of the 0th order vs the reference 0th order
    rel = g_np[:, 0] / (a_ref[:, 0].numpy() + 1e-30)
    ph = np.angle(rel)
    # wrap to [-2*pi, 0] like the meep original
    ph = np.where(ph > 0, ph - 2 * np.pi, ph)
    mode_phase[k, :] = ph
    print(f"  gdc={gdc:.2f}: T0={mode_tran[k, nfreq//2]:.4f} "
          f"phase={mode_phase[k, nfreq//2]:.3f}")

# %% [markdown]
# ## Transmittance / phase maps

# %%
wl = 1.0 / freqs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
im1 = ax1.pcolormesh(wl, gdcs, mode_tran, cmap="hot_r", shading="gouraud",
                     vmin=0, vmax=max(mode_tran.max(), 1e-6))
ax1.set_xlabel("wavelength (um)")
ax1.set_ylabel("grating duty cycle")
ax1.set_title("0th-order transmittance")
fig.colorbar(im1, ax=ax1)
vmax = max(abs(mode_phase).max(), 1e-6)
im2 = ax2.pcolormesh(wl, gdcs, mode_phase, cmap="RdBu", shading="gouraud",
                     vmin=-vmax, vmax=vmax)
ax2.set_xlabel("wavelength (um)")
ax2.set_ylabel("grating duty cycle")
ax2.set_title("0th-order phase (rad) vs reference")
fig.colorbar(im2, ax=ax2)
fig.tight_layout()
fig.savefig(OUT / "binary_grating_phasemap.png", dpi=150)
plt.close(fig)

i_mid = nfreq // 2
print("binary_grating_phasemap.png")
print(f"at wvl={wl[i_mid]:.3f}: T0 range = {mode_tran[:, i_mid].min():.3f}.."
      f"{mode_tran[:, i_mid].max():.3f}, phase range = "
      f"{mode_phase[:, i_mid].min():.2f}..{mode_phase[:, i_mid].max():.2f} rad")