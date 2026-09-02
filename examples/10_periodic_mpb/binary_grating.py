# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Binary grating diffraction via mode decomposition
#
# Transmittance per diffraction order of a glass binary grating (period 10,
# height 0.5, duty 0.5), following meep's two-run structure: a
# substrate-only reference run gives the incident power, then the grating
# run's forward mode amplitudes give the per-order transmittance. With PML
# rather than Bloch y-walls, the PEC-box modes of the output cross-section
# stand in for the even-Y diffraction orders.

# %%
"""meep-torch port of meep/python/examples/binary_grating.py.

Binary grating (glass substrate + single ridged period) diffraction via
mode-decomposition of the transmitted field. The meep original uses a
Bloch-periodic y boundary (one grating period, k_point=0, Mirror(Y)) and reads
per-order eigenmode coefficients across nfreq=21 frequencies, nmode=10 orders.

meep-torch has no Bloch-periodic y boundary (only PML/PMC), so the y walls here
are PML: the cell is a *single-period finite grating*, and the modes solve_1d
returns at the output port (in air) are the PEC-box modes of the uniform
post-grating cross-section -- the 0th is the forward planewave (0th order),
higher modes approximate the even-Y diffraction orders of the periodic grating.
Normal incidence (k=0) is preserved. Per the port instructions the per-order
power is |decompose_forward[m]|^2 normalized by the reference (substrate-only)
run, matching the meep two-run structure (input_flux = reference forward flux).

Launch is a full-y LineSource (planewave in the glass substrate), NOT
ModeSourceTM: the substrate is a uniform slab with no guided mode
(neff_box <= n_clad), so ModeSourceTM cannot inject. The meep original also uses
a line source.

Reduced for runtime: resolution 20 (dx=0.05) vs meep 60, nfreq=5 vs 21,
n_modes=5 vs 10, n_steps=2500. Run: uv run python examples/10_periodic_mpb/binary_grating.py
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
# ## Geometry: substrate slab + single ridge

# %%
# --- geometry (meep units: um) ---
dpml, dsub, dpad = 1.0, 3.0, 3.0
gh, gp, gdc = 0.5, 10.0, 0.5
sx = dpml + dsub + gh + dpad + dpml          # 8.5
sy = gp                                      # 10.0
n_glass = 1.5
eps_glass = n_glass ** 2                      # 2.25

dx = 0.05                                     # resolution 20 (meep: 60)
nx = int(round(sx / dx))                       # 170
ny = int(round(sy / dx))                       # 200
npml = int(round(dpml / dx))                   # 20
grid = Grid(nx=nx, ny=ny, dx=dx)

# substrate: glass slab x in [0, dpml+dsub]=[0,4], full y.
# ridge: glass block x in [4, 4.5], y in [2.5, 7.5] (gdc*gp=5, centered).
sub = Polygon(np.array([[0.0, 0.0], [dpml + dsub, 0.0],
                        [dpml + dsub, sy], [0.0, sy]]))
ridge = Polygon(np.array([[dpml + dsub, 0.5 * (sy - gdc * gp)],
                          [dpml + dsub + gh, 0.5 * (sy - gdc * gp)],
                          [dpml + dsub + gh, 0.5 * (sy + gdc * gp)],
                          [dpml + dsub, 0.5 * (sy + gdc * gp)]]))


def build_eps(with_ridge: bool) -> torch.Tensor:
    items = [(sub, eps_glass)]
    if with_ridge:
        items.append((ridge, eps_glass))
    return rasterize_scene(items, grid, background=1.0).eps_tm


# %% [markdown]
# ## Broadband source, output monitor, and the two runs
#
# A full-y planewave line source in the substrate covers 0.4-0.6 um; the
# output flux monitor in the air pad is decomposed into 5 PEC-box modes.
# The reference (no-ridge) run normalizes the grating run per frequency.

# %%
# --- source / monitor ---
wvl_min, wvl_max = 0.4, 0.6
fmin, fmax = 1.0 / wvl_max, 1.0 / wvl_min
fcen, df = 0.5 * (fmin + fmax), fmax - fmin
nfreq = 5
freqs = np.linspace(fmin, fmax, nfreq)

# LineSource in the substrate (x = dpml+0.5*dsub = 2.5 -> ix=50), full y interior.
src_ix = int(round((dpml + 0.5 * dsub) / dx))
src = LineSource(grid, ix=src_ix, j0=npml, j1=ny - npml,
                 profile=GaussianPulse(fcen, df))
# Output flux monitor in the post-grating air pad (x = sx-dpml-0.5*dpad = 6.5 -> ix=130).
out_idx = int(round((sx - dpml - 0.5 * dpad) / dx))
mon = FluxMonitorTM(grid, freqs.tolist(), x_normal=True,
                    idx=out_idx, lo=npml, hi=ny - npml)

n_steps = 2500
n_modes = 5


def run_and_decompose(eps: torch.Tensor):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    state, (acc,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                            source=src, monitors=[mon])
    eps_line = eps[out_idx, npml:ny - npml].detach()
    a_p, _ = decompose(acc, eps_line, grid.dx, freqs.tolist(), "tm", n_modes,
                       axis_sign=-1.0)
    return a_p  # (nf, n_modes) complex forward amplitudes


# reference run (substrate only) -> total incident forward power (meep: input_flux).
# The uniform LineSource does NOT match PEC-box mode 0 (sin(pi*y/W)), so the
# incident planewave projects across many PEC modes; normalizing by the *total*
# reference forward power (sum over modes) is the correct analogue of meep's
# input_flux and yields per-mode T that sums to the total transmittance.
eps_ref = build_eps(with_ridge=False)
a_ref = run_and_decompose(eps_ref)                 # (nf, n_modes)
P_in = (np.abs(a_ref.numpy()) ** 2).sum(axis=1)     # total incident power per freq
# grating run (substrate + ridge) -> per-order forward power
eps_grt = build_eps(with_ridge=True)
a_grt = run_and_decompose(eps_grt)

# %% [markdown]
# ## Per-order transmittance map

# %%
# per-mode transmittance = |a_grt[m]|^2 / P_in[f]  (meep: |coeff|^2 / input_flux)
T_mode = (np.abs(a_grt.numpy()) ** 2) / (P_in[:, None] + 1e-30)
wl = 1.0 / freqs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.imshow(eps_grt.T, origin="lower", cmap="binary", alpha=0.9,
           extent=[0, sx, 0, sy])
ax1.axvline(src_ix * dx, color="g", ls="--", lw=0.8, label="source")
ax1.axvline(out_idx * dx, color="r", ls="--", lw=0.8, label="monitor")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("binary grating (substrate + ridge, glass n=1.5)")
ax1.set_xlabel("x (um)")
ax1.set_ylabel("y (um)")
vmax = max(T_mode.max(), 1e-6)
im = ax2.pcolormesh(wl, np.arange(n_modes), T_mode.T,
                    cmap="Blues", shading="nearest", vmin=0, vmax=vmax)
ax2.set_xlabel("wavelength (um)")
ax2.set_ylabel("output mode index m (~diffraction order)")
ax2.set_title("transmittance per order |a_m|^2 / |a_ref0|^2")
fig.colorbar(im, ax=ax2)
fig.tight_layout()
fig.savefig(OUT / "binary_grating.png", dpi=150)
plt.close(fig)

i_mid = nfreq // 2
print("binary_grating.png")
print(f"at fcen={fcen:.4f} (wvl={wl[i_mid]:.3f}): "
      f"T0={T_mode[i_mid, 0]:.4f}, sum_T={T_mode[i_mid].sum():.4f}")