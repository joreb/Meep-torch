# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/finite_grating.py.

Finite binary grating (5 unit cells) diffraction spectrum. The meep original
records the DFT field (add_dft_fields) on a 1D y-cross-section in the air region
adjacent to the grating for a flat-substrate reference run and the grating run,
subtracts them (scattered = grating - flat), and FFTs the scattered field in y
to get the diffraction spectrum (power vs k_y).

meep-torch has no add_dft_fields (2D/1D DFT field arrays), so we use the
mode-decomposition analogue: a FluxMonitorTM line at the same monitor plane is
decomposed (decompose, PEC sine-Fourier basis) into forward/backward mode
amplitudes for both runs. The scattered mode amplitude is a_grt[m] - a_ref[m]
(complex), and the scattered power per mode is |a_grt[m] - a_ref[m]|^2 -- the
mode-basis analogue of the meep |FFT|^2. The PEC modes (sin(k*pi*y/W)) are a
sine-Fourier basis, so this is equivalent to a sine transform of the scattered
field, capturing the same diffraction-spectrum structure as the meep FFT.

The y boundary is PML on all sides (the meep original also uses PML on all
sides -- the finite grating does not need Bloch periodicity).

Reduced for runtime: resolution 25 (dx=0.04) vs meep 50, n_steps=2500.
Run: uv run python examples/finite_grating.py
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

OUT = pathlib.Path(__file__).parent

# --- geometry (meep units: um) ---
dpml, dsub, dpad = 1.0, 2.0, 1.0
gh, gp, gdc = 0.5, 1.0, 0.5
num_cells = 5
dair = dpad                                   # field_profile=False case (1D cross-section)
sx = dpml + dsub + gh + dair + dpml            # 5.5
sy = dpml + dpad + num_cells * gp + dpad + dpml  # 9.0
n_glass = 1.5
eps_glass = n_glass ** 2

dx = 0.04                                     # resolution 25 (meep: 50)
nx = int(round(sx / dx))                       # 138
ny = int(round(sy / dx))                       # 225
npml = int(round(dpml / dx))                   # 25
grid = Grid(nx=nx, ny=ny, dx=dx)

# source: LineSource in the substrate (x = dpml+0.5*dsub = 2.0), full y interior.
wvl = 0.5
fcen = 1.0 / wvl
fwidth = 0.2 * fcen
src_ix = int(round((dpml + 0.5 * dsub) / dx))
src = LineSource(grid, ix=src_ix, j0=npml, j1=ny - npml,
                 profile=GaussianPulse(fcen, fwidth))
# monitor: 1D y-line in the air region (x = sx-dpml-0.5*dair = 4.0 -> ix=100)
out_idx = int(round((sx - dpml - 0.5 * dair) / dx))
mon = FluxMonitorTM(grid, [fcen], x_normal=True,
                    idx=out_idx, lo=npml, hi=ny - npml)

n_steps = 2500
n_modes = 25  # resolve the diffraction spectrum across ~25 PEC modes

# substrate block (x in [0, dpml+dsub]=[0,3], full y interior)
sub = Polygon(np.array([[0.0, npml * dx], [dpml + dsub, npml * dx],
                        [dpml + dsub, (ny - npml) * dx], [0.0, (ny - npml) * dx]]))


def build_eps(with_grating: bool) -> torch.Tensor:
    items = [(sub, eps_glass)]
    if with_grating:
        for j in range(num_cells):
            yc = -0.5 * sy + dpml + dpad + (j + 0.5) * gp   # meep cell-centered y
            # shift to grid coords (grid y = meep y + sy/2)
            yc_grid = yc + 0.5 * sy
            y0 = yc_grid - 0.5 * gdc * gp
            y1 = yc_grid + 0.5 * gdc * gp
            items.append((Polygon(np.array([
                [dpml + dsub, y0], [dpml + dsub + gh, y0],
                [dpml + dsub + gh, y1], [dpml + dsub, y1]])), eps_glass))
    return rasterize_scene(items, grid, background=1.0).eps_tm


def run_decompose(eps: torch.Tensor):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    state, (acc,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                            source=src, monitors=[mon])
    eps_line = eps[out_idx, npml:ny - npml].detach()
    a_p, _ = decompose(acc, eps_line, grid.dx, [fcen], "tm", n_modes,
                       axis_sign=-1.0)
    return a_p[0]  # (n_modes,) complex amplitudes at fcen


# reference (flat substrate) + grating runs
a_ref = run_decompose(build_eps(with_grating=False))
a_grt = run_decompose(build_eps(with_grating=True))

# scattered mode amplitude = grating - reference (complex subtraction, meep-style)
a_scat = (a_grt - a_ref).numpy()
scat_power = np.abs(a_scat) ** 2
# mode transverse wavenumber k_y = (m+1)*pi/W (PEC Dirichlet, m=0..n_modes-1)
W = (ny - 2 * npml) * dx
k_y = (np.arange(n_modes) + 1) * np.pi / W
k_y_2pi = k_y / (2 * np.pi)   # in units of 2*pi/um

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"hspace": 0.35})
ax1.imshow(build_eps(with_grating=True).T, origin="lower", cmap="binary",
           alpha=0.9, extent=[0, sx, 0, sy])
ax1.axvline(src_ix * dx, color="g", ls="--", lw=0.8, label="source")
ax1.axvline(out_idx * dx, color="r", ls="--", lw=0.8, label="monitor")
ax1.legend(loc="upper right", fontsize=8)
ax1.set_title("finite grating (5 cells, glass n=1.5)")
ax1.set_xlabel("x (um)")
ax1.set_ylabel("y (um)")
ax2.plot(k_y_2pi, scat_power, "ro-", ms=3)
ax2.set_xlabel(r"transverse wavenumber $k_y$, 2$\pi$ ($\mu$m)$^{-1}$")
ax2.set_ylabel("scattered power per PEC mode |a_grt-a_ref|$^2$")
ax2.set_title("diffraction spectrum (mode-decomposition analogue of meep FFT)")
ax2.set_xlim(-0.1, k_y_2pi[-1] + 0.1)
fig.tight_layout()
fig.savefig(OUT / "finite_grating.png", dpi=150)
plt.close(fig)

i_peak = int(np.argmax(scat_power))
print("finite_grating.png")
print(f"scattered peak at mode {i_peak} (k_y={k_y_2pi[i_peak]:.3f} *2pi/um), "
      f"power={scat_power[i_peak]:.4e}")