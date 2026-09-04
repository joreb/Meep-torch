# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Third-harmonic generation in a 1D Kerr medium
#
# A plane wave propagates through a chi(3) (Kerr) medium and generates a
# third-harmonic component at 3*f0. The nonlinear polarization P ~ chi3*E^3
# mixes the fundamental into its odd harmonics; we measure both peaks from a
# probe time series via FFT.

# %%
"""meep-torch port of meep/python/examples/3rd-harm-1d.py.

A plane wave propagates through a Kerr (chi3) medium and generates a
third-harmonic component. The meep original uses a 1D simulation with
mp.Medium(index=1, chi3=k) and DFT flux monitors at w and 3w. meep-torch has
no 1D mode and no DFT flux object, so per the port instructions we model it as
1D (thin 2D + PMC on y), paint chi3 over the whole domain via
rasterize_scene(chi3_items=...), pass chi3 into build_tm_coeffs, and measure
the fundamental + 3rd harmonic from a PointProbe time series with a numpy FFT.

meep -> meep-torch mapping:
  sz=100, resolution 25, dpml=1            ->  sz=30 (reduced), dx=0.05, nx=600, npml=20
  mp.Medium(index=1, chi3=k) everywhere    ->  rasterize_scene chi3_items over full cell
  mp.GaussianSource(fcen=1/3, fwidth=df)  ->  GaussianPulse(fcen, fwidth=fcen/5)
  DFT flux @ w, 3w                         ->  PointProbe + numpy FFT peaks
Run: uv run python examples/12_nonlinear/3rd-harm-1d.py
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
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Kerr medium and grid
#
# 1D is modeled as a thin 2D strip with PMC on y. The chi3 nonlinearity is
# painted over the whole cell and passed into the TM update coefficients;
# KERR=1 with a boosted drive amplitude puts us in a strong-drive regime chosen
# purely so the harmonic is visible above the FFT noise floor.

# %%
FCEN = 1.0 / 3.0
KERR = 1.0  # chi3 strength (meep original sweeps 1e-3 .. 1)
AMP = 3.0   # source amplitude boost to make 3rd harmonic visible above noise.
# NOTE: KERR=1, AMP=3 -> chi3*E^2 ~ 9, a STRONG-drive (non-perturbative) regime
# chosen only for harmonic visibility in this demo. The perturbative cube-scaling
# physics (P3 ~ amp^6 ~ P0^3) is verified honestly in tests/examples/test_3rd_harm_1d.py
# (baseline-subtracted, weak drive).

# 1D via thin 2D + PMC on y; sz reduced 100 -> 30 um for runtime
sz = 30.0
res = 20  # resolution (um^-1); original 25
dx = 1.0 / res
grid = Grid(nx=int(sz * res), ny=3, dx=dx)
npml = res  # 1 um PML

# paint chi3 over the whole cell (polygon spanning [0,sz] x [0, ny*dx])
kerr_poly = Polygon(np.array([[0.0, 0.0], [sz, 0.0], [sz, grid.ny * dx], [0.0, grid.ny * dx]]))
rm = rasterize_scene([], grid, background=1.0, chi3_items=[(kerr_poly, KERR)])
coeffs = build_tm_coeffs(
    grid, eps_inf=rm.eps_tm, poles=rm.poles, pole_masks=rm.pole_masks,
    chi3=rm.chi3, npml=npml, y_pmc=True,
)


# %% [markdown]
# ## Source and probe
#
# A Gaussian pulse at f0 = 1/3 drives a line source just inside the left PML;
# a point probe near the right edge records Ez(t) after propagation through
# the nonlinear medium.

# %%
class ScaledPulse:
    def __init__(self, base, amp):
        self.base, self.amp = base, amp

    def amplitude(self, t):
        return self.amp * self.base.amplitude(t)


# broader source (fwidth=fcen/5) than original (fcen/20) to shorten runtime
src = LineSource(
    grid, ix=npml, j0=0, j1=grid.ny,
    profile=ScaledPulse(GaussianPulse(fcen=FCEN, fwidth=FCEN / 5.0), AMP),
)

# monitor near the right edge, just inside the PML
probe = PointProbe(ix=grid.nx - npml - 5, iy=1)

# %% [markdown]
# ## Run and spectral analysis
#
# After 180 time units the Hann-windowed FFT of the probe series shows the
# fundamental at f0 and the generated peak at 3*f0; their ratio quantifies the
# conversion at this drive strength.

# %%
state = init_tm_state(grid, n_poles=len(rm.poles))
n_steps = int(round(180.0 / grid.dt))  # 180 time units
state, (ez_t,) = run_tm(state, coeffs, n_steps, source=src, monitors=(probe,))

ez = ez_t.detach().cpu().numpy()
dt = grid.dt
# FFT of the probe time series
N = ez.size
spec = np.fft.rfft(ez * np.hanning(N))
freqs_fft = np.fft.rfftfreq(N, d=dt)


def peak_near(f_target, arr=freqs_fft, mag=np.abs(spec)):
    idx = np.argmin(np.abs(arr - f_target))
    lo, hi = max(0, idx - 5), min(len(arr), idx + 6)
    return float(mag[lo:hi].max())


fund = peak_near(FCEN)
third = peak_near(3.0 * FCEN)
ratio = third / fund if fund > 0 else float("nan")

# %% [markdown]
# ## Plot
#
# Log-scale spectrum with the fundamental and third-harmonic frequencies marked.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(freqs_fft, np.abs(spec), lw=0.8)
ax.axvline(FCEN, color="g", ls="--", alpha=0.7, label=f"f0 = {FCEN:.3f}")
ax.axvline(3 * FCEN, color="r", ls="--", alpha=0.7, label=f"3 f0 = {3*FCEN:.3f}")
ax.set_title(f"3rd-harmonic spectrum (chi3={KERR}, amp={AMP}); 3w/1w = {ratio:.2e}")
ax.set_xlabel("frequency (1/um)")
ax.set_ylabel("|FFT(Ez)|")
ax.set_xlim(0, 6 * FCEN)
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "3rd-harm-1d.png", dpi=150)
plt.close(fig)

print(f"3rd-harm-1d.png  fundamental={fund:.3e}  3rd-harmonic={third:.3e}  ratio={ratio:.3e}")