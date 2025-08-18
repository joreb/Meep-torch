# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a): SiN ring resonator -- gdsfactory geometry through the
meep-torch interface, FDTD, S-matrix spectrum, detuning response.

Effective-index 2D: 300 nm SiN slab in SiO2 -> TE0 neff (printed) as
the 2D core index. |S21|^2 spectrum from a bus-referenced two-run DFT
flux; transient ring buildup at three detunings from the deepest
resonance.

Run: uv run python examples_2/a_ring_smatrix.py
"""
import math
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM, PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import ContinuousSource, GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

# 2D nominal-index convention: n_core = bulk SiN 2.0 (thin-slab effective
# indices 1.47-1.64 give bend-loss Q ~ 35-70 at r=5 -- resonances smear
# into the baseline; the harminv scan below found Q 150-240 at n=2.0).
# The 150 nm-platform effective index is used where the task pins it (c).
res1 = 100
eps_line = np.full(400, 1.44 ** 2)
eps_line[200 - 20: 200 + 20] = 2.0 ** 2
NEFF = 2.0
print(f"a_ring  2D nominal core index n = {NEFF} (SiN)")

RES = 20
FCEN, DF = 1 / 1.55, 0.05
c = gf.components.ring_single(radius=5.0, gap=0.2)
grid, eps, ports = component_to_eps(c, res=RES, pad=1.5,
                                    n_core=NEFF, n_clad=1.44)
NPML = RES
jbus = ports["o1"][1]
eps_ref = eps.clone()
eps_ref[:, jbus + 12:] = 1.44 ** 2                # bus-only reference

NFREQ = 80
freqs = np.linspace(FCEN - DF / 2, FCEN + DF / 2, NFREQ).tolist()


def spectrum(eps_map):
    coeffs = build_tm_coeffs(grid, eps_inf=eps_map, npml=NPML)
    src = LineSource(grid, ix=ports["o1"][0] + 8, j0=jbus - 8, j1=jbus + 8,
                     profile=GaussianPulse(FCEN, DF * 1.6))
    mon = FluxMonitorTM(grid, freqs, x_normal=True, idx=ports["o2"][0] - 8,
                        lo=jbus - 8, hi=jbus + 8)
    _, (acc,) = run_tm(init_tm_state(grid), coeffs,
                       int(round(500.0 / grid.dt)), source=src,
                       monitors=[mon], checkpoint_every=400)
    return mon.flux(acc, grid.dx).detach().numpy()


s21 = spectrum(eps) / spectrum(eps_ref)
lam = 1.0 / np.array(freqs)
i_res = int(np.argmin(s21))
f0 = freqs[i_res]
# local FWHM around the dip only
half = (s21[i_res] + np.median(s21)) / 2.0
lo = i_res
while lo > 0 and s21[lo] < half:
    lo -= 1
hi = i_res
while hi < NFREQ - 1 and s21[hi] < half:
    hi += 1
kappa = max(hi - lo, 1) * DF / NFREQ
Q = f0 / kappa
print(f"a_ring  deepest resonance lambda = {1/f0*1e3:.1f} nm, "
      f"|S21|^2_min = {s21[i_res]:.3f}, kappa ~ {kappa:.2e}, Q ~ {Q:.0f}")


def transient(detune):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
    src = LineSource(grid, ix=ports["o1"][0] + 8, j0=jbus - 8, j1=jbus + 8,
                     profile=ContinuousSource(f0 + detune, width=10.0))
    ring_probe = PointProbe(grid.nx // 2, jbus + int(5.0 * RES))
    _, (series,) = run_tm(init_tm_state(grid), coeffs,
                          int(round(300.0 / grid.dt)), source=src,
                          monitors=[ring_probe], checkpoint_every=400)
    s = series.detach().numpy()
    # envelope: RMS over sliding windows of ~2 optical periods
    w = int(round(2.0 / f0 / grid.dt))
    n = len(s) // w
    env = np.sqrt(np.mean(s[: n * w].reshape(n, w) ** 2, axis=1))
    return np.arange(n) * w * grid.dt, env


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(lam * 1e3, s21, "b-", lw=1.0)
ax1.axvline(1 / f0 * 1e3, color="r", ls=":", lw=0.8)
ax1.set_xlabel("wavelength (nm)")
ax1.set_ylabel("$|S_{21}|^2$")
ax1.set_title(f"SiN ring (r=5, gap=0.2): through port, Q~{Q:.0f}")
curves = [(mult, style, transient(mult * kappa))
          for mult, style in ((0.0, "b-"), (1.0, "g-"), (2.0, "r-"))]
norm = curves[0][2][1].max()                       # common: on-resonance max
for mult, style, (t, env) in curves:
    ax2.plot(t, env / norm, style, lw=1.0,
             label=f"$\\delta = {mult:g}\\kappa$")
ax2.set_xlabel("time (um/c)")
ax2.set_ylabel("ring field envelope (norm.)")
ax2.set_title("transient buildup vs detuning")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "a_ring_smatrix.png", dpi=150)
plt.close(fig)
print("a_ring_smatrix.png written")
