# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (b): SiN + LiNbO3 ring resonators with chi2 / chi3.

Same gdsfactory racetrack as (a), core index 2.2 (LiNbO3, 2D nominal):
  * SHG: CW pump on a ring resonance -> spectrum grows a line at 2 f0
    (chi2, the a44f0e2 core addition; P(2f0) ~ chi2^2 P(f0)^2).
  * Kerr: chi3 ring pumped at two powers -> the resonance red-shifts
    with intensity (harminv'd, shift ~ A^2).

Run: uv run python examples_2/b_ring_nonlinear.py
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
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import ContinuousSource, GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.harminv import harminv

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()
RES = 20
N_LN = 2.2

c = gf.components.ring_single(radius=5.0, gap=0.2)
grid, eps, ports = component_to_eps(c, res=RES, pad=1.5,
                                    n_core=N_LN, n_clad=1.44)
core_mask = (eps > 3.0).to(eps.dtype)             # LiNbO3 regions
jbus = ports["o1"][1]
J_RING = (grid.nx // 2, 259)                      # top ring waveguide


def find_resonance():
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES)
    src = LineSource(grid, ix=grid.nx // 2, j0=258, j1=261,
                     profile=GaussianPulse(1 / 1.55, 0.08))
    _, (s,) = run_tm(init_tm_state(grid), coeffs,
                     int(round(600.0 / grid.dt)), source=src,
                     monitors=[PointProbe(*J_RING)], checkpoint_every=400)
    tail = s.detach().numpy()[int(round(120.0 / grid.dt)):]
    modes = sorted([m for m in harminv(tail, grid.dt, 0.60, 0.69) if m.Q > 80],
                   key=lambda m: -m.amp)
    return modes[0]


m0 = find_resonance()
F0 = m0.freq
print(f"b_ring  LiNbO3 ring resonance f0={F0:.5f} (lam={1e3/F0:.1f} nm) "
      f"Q={m0.Q:.0f}")


def pumped_spectrum(chi2_val, chi3_val, amp):
    kw = {}
    if chi2_val:
        kw["chi2"] = chi2_val * core_mask
    if chi3_val:
        kw["chi3"] = chi3_val * core_mask
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES, **kw)
    src = LineSource(grid, ix=ports["o1"][0] + 8, j0=jbus - 8, j1=jbus + 8,
                     profile=ContinuousSource(F0, width=15.0))
    src.mask = src.mask * amp
    _, (s,) = run_tm(init_tm_state(grid), coeffs,
                     int(round(400.0 / grid.dt)), source=src,
                     monitors=[PointProbe(*J_RING)], checkpoint_every=400)
    sig = s.detach().numpy()[int(round(100.0 / grid.dt)):]
    spec = np.abs(np.fft.rfft(sig)) ** 2
    fr = np.fft.rfftfreq(len(sig), grid.dt)
    return fr, spec / spec.max()


# --- SHG (chi2) ---
fr, spec = pumped_spectrum(chi2_val=0.5, chi3_val=0.0, amp=1.0)


def band_peak(fr, spec, f, w=0.05):
    m = (fr > f - w) & (fr < f + w)
    return float(spec[m].max())


p_f0 = band_peak(fr, spec, F0)
p_2f0 = band_peak(fr, spec, 2 * F0)
print(f"b_ring  SHG: P(2f0)/P(f0) = {p_2f0 / p_f0:.2e}")

# --- Kerr shift (chi3) ---
shifts = []
for amp in (0.5, 2.0):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES,
                             chi3=0.05 * core_mask)
    src = LineSource(grid, ix=ports["o1"][0] + 8, j0=jbus - 8, j1=jbus + 8,
                     profile=GaussianPulse(F0, 0.03))
    src.mask = src.mask * amp
    _, (s,) = run_tm(init_tm_state(grid), coeffs,
                     int(round(600.0 / grid.dt)), source=src,
                     monitors=[PointProbe(*J_RING)], checkpoint_every=400)
    tail = s.detach().numpy()[int(round(150.0 / grid.dt)):]
    modes = sorted([m for m in harminv(tail, grid.dt, F0 - 0.02, F0 + 0.02)
                    if m.Q > 80], key=lambda m: -m.amp)
    shifts.append(modes[0].freq if modes else float("nan"))
print(f"b_ring  Kerr: f(low power)={shifts[0]:.6f}, f(4x power)={shifts[1]:.6f}, "
      f"shift = {(shifts[1] - shifts[0]) * 1e6:.1f} ppm-units")

fig, ax = plt.subplots(figsize=(7.2, 4))
ax.semilogy(fr, np.maximum(spec, 1e-12), lw=0.8)
ax.axvline(F0, color="g", ls=":", label="pump $f_0$")
ax.axvline(2 * F0, color="r", ls=":", label="SHG $2 f_0$")
ax.set_xlim(0, 3 * F0)
ax.set_xlabel("frequency (c/um)")
ax.set_ylabel("ring spectrum (norm.)")
ax.set_title(f"LiNbO3 ring, chi2 pump at resonance: SHG line at 2f0 "
             f"(P ratio {p_2f0 / p_f0:.1e})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "b_ring_nonlinear.png", dpi=150)
plt.close(fig)
print("b_ring_nonlinear.png written")
