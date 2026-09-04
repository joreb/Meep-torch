# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch test of meep/python/examples/3rd-harm-1d.py.

Third-harmonic generation in a 1D Kerr medium via the engine's instantaneous
chi3 (fixed-point E-from-D in step_tm). 1D -> 2D grid uniform in y with
y_pmc=True (PML on x only). A finite Gaussian pulse propagates through the
uniform Kerr medium; a FluxMonitorTM at the far end measures DFT flux at f0
and 3*f0.

Physics gates (not golden numbers):
  - P(3f0) grows as P(f0)^3 in the perturbative regime: a third-harmonic
    field E3 ~ a^3 (THG is a three-photon process), so P3 ~ a^6 and
    P0 ~ a^2, giving log-log slope(P3 vs P0) = 3. Sweeping source amplitude
    a at fixed chi3 must recover slope ~3 (within ~10%).
  - Slope(P3 vs a) ~ 6 (within ~10%) -- the direct cube-power signature.
  - With chi3 = 0 the 3rd-harmonic flux vanishes (P3 << P0).

The 3f0 flux has a small boxcar-DFT leakage floor from the strong f0 line
(~1e-10, present even at chi3=0, scaling ~ a^2). It is subtracted per
amplitude: P3_net(a) = P3(a, chi3) - P3(a, 0), which restores the clean
perturbative a^6 / P0^3 scaling over the swept range.

meep original: cell sz=100, fcen=1/3, df=fcen/20, dpml=1, res=25, eps=1+chi3*E^2.
Here sz=80 (NX=2000), dx=0.04 (same res), npml=25 (dpml=1), eps_inf=1.0; the
pulse is broadened (fwidth=fcen/4, cutoff=3) to keep the source spectrum clear
of 3*f0 and the run under ~15 s.
"""
import numpy as np
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

FCEN = 1.0 / 3.0
F3 = 3.0 * FCEN
DX = 0.04
NX = 2000                      # sz = 80 um
NY = 8
NPML = 25                      # dpml = 1 um
N_STEPS = 13000
CHI3 = 0.3
IX_SRC = NPML + 2
IX_MON = NX - NPML - 3
AMPS = [0.4, 0.6, 0.8, 1.0, 1.2]


class _Scaled:
    def __init__(self, base, a):
        self.base, self.a = base, a

    def amplitude(self, t):
        return self.a * self.base.amplitude(t)


def _fluxes(amp, chi3):
    g = Grid(nx=NX, ny=NY, dx=DX)
    m = build_tm_coeffs(g, eps_inf=1.0, chi3=chi3, npml=NPML, y_pmc=True)
    src = LineSource(g, ix=IX_SRC, j0=0, j1=NY,
                     profile=_Scaled(GaussianPulse(FCEN, FCEN / 4.0, cutoff=3.0), amp))
    mon = FluxMonitorTM(g, [FCEN, F3], x_normal=True, idx=IX_MON, lo=0, hi=NY)
    s = init_tm_state(g, y_pmc=True)
    _, (acc,) = run_tm(s, m, N_STEPS, source=src, monitors=[mon])
    P = mon.flux(acc, DX).detach().cpu().numpy()
    return float(P[0]), float(P[1])          # P0, P3


def test_3rd_harmonic_scales_as_cube():
    P0, P3, P3_floor = [], [], []
    for a in AMPS:
        p0, p3 = _fluxes(a, CHI3)
        _, p3f = _fluxes(a, 0.0)
        P0.append(p0); P3.append(p3); P3_floor.append(p3f)
    P0 = np.array(P0)
    P3_net = np.array(P3) - np.array(P3_floor)       # remove f0->3f0 leakage floor
    assert np.all(np.diff(P0) > 0), f"P0 not monotone in amp: {P0}"
    assert np.all(P3_net > 0), f"P3_net nonpositive after floor subtraction: {P3_net}"
    slope_p0 = np.polyfit(np.log(P0), np.log(P3_net), 1)[0]
    slope_amp = np.polyfit(np.log(AMPS), np.log(P3_net), 1)[0]
    assert 2.7 < slope_p0 < 3.3, f"THG slope(P3 vs P0)={slope_p0:.3f}, expected ~3"
    assert 5.4 < slope_amp < 6.6, f"THG slope(P3 vs amp)={slope_amp:.3f}, expected ~6"
    print(f"3rd-harm: amps={AMPS}")
    print(f"  P0={P0.tolist()}")
    print(f"  P3_net={P3_net.tolist()}")
    print(f"  slope(P3 vs P0)={slope_p0:.3f}  slope(P3 vs amp)={slope_amp:.3f}")
    print(f"  P3_net/P0 at amp=1.0 = {P3_net[3] / P0[3]:.3e}")


def test_3rd_harmonic_absent_without_chi3():
    P0, P3 = _fluxes(1.0, chi3=0.0)
    assert P3 < 1e-6 * P0, f"THG present without chi3: P3={P3:.3e} P0={P0:.3e}"