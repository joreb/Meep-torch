# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5.5 T2: gyrotropic 1D vector stepper — Faraday rotation vs analytic."""
import cmath
import math

import numpy as np

from meep_torch.core.grid import Grid
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper_1d import build_1d_coeffs, init_1d_state, run_1d

EPSN, F0, GAMMA, SN, B0 = 1.5, 1.0, 1e-6, 0.1, 0.15
FSRC = 0.8


def _kappa_analytic():
    # rotation wavenumber (meep faraday tutorial): kappa = (k- - k+)/2,
    # k_+- = 2 pi f sqrt(eperp -+ eta)
    dfsq = F0 ** 2 - 1j * FSRC * GAMMA - FSRC ** 2
    eperp = EPSN + SN * F0 ** 2 * dfsq / (dfsq ** 2 - (FSRC * B0) ** 2)
    eta = SN * F0 ** 2 * FSRC * B0 / (dfsq ** 2 - (FSRC * B0) ** 2)
    kp = 2 * math.pi * FSRC * cmath.sqrt(eperp - eta).real
    km = 2 * math.pi * FSRC * cmath.sqrt(eperp + eta).real
    return (km - kp) / 2


def test_faraday_rotation_rate():
    grid = Grid(nx=1000, ny=4, dx=0.02)           # L = 20 at res 50
    coeffs = build_1d_coeffs(grid, eps=EPSN, npml=50,
                             gyro=(F0, GAMMA, SN, B0))
    src_ix = 100                                  # x = 2
    probes_ix = [350, 450]                        # d = 5 and 7: both theta < pi/2
    state = init_1d_state(grid)
    prof = ContinuousSource(FSRC)
    n_steps = int(round(100.0 / grid.dt))
    state, series = run_1d(state, coeffs, n_steps, src_ix=src_ix,
                           src_profile=prof, probe_ix=probes_ix)
    t = np.arange(n_steps) * grid.dt
    kern = np.exp(-2j * math.pi * FSRC * t)
    sel = t > 60.0                                # steady state only
    angles = []
    for ey, ez in series:
        py = np.trapezoid(ey[sel] * kern[sel], t[sel])
        pz = np.trapezoid(ez[sel] * kern[sel], t[sel])
        # signed rotation: project pz onto py's phase (linear polarization)
        pz_signed = (pz * np.conj(py)).real / abs(py)
        angles.append(math.atan2(pz_signed, abs(py)))
    kappa = (angles[1] - angles[0]) / ((probes_ix[1] - probes_ix[0]) * grid.dx)
    k_an = _kappa_analytic()
    # magnitude gate; the rotation SENSE is pinned by the oracle profile test
    assert abs(abs(kappa) - k_an) / k_an < 0.02, f"kappa={kappa:.5f} vs {k_an:.5f}"


def test_faraday_profiles_vs_oracle():
    import json
    import pathlib

    from meep_torch.core.stepper_1d import step_1d

    with open(pathlib.Path(__file__).parent / "data" / "faraday.json") as f:
        ref = json.load(f)
    grid = Grid(nx=1000, ny=4, dx=0.02)
    coeffs = build_1d_coeffs(grid, eps=ref["epsn"], npml=50,
                             gyro=(ref["f0"], ref["gamma"], ref["sn"], ref["b0"]))
    state = init_1d_state(grid)
    # meep ContinuousSource: cos(2 pi f t), abrupt turn-on (width=0); our
    # ContinuousSource is sin + ramped -- match meep's carrier exactly.
    # Global minus: meep's unit-amplitude current produces the -1 polarity
    # of our soft +J injection (source convention, not physics).
    src_ix = int(round((ref["src_z"] + ref["L"] / 2) / grid.dx))
    for t in range(int(round(ref["tmax"] / grid.dt))):
        amp = -math.cos(2 * math.pi * ref["fsrc"] * (t + 0.5) * grid.dt)
        state = step_1d(state, coeffs, amp, src_ix)
    got_y = state.Ey.cpu().numpy()                     # meep Ex -> our Ey
    got_z = state.Ez.cpu().numpy()                     # meep Ey -> our Ez
    ref_x = np.array(ref["ex"])
    ref_y = np.array(ref["ey"])
    n_ours = max(abs(got_y).max(), abs(got_z).max())
    n_ref = max(abs(ref_x).max(), abs(ref_y).max())
    core = slice(100, 900)                       # exclude PML regions
    rms = math.sqrt(float(
        np.mean((got_y[core] / n_ours - ref_x[core] / n_ref) ** 2
                + (got_z[core] / n_ours - ref_y[core] / n_ref) ** 2)))
    assert rms <= 0.05, f"profile rms vs oracle {rms:.4f}"
