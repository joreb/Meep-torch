# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5.75 chi2 (instantaneous second-order) nonlinear stepper tests.

Gate (i): chi2=0 is bit-identical to the linear fast path (the chi2
plumbing does not perturb the linear solver).
Gate (ii): SHG -- a CW pump at f0 through a chi2 slab generates a 2f0
field whose power P(2f0) ~ A^4 (undepleted pump: E(2w) ~ chi2 * E(w)^2
~ chi2 * A^2, so P ~ |E(2w)|^2 ~ A^4) within 20% across A and 2A.
Gate (iii): P(2f0) ~ chi2^2 within 20% across two chi2 values.
"""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import CustomSource, SoftSource
from meep_torch.core.stepper import init_tm_state

F0 = 0.3
NX, NY, DX, NPML = 600, 3, 0.05, 40
GRID = Grid(nx=NX, ny=NY, dx=DX)
SLAB0, SLAB1 = 280, 320          # chi2 slab (vacuum eps, pure chi2)
SRC_IX, PROBE_IX = 150, 380
T_RAMP = 20.0                    # CW turn-on ramp (smooth -> minimize broadband)
T_END = 400.0


def _cw_profile(amp):
    def fn(t):
        s = min(1.0, t / T_RAMP)
        ramp = 0.5 - 0.5 * math.cos(math.pi * s)
        return amp * math.sin(2 * math.pi * F0 * t) * ramp
    return CustomSource(fn)


def _run(A, chi2_val):
    eps = torch.ones(NX, NY, dtype=GRID.dtype)
    chi2 = torch.zeros(NX, NY, dtype=GRID.dtype)
    chi2[SLAB0:SLAB1, :] = chi2_val
    coeffs = build_tm_coeffs(GRID, eps_inf=eps, chi2=chi2, npml=NPML, y_pmc=True)
    mask = GRID.zeros(NX, NY)
    mask[SRC_IX, 1] = 1.0
    src = SoftSource(GRID, mask, _cw_profile(A))
    n_steps = int(round(T_END / GRID.dt))
    _, (series,) = run_tm(init_tm_state(GRID, y_pmc=True), coeffs, n_steps,
                          source=src, monitors=[PointProbe(PROBE_IX, 1)])
    return series.detach().numpy()


def _p2f0(series):
    """Power in the 2f0 component of the late (steady-state) probe window."""
    t0 = int(round(150.0 / GRID.dt))     # drop turn-on + transit transient
    e = series[t0:]
    t = (np.arange(len(e)) + t0) * GRID.dt
    e2f0 = np.sum(e * np.exp(-1j * 2 * math.pi * (2 * F0) * t)) * GRID.dt
    return abs(e2f0) ** 2


def test_chi2_zero_bit_parity():
    """Gate (i): chi2=0 fixed-point loop is bit-identical to the fast path."""
    eps = torch.ones(NX, NY, dtype=GRID.dtype)
    coeffs_lin = build_tm_coeffs(GRID, eps_inf=eps, npml=NPML, y_pmc=True)
    coeffs_chi2 = build_tm_coeffs(GRID, eps_inf=eps, chi2=0.0, npml=NPML,
                                  y_pmc=True)
    # force the chi2 fixed-point path on a copy with has_chi2=True but chi2=0
    coeffs_fp = coeffs_chi2._replace(chi2=torch.zeros_like(eps), has_chi2=True)
    mask = GRID.zeros(NX, NY)
    mask[SRC_IX, 1] = 1.0
    src = SoftSource(GRID, mask, _cw_profile(0.5))
    n_steps = int(round(120.0 / GRID.dt))
    _, (a,) = run_tm(init_tm_state(GRID, y_pmc=True), coeffs_lin, n_steps,
                     source=src, monitors=[PointProbe(PROBE_IX, 1)])
    _, (b,) = run_tm(init_tm_state(GRID, y_pmc=True), coeffs_fp, n_steps,
                     source=src, monitors=[PointProbe(PROBE_IX, 1)])
    assert torch.equal(a, b), "chi2=0 fixed point != linear fast path"


def test_shg_power_A4_scaling():
    """Gate (ii): P(2f0) ~ A^4 within 20% across A and 2A (undepleted pump)."""
    chi2_val = 0.1
    p1 = _p2f0(_run(0.5, chi2_val))
    p2 = _p2f0(_run(1.0, chi2_val))
    print(f"SHG A-scaling: P(A)={p1:.4g} P(2A)={p2:.4g} ratio={p2/p1:.3f} (want 16)")
    assert p1 > 1e-12 and p2 > 1e-12, "no 2f0 signal generated"
    ratio = p2 / p1
    assert 12.8 <= ratio <= 19.2, f"P(2f0) A^4 scaling ratio {ratio:.3f} not ~16"


def test_shg_power_chi2_squared_scaling():
    """Gate (iii): P(2f0) ~ chi2^2 within 20% across two chi2 values (A fixed)."""
    A = 0.5
    p1 = _p2f0(_run(A, 0.1))
    p2 = _p2f0(_run(A, 0.2))
    print(f"SHG chi2-scaling: P(chi2)={p1:.4g} P(2chi2)={p2:.4g} "
          f"ratio={p2/p1:.3f} (want 4)")
    ratio = p2 / p1
    assert 3.2 <= ratio <= 4.8, f"P(2f0) chi2^2 scaling ratio {ratio:.3f} not ~4"