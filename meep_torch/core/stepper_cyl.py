# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""1D radial body-of-revolution stepper (fields F(r) e^{i m phi}, d/dz = 0).

Ez-polarized set {Ez, Hr, Hphi}, complex fields (the i m / r curl terms):

  dBr/dt   = -(i m / r) Ez        Hr, Ez at r_i = i dr
  dBphi/dt = + dEz/dr             Hphi at r_{i+1/2}
  dDz/dt   = (r_{i+1/2} Hphi_i - r_{i-1/2} Hphi_{i-1})/(r_i dr) - (i m/r_i) Hr

Axis (m != 0): Ez(0) = Hr(0) = 0 by e^{i m phi} regularity — the axis row
is pinned, not updated. m = 0 (ampere-loop axis) ships with W7.5.
Outer boundary: matched sigma_d = sigma_b quadratic absorber ramp (the
meep ring-cyl tutorial uses dpml = 32 + pad 4; plain absorption at that
depth suffices — W7.5 brings cylindrical CPML). Courant dt = 0.5 dr.
"""
import math
from typing import NamedTuple

import torch

from .pml import absorber_sigma
from .sources import bind_dt
from .grid import default_device


class CylCoeffs(NamedTuple):
    dt: float
    dr: float
    m: int
    i_min: int              # axis rows pinned to zero (regularity + stability)
    eps: torch.Tensor       # (nr,) real
    inv_r: torch.Tensor     # (nr,) 1/r_i (0 at the pinned axis row)
    r_half: torch.Tensor    # (nr-1,) r_{i+1/2}
    dz1: torch.Tensor       # (nr,) Dz damping (absorber)
    dz2: torch.Tensor       # (nr,) Dz curl coefficient
    h1: torch.Tensor        # (nr,) Hr damping
    h2: torch.Tensor        # (nr,)
    hp1: torch.Tensor       # (nr-1,) Hphi damping
    hp2: torch.Tensor       # (nr-1,)


class CylState(NamedTuple):
    Ez: torch.Tensor        # (nr,) complex
    Hr: torch.Tensor        # (nr,) complex
    Hphi: torch.Tensor      # (nr-1,) complex
    Dz: torch.Tensor        # (nr,) complex


def build_cyl_coeffs(nr: int, dr: float, m: int, eps=1.0,
                     n_absorber: int = 0, dtype=torch.float64,
                     device=None) -> CylCoeffs:
    device = device or default_device()
    if m == 0:
        raise NotImplementedError("m = 0 axis update lands with W7.5")
    dt = 0.5 * dr
    e = (eps if isinstance(eps, torch.Tensor)
         else torch.full((nr,), float(eps), dtype=dtype, device=device))
    r = torch.arange(nr, dtype=dtype, device=device) * dr
    inv_r = torch.zeros(nr, dtype=dtype, device=device)
    inv_r[1:] = 1.0 / r[1:]
    r_half = (torch.arange(nr - 1, dtype=dtype, device=device) + 0.5) * dr

    # one-sided (outer) absorber ramp: build both-ended profile, zero the axis side
    sig = absorber_sigma(nr, n_absorber, dr, dtype=dtype, device=device)
    sig[: nr // 2] = 0.0
    sig_h = 0.5 * (sig[:-1] + sig[1:])
    dz1 = (1 - sig * dt / 2) / (1 + sig * dt / 2)
    dz2 = dt / (1 + sig * dt / 2)
    h1 = (1 - sig * dt / 2) / (1 + sig * dt / 2)
    h2 = dt / (1 + sig * dt / 2)
    hp1 = (1 - sig_h * dt / 2) / (1 + sig_h * dt / 2)
    hp2 = dt / (1 + sig_h * dt / 2)
    # the (i m / r) axis coupling rotates at omega = m / r; leapfrog needs
    # omega dt < 2, i.e. r > m dt / 2 = m dr / 4. Fields ~ r^m are
    # negligible there, so pin the offending rows (regularity pins row 0
    # for any m != 0 regardless).
    i_min = max(1, math.floor(m / 4.0) + 1)
    return CylCoeffs(dt=dt, dr=dr, m=m, i_min=i_min, eps=e, inv_r=inv_r,
                     r_half=r_half, dz1=dz1, dz2=dz2, h1=h1, h2=h2,
                     hp1=hp1, hp2=hp2)


def init_cyl_state(nr: int, dtype=torch.complex128, device=None) -> CylState:
    device = device or default_device()
    z = lambda n: torch.zeros(n, dtype=dtype, device=device)
    return CylState(z(nr), z(nr), z(nr - 1), z(nr))


def step_cyl(state: CylState, c: CylCoeffs, src_amp: float = 0.0,
             src_ix: int | None = None) -> CylState:
    im = 1j * c.m

    # --- H update (from Ez at n) ---
    Hr = c.h1 * state.Hr - c.h2 * (im * c.inv_r) * state.Ez
    Hr[: c.i_min] = 0.0
    dEz = (state.Ez[1:] - state.Ez[:-1]) / c.dr
    Hphi = c.hp1 * state.Hphi + c.hp2 * dEz

    # --- Dz update (from H at n+1/2); axis row pinned (m != 0) ---
    curl = torch.zeros_like(state.Dz)
    curl[1:-1] = (c.r_half[1:] * Hphi[1:] - c.r_half[:-1] * Hphi[:-1]) \
        * c.inv_r[1:-1] / c.dr - im * c.inv_r[1:-1] * Hr[1:-1]
    Dz = c.dz1 * state.Dz + c.dz2 * curl
    if src_ix is not None:
        Dz[src_ix] = Dz[src_ix] - c.dt * src_amp
    Ez = Dz / c.eps
    Ez[: c.i_min] = 0.0                           # axis regularity, m != 0
    return CylState(Ez, Hr, Hphi, Dz)


def run_cyl(state: CylState, c: CylCoeffs, n_steps: int, *, src_ix: int,
            src_profile, probe_ix: int):
    import numpy as np
    bind_dt(src_profile, c.dt)
    series = np.empty(n_steps, dtype=complex)
    for t in range(n_steps):
        state = step_cyl(state, c, src_profile.amplitude((t + 0.5) * c.dt), src_ix)
        series[t] = state.Ez[probe_ix].item()
    return state, series
