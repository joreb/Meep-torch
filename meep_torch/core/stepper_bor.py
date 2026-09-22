# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2D (r, z) body-of-revolution FDTD for azimuthal number m (W7.5).

All six components, complex fields F(r, z) e^{i m phi}. Yee staggering:
  Er(i+1/2, j)   Ephi(i, j)         Ez(i, j+1/2)
  Hr(i, j+1/2)   Hphi(i+1/2, j+1/2) Hz(i+1/2, j)

Axis: m = 0 -> Ephi/Hr rows pinned, Ez axis via the ampere loop
(4/dr Hphi_{1/2}); m != 0 -> integer-r rows below r_min = m dt / 2
pinned (regularity + the i m / r leapfrog limit, W7 wave-1), half-r rows
likewise. Boundaries: matched sigma_d = sigma_b quadratic absorber on
outer r and both z ends. Courant dt = 0.5 dr / sqrt(2).
"""
import math
from typing import NamedTuple

import torch

from .pml import absorber_sigma
from .grid import default_device


class BORCoeffs(NamedTuple):
    dt: float
    dr: float
    m: int
    i_int: int               # pinned integer-r rows (Ephi, Ez, Hr)
    i_half: int              # pinned half-r rows (Er, Hphi, Hz)
    eps_r: torch.Tensor      # (nr-1, nz)
    eps_p: torch.Tensor      # (nr, nz)
    eps_z: torch.Tensor      # (nr, nz-1)
    inv_ri: torch.Tensor     # (nr, 1)   1/(i dr), 0 at i = 0
    inv_rh: torch.Tensor     # (nr-1, 1) 1/((i+1/2) dr)
    ri: torch.Tensor         # (nr, 1)   i dr
    rh: torch.Tensor         # (nr-1, 1) (i+1/2) dr
    damp: dict               # per-grid (d1, d2) absorber pairs


class BORState(NamedTuple):
    Er: torch.Tensor         # (nr-1, nz) complex
    Ephi: torch.Tensor       # (nr, nz)
    Ez: torch.Tensor         # (nr, nz-1)
    Hr: torch.Tensor         # (nr, nz-1)
    Hphi: torch.Tensor       # (nr-1, nz-1)
    Hz: torch.Tensor         # (nr-1, nz)


def _davg_r(v):
    return 0.5 * (v[:-1, :] + v[1:, :])


def _davg_z(v):
    return 0.5 * (v[:, :-1] + v[:, 1:])


def build_bor_coeffs(nr: int, nz: int, dr: float, m: int, eps=None,
                     n_abs_r: int = 0, n_abs_z: int = 0,
                     dtype=torch.float64, device=None) -> BORCoeffs:
    device = device or default_device()
    dt = 0.5 * dr / math.sqrt(2.0)
    # Coerce at the boundary: a caller-supplied eps may live anywhere, and
    # every coefficient built below lands on `device`. Trusting the caller
    # to have matched them makes each call site a place this can break.
    e = (eps.to(device=device, dtype=dtype) if isinstance(eps, torch.Tensor)
         else torch.ones(nr, nz, dtype=dtype, device=device))
    ri = torch.arange(nr, dtype=dtype, device=device)[:, None] * dr
    rh = (torch.arange(nr - 1, dtype=dtype, device=device)[:, None] + 0.5) * dr
    inv_ri = torch.zeros_like(ri)
    inv_ri[1:] = 1.0 / ri[1:]
    inv_rh = 1.0 / rh

    # absorber sigma: outer r only + both z ends, max-combined
    sig_r = absorber_sigma(nr, n_abs_r, dr, dtype=dtype, device=device)
    sig_r[: nr // 2] = 0.0
    sig_z = absorber_sigma(nz, n_abs_z, dr, dtype=dtype, device=device)
    sig = torch.maximum(sig_r[:, None].expand(nr, nz),
                        sig_z[None, :].expand(nr, nz))

    def dpair(s):
        return ((1 - s * dt / 2) / (1 + s * dt / 2), dt / (1 + s * dt / 2))

    damp = {
        "er": dpair(_davg_r(sig)), "ep": dpair(sig), "ez": dpair(_davg_z(sig)),
        "hr": dpair(_davg_z(sig)), "hp": dpair(_davg_z(_davg_r(sig))),
        "hz": dpair(_davg_r(sig)),
    }
    if m == 0:
        i_int, i_half = 1, 0          # Ephi/Hr regularity; Ez axis via loop
    else:
        r_min = m * dt / 2.0
        i_int = max(1, math.ceil(r_min / dr))
        i_half = max(0, math.ceil(r_min / dr - 0.5))
    return BORCoeffs(dt=dt, dr=dr, m=m, i_int=i_int, i_half=i_half,
                     eps_r=_davg_r(e), eps_p=e, eps_z=_davg_z(e),
                     inv_ri=inv_ri, inv_rh=inv_rh, ri=ri, rh=rh, damp=damp)


def init_bor_state(nr: int, nz: int, dtype=torch.complex128,
                   device=None) -> BORState:
    device = device or default_device()
    z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
    return BORState(z(nr - 1, nz), z(nr, nz), z(nr, nz - 1),
                    z(nr, nz - 1), z(nr - 1, nz - 1), z(nr - 1, nz))


def step_bor(state: BORState, c: BORCoeffs, src_amp: float = 0.0,
             src=None) -> BORState:
    """src: optional ("Er"|"Ez"|"Ephi", i, j) soft-current injection point."""
    dt, dr = c.dt, c.dr
    im = 1j * c.m

    # --- H update (from E at n) ---
    d1, d2 = c.damp["hr"]
    Hr = d1 * state.Hr + d2 * (-(im * c.inv_ri) * state.Ez
                               + (state.Ephi[:, 1:] - state.Ephi[:, :-1]) / dr)
    if abs(c.m) == 1:
        # |m| = 1: Hr finite at the axis. (im/r) Ez -> im * slope, with
        # Ez ~ alpha r and alpha = Ez[1] / dr (Ez[0] pinned to 0).
        Hr[0] = d1[0] * state.Hr[0] + d2[0] * (
            -im * state.Ez[1] / dr
            + (state.Ephi[0, 1:] - state.Ephi[0, :-1]) / dr)
    else:
        Hr[: c.i_int] = 0.0
    d1, d2 = c.damp["hp"]
    Hphi = d1 * state.Hphi + d2 * (
        -(state.Er[:, 1:] - state.Er[:, :-1]) / dr
        + (state.Ez[1:, :] - state.Ez[:-1, :]) / dr)
    Hphi[: c.i_half] = 0.0
    d1, d2 = c.damp["hz"]
    Hz = d1 * state.Hz + d2 * (
        -(c.ri[1:] * state.Ephi[1:, :] - c.ri[:-1] * state.Ephi[:-1, :])
        * c.inv_rh / dr + (im * c.inv_rh) * state.Er)
    Hz[: c.i_half] = 0.0

    # --- E update (from H at n+1/2) ---
    d1, d2 = c.damp["er"]
    curl_r = torch.zeros_like(state.Er)
    curl_r[:, 1:-1] = (im * c.inv_rh) * Hz[:, 1:-1] \
        - (Hphi[:, 1:] - Hphi[:, :-1]) / dr
    Er = (d1 * state.Er * c.eps_r + d2 * curl_r) / c.eps_r
    Er[: c.i_half] = 0.0

    d1, d2 = c.damp["ep"]
    curl_p = torch.zeros_like(state.Ephi)
    curl_p[1:-1, 1:-1] = (Hr[1:-1, 1:] - Hr[1:-1, :-1]) / dr \
        - (Hz[1:, 1:-1] - Hz[:-1, 1:-1]) / dr
    if abs(c.m) == 1:
        # |m| = 1: Ephi finite at the axis; Hz is odd in r (Hz ~ r), so
        # dHz/dr|_0 = 2 Hz(dr/2) / dr via the odd extension
        curl_p[0, 1:-1] = (Hr[0, 1:] - Hr[0, :-1]) / dr - 2.0 * Hz[0, 1:-1] / dr
    Ephi = (d1 * state.Ephi * c.eps_p + d2 * curl_p) / c.eps_p
    if abs(c.m) != 1:
        Ephi[: c.i_int] = 0.0

    d1, d2 = c.damp["ez"]
    curl_z = torch.zeros_like(state.Ez)
    curl_z[1:-1, :] = (c.rh[1:] * Hphi[1:, :] - c.rh[:-1] * Hphi[:-1, :]) \
        * c.inv_ri[1:-1] / dr - (im * c.inv_ri[1:-1]) * Hr[1:-1, :]
    if c.m == 0:
        curl_z[0, :] = (4.0 / dr) * Hphi[0, :]    # ampere loop at the axis
    Ez = (d1 * state.Ez * c.eps_z + d2 * curl_z) / c.eps_z
    if c.m != 0:
        Ez[: c.i_int] = 0.0

    if src is not None:
        terms = [src] if isinstance(src[0], str) else src
        for term in terms:
            comp, i, j = term[:3]
            fac = term[3] if len(term) > 3 else 1.0
            if comp == "Er":
                Er[i, j] = Er[i, j] - dt * src_amp * fac / c.eps_r[i, j]
            elif comp == "Ez":
                Ez[i, j] = Ez[i, j] - dt * src_amp * fac / c.eps_z[i, j]
            else:
                Ephi[i, j] = Ephi[i, j] - dt * src_amp * fac / c.eps_p[i, j]
    return BORState(Er, Ephi, Ez, Hr, Hphi, Hz)


def run_bor(state: BORState, c: BORCoeffs, n_steps: int, *, src,
            src_profile, probe):
    """probe: (component, i, j); returns the complex probe series."""
    import numpy as np
    series = np.empty(n_steps, dtype=complex)
    comp, pi, pj = probe
    for t in range(n_steps):
        state = step_bor(state, c, src_profile.amplitude((t + 0.5) * c.dt), src)
        series[t] = getattr(state, comp)[pi, pj].item()
    return state, series
