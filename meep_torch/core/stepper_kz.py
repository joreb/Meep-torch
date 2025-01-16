# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2.5D k_z stepper, 1D-in-x reduction (W2 kz_2d deferral).

Fields F(x) e^{i k_z z}, d/dy = 0. The {Ex, Ez, Hy} set closes (p-pol,
incidence plane (x, z), interface normal x):

  dDx/dt = -i k_z Hy          Ex at x_{i+1/2}? -- collocated with Hy pairs:
  dBy/dt = -i k_z Ex + dEz/dx     Hy at x_{i+1/2}
  dDz/dt = + dHy/dx               Ez at x_i

Complex fields; outer matched sigma absorber both ends (reuses W5).
Courant dt = 0.5 dx.
"""
import math
from typing import NamedTuple

import torch

from .pml import absorber_sigma


class KzCoeffs(NamedTuple):
    dt: float
    dx: float
    kz: float
    eps_x: torch.Tensor     # (nx-1,) at x_{i+1/2} (Ex positions)
    eps_z: torch.Tensor     # (nx,)   at x_i (Ez positions)
    de1: torch.Tensor       # (nx,)   Ez damping
    de2: torch.Tensor
    dx1: torch.Tensor       # (nx-1,) Ex damping
    dx2: torch.Tensor
    h1: torch.Tensor        # (nx-1,) Hy damping
    h2: torch.Tensor


class KzState(NamedTuple):
    Ex: torch.Tensor        # (nx-1,) complex
    Ez: torch.Tensor        # (nx,)
    Hy: torch.Tensor        # (nx-1,)


def build_kz_coeffs(nx: int, dx: float, kz: float, eps=None,
                    n_abs: int = 0, dtype=torch.float64,
                    device="cpu") -> KzCoeffs:
    dt = 0.5 * dx
    e = (eps if isinstance(eps, torch.Tensor)
         else torch.ones(nx, dtype=dtype, device=device))
    e_x = 0.5 * (e[:-1] + e[1:])
    sig = absorber_sigma(nx, n_abs, dx, dtype=dtype, device=device)
    sig_h = 0.5 * (sig[:-1] + sig[1:])

    def dpair(s):
        return ((1 - s * dt / 2) / (1 + s * dt / 2), dt / (1 + s * dt / 2))

    de1, de2 = dpair(sig)
    dx1, dx2 = dpair(sig_h)
    h1, h2 = dpair(sig_h)
    return KzCoeffs(dt=dt, dx=dx, kz=kz, eps_x=e_x, eps_z=e,
                    de1=de1, de2=de2, dx1=dx1, dx2=dx2, h1=h1, h2=h2)


def init_kz_state(nx: int, dtype=torch.complex128, device="cpu") -> KzState:
    z = lambda n: torch.zeros(n, dtype=dtype, device=device)
    return KzState(z(nx - 1), z(nx), z(nx - 1))


def step_kz(state: KzState, c: KzCoeffs, src_amp: float = 0.0,
            src_ix: int | None = None) -> KzState:
    dt, dx = c.dt, c.dx
    ikz = 1j * c.kz

    # --- Hy update (from E at n) ---
    dEz = (state.Ez[1:] - state.Ez[:-1]) / dx
    Hy = c.h1 * state.Hy + c.h2 * (-ikz * state.Ex + dEz)

    # --- E update (from Hy at n+1/2) ---
    Ex = (c.dx1 * state.Ex * c.eps_x + c.dx2 * (-ikz * Hy)) / c.eps_x
    curl = torch.zeros_like(state.Ez)
    curl[1:-1] = (Hy[1:] - Hy[:-1]) / dx
    Ez = (c.de1 * state.Ez * c.eps_z + c.de2 * curl) / c.eps_z
    if src_ix is not None:
        Ez[src_ix] = Ez[src_ix] + dt * src_amp / c.eps_z[src_ix]
    return KzState(Ex, Ez, Hy)
