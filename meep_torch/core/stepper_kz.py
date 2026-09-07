# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2.5D k_z stepper, 1D-in-x reduction (W2 kz_2d deferral).

Fields F(x) e^{i k_z z}, d/dy = 0. The {Ex, Ez, Hy} set closes (p-pol,
incidence plane (x, z), interface normal x):

  dDx/dt = -i k_z Hy          Ex at x_{i+1/2}? -- collocated with Hy pairs:
  dBy/dt = -i k_z Ex + dEz/dx     Hy at x_{i+1/2}
  dDz/dt = + dHy/dx               Ez at x_i

Complex fields. Two boundary modes:
  * ``n_abs > 0`` -- wave-1 matched-sigma absorber both ends (reuses W5).
  * ``npml > 0`` -- CFS-CPML on the two x-derivatives (dEz/dx in the Hy
    update at H positions; dHy/dx in the Ez update at E positions). The
    ik_z terms are the z-derivative and are NOT PML'd (z is the
    propagation axis, k_z fixed) -- only the x-derivatives are stretched.
    A matched-sigma absorber reflects near grazing because the x
    wavelength lambda_x = 2 pi / (w cos th) outgrows the graded layer,
    so the onset reads as a hard step; the CPML is impedance-matched by
    construction (reflectionless in the continuum limit at every angle),
    and the low x-group velocity near grazing lengthens the residence
    time so the sigma decay integrates over more cycles, not fewer.
Courant dt = 0.5 dx.
"""
import math
from typing import NamedTuple

import torch

from .pml import absorber_sigma, cpml_profiles


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
    # CPML on x-derivatives (None in absorber mode). bH/aH/kH live at the
    # H positions (x_{i+1/2}, length nx-1) for dEz/dx in the Hy update;
    # bE/aE/kE live at the interior E positions (x_1..x_{nx-2}, length
    # nx-2) for dHy/dx in the Ez update.
    cpml: tuple | None = None


class KzState(NamedTuple):
    Ex: torch.Tensor        # (nx-1,) complex
    Ez: torch.Tensor        # (nx,)   complex
    Hy: torch.Tensor        # (nx-1,) complex
    psi_dEz: torch.Tensor | None = None   # (nx-1,) CPML memory, dEz/dx
    psi_dHy: torch.Tensor | None = None   # (nx-2,) CPML memory, dHy/dx


def build_kz_coeffs(nx: int, dx: float, kz: float, eps=None,
                    n_abs: int = 0, npml: int = 0, dtype=torch.float64,
                    device="cpu", pml_m: int = 3, pml_R0: float = 1e-6,
                    pml_alpha: float = 0.05) -> KzCoeffs:
    """Build the 2.5D stepper coefficients. ``n_abs`` selects the wave-1
    matched-sigma absorber; ``npml`` selects the kz-aware CFS-CPML (the two
    are mutually exclusive -- pass exactly one). ``pml_*`` tune the CFS
    profile (polynomial grade ``m``, target reflection ``R0``, CFS
    ``alpha_max``); defaults match ``cpml_profiles``."""
    if n_abs > 0 and npml > 0:
        raise ValueError("kz stepper: pass n_abs OR npml, not both")
    dt = 0.5 * dx
    e = (eps if isinstance(eps, torch.Tensor)
         else torch.ones(nx, dtype=dtype, device=device))
    e_x = 0.5 * (e[:-1] + e[1:])

    if npml > 0:
        # Identity E/H damping (the CPML absorbs); CFS profiles on the two
        # x-derivatives. p-pol set: Hy uses dEz/dx, Ez uses dHy/dx.
        bH, aH, kH = cpml_profiles(nx - 1, npml, 0.5, dt, dx, m=pml_m,
                                   R0=pml_R0, alpha_max=pml_alpha,
                                   dtype=dtype, device=device)
        bE, aE, kE = cpml_profiles(nx, npml, 0.0, dt, dx, m=pml_m,
                                   R0=pml_R0, alpha_max=pml_alpha,
                                   dtype=dtype, device=device)
        bE, aE, kE = bE[1:-1], aE[1:-1], kE[1:-1]
        one = torch.ones((), dtype=dtype, device=device)
        de1 = torch.ones(nx, dtype=dtype, device=device)
        de2 = torch.full((nx,), dt, dtype=dtype, device=device)
        dx1 = torch.ones(nx - 1, dtype=dtype, device=device)
        dx2 = torch.full((nx - 1,), dt, dtype=dtype, device=device)
        h1 = torch.ones(nx - 1, dtype=dtype, device=device)
        h2 = torch.full((nx - 1,), dt, dtype=dtype, device=device)
        cpml = (bH, aH, kH, bE, aE, kE)
        return KzCoeffs(dt=dt, dx=dx, kz=kz, eps_x=e_x, eps_z=e,
                       de1=de1, de2=de2, dx1=dx1, dx2=dx2, h1=h1, h2=h2,
                       cpml=cpml)

    sig = absorber_sigma(nx, n_abs, dx, dtype=dtype, device=device)
    sig_h = 0.5 * (sig[:-1] + sig[1:])

    def dpair(s):
        return ((1 - s * dt / 2) / (1 + s * dt / 2), dt / (1 + s * dt / 2))

    de1, de2 = dpair(sig)
    dx1, dx2 = dpair(sig_h)
    h1, h2 = dpair(sig_h)
    return KzCoeffs(dt=dt, dx=dx, kz=kz, eps_x=e_x, eps_z=e,
                    de1=de1, de2=de2, dx1=dx1, dx2=dx2, h1=h1, h2=h2,
                    cpml=None)


def init_kz_state(nx: int, npml: int = 0, dtype=torch.complex128,
                  device="cpu") -> KzState:
    z = lambda n: torch.zeros(n, dtype=dtype, device=device)
    psi_dEz = z(nx - 1) if npml > 0 else None
    psi_dHy = z(nx - 2) if npml > 0 else None
    return KzState(z(nx - 1), z(nx), z(nx - 1), psi_dEz, psi_dHy)


def step_kz(state: KzState, c: KzCoeffs, src_amp: float = 0.0,
            src_ix: int | None = None) -> KzState:
    dt, dx = c.dt, c.dx
    ikz = 1j * c.kz
    cpml = c.cpml

    if cpml is None:
        # --- wave-1 matched-sigma absorber path ---
        dEz = (state.Ez[1:] - state.Ez[:-1]) / dx
        Hy = c.h1 * state.Hy + c.h2 * (-ikz * state.Ex + dEz)
        Ex = (c.dx1 * state.Ex * c.eps_x + c.dx2 * (-ikz * Hy)) / c.eps_x
        curl = torch.zeros_like(state.Ez)
        curl[1:-1] = (Hy[1:] - Hy[:-1]) / dx
        Ez = (c.de1 * state.Ez * c.eps_z + c.de2 * curl) / c.eps_z
        if src_ix is not None:
            Ez[src_ix] = Ez[src_ix] - dt * src_amp / c.eps_z[src_ix]
        return KzState(Ex, Ez, Hy)

    # --- CFS-CPML path: stretch only the x-derivatives (ikz terms untouched) ---
    bH, aH, kH, bE, aE, kE = cpml

    # Hy <- dEz/dx  (forward diff at H positions x_{i+1/2}, length nx-1)
    dEz = (state.Ez[1:] - state.Ez[:-1]) / dx
    psi_dEz = bH * state.psi_dEz + aH * dEz
    ce_x = dEz / kH + psi_dEz
    Hy = c.h1 * state.Hy + c.h2 * (-ikz * state.Ex + ce_x)

    # Ex has no x-derivative (pure ikz coupling); no CPML.
    Ex = (c.dx1 * state.Ex * c.eps_x + c.dx2 * (-ikz * Hy)) / c.eps_x

    # Ez <- dHy/dx  (backward diff at interior E positions x_1..x_{nx-2})
    dHy = (Hy[1:] - Hy[:-1]) / dx
    psi_dHy = bE * state.psi_dHy + aE * dHy
    ch_x = dHy / kE + psi_dHy
    curl = torch.zeros_like(state.Ez)
    curl[1:-1] = ch_x
    Ez = (c.de1 * state.Ez * c.eps_z + c.de2 * curl) / c.eps_z
    if src_ix is not None:
        Ez[src_ix] = Ez[src_ix] - dt * src_amp / c.eps_z[src_ix]
    return KzState(Ex, Ez, Hy, psi_dEz, psi_dHy)