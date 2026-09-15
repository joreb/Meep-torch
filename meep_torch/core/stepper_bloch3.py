# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D Bloch stepper: periodic x/y with Bloch phases, absorber z (MPB track).

Complex fields (no quadrature pairs needed in 3D — torch complex128),
full (nx, ny, nz) shapes, roll-based periodic derivatives with the
midpoint-averaged i k couplings (stepper_bloch2 pattern), one-sided z
derivatives with PEC walls behind a matched sigma absorber. Slab band
structures (mpb_hole_slab class): source at the slab midplane naturally
selects the z-even modes.

Triply-periodic mode (n_abs_z=0 + kz given): z is also periodic, using
the same roll-based _dp/_dm with i k_z midpoint coupling as x/y. No
absorber, no PEC -- the bulk 3D photonic crystal (e.g. mpb_diamond).
The absorber-z slab mode (n_abs_z>0) is untouched.

CFL: dt = 0.28 dx (3D bound with margin) and |k| dt < 2.
"""
import math
from typing import NamedTuple

import torch

from .pml import absorber_sigma
from .symbolic import compiled_uniform_difference

# Differences derived in core.symbolic rather than written here. The Bloch
# steppers take their neighbour by periodic `torch.roll` rather than by
# slicing, so there is no target component to infer the direction from -- these
# are the direction-given form. The compiled stencil is pure arithmetic, so it
# works unchanged on the complex fields a Bloch run carries.
_FWD = compiled_uniform_difference("forward")
_BWD = compiled_uniform_difference("backward")


class Bloch3State(NamedTuple):
    Ex: torch.Tensor; Ey: torch.Tensor; Ez: torch.Tensor    # (nx, ny, nz)
    Hx: torch.Tensor; Hy: torch.Tensor; Hz: torch.Tensor


class Bloch3Coeffs(NamedTuple):
    dt: float
    dx: float
    kx: float
    ky: float
    kz: float
    periodic_z: bool        # True: roll-based z (bulk 3D); False: absorber z (slab)
    eps: torch.Tensor        # (nx, ny, nz) real
    de1: torch.Tensor        # (1, 1, nz) absorber damping pairs (E)
    de2: torch.Tensor
    dh1: torch.Tensor        # (1, 1, nz) (H)
    dh2: torch.Tensor


def build_bloch3_coeffs(eps: torch.Tensor, dx: float, kx: float, ky: float,
                        n_abs_z: int = 0, kz: float = 0.0) -> Bloch3Coeffs:
    dt = 0.28 * dx
    periodic_z = n_abs_z == 0
    if max(abs(kx), abs(ky), abs(kz) if periodic_z else 0.0) >= 1.0 / dt:
        raise ValueError("Bloch k too large for dt")
    nz = eps.shape[2]
    sig = absorber_sigma(nz, n_abs_z, dx, dtype=eps.dtype, device=eps.device)
    sig = sig[None, None, :]
    de1 = (1 - sig * dt / 2) / (1 + sig * dt / 2)
    de2 = dt / (1 + sig * dt / 2)
    return Bloch3Coeffs(dt=dt, dx=dx, kx=kx, ky=ky, kz=kz,
                        periodic_z=periodic_z, eps=eps,
                        de1=de1, de2=de2, dh1=de1.clone(), dh2=de2.clone())


def init_bloch3_state(nx, ny, nz, device="cpu") -> Bloch3State:
    z = lambda: torch.zeros(nx, ny, nz, dtype=torch.complex128, device=device)
    return Bloch3State(z(), z(), z(), z(), z(), z())


def _dp(F, axis, dx, k):
    """(d/du + i k) forward-periodic, midpoint-averaged coupling."""
    return _FWD(F, torch.roll(F, -1, dims=axis), dx) \
        + 1j * k * 0.5 * (F + torch.roll(F, -1, dims=axis))


def _dm(F, axis, dx, k):
    """(d/du + i k) backward-periodic, midpoint-averaged coupling."""
    return _BWD(F, torch.roll(F, 1, dims=axis), dx) \
        + 1j * k * 0.5 * (F + torch.roll(F, 1, dims=axis))


def _dz_f(F, dx):
    out = torch.zeros_like(F)
    out[:, :, :-1] = _FWD(F[:, :, :-1], F[:, :, 1:], dx)
    return out


def _dz_b(F, dx):
    out = torch.zeros_like(F)
    out[:, :, 1:] = _BWD(F[:, :, 1:], F[:, :, :-1], dx)
    return out


def step_bloch3(st: Bloch3State, c: Bloch3Coeffs, src_amp=0.0,
                src=None) -> Bloch3State:
    dt, dx, kx, ky, kz = c.dt, c.dx, c.kx, c.ky, c.kz
    # z derivative: roll-based periodic (bulk 3D) or one-sided (absorber slab).
    if c.periodic_z:
        dz_f = lambda F: _dp(F, 2, dx, kz)
        dz_b = lambda F: _dm(F, 2, dx, kz)
    else:
        dz_f = lambda F: _dz_f(F, dx)
        dz_b = lambda F: _dz_b(F, dx)

    Hx = c.dh1 * st.Hx - c.dh2 * (_dp(st.Ez, 1, dx, ky) - dz_f(st.Ey))
    Hy = c.dh1 * st.Hy - c.dh2 * (dz_f(st.Ex) - _dp(st.Ez, 0, dx, kx))
    Hz = c.dh1 * st.Hz - c.dh2 * (_dp(st.Ey, 0, dx, kx) - _dp(st.Ex, 1, dx, ky))

    Ex = (c.de1 * st.Ex * c.eps
          + c.de2 * (_dm(Hz, 1, dx, ky) - dz_b(Hy))) / c.eps
    Ey = (c.de1 * st.Ey * c.eps
          + c.de2 * (dz_b(Hx) - _dm(Hz, 0, dx, kx))) / c.eps
    Ez = (c.de1 * st.Ez * c.eps
          + c.de2 * (_dm(Hy, 0, dx, kx) - _dm(Hx, 1, dx, ky))) / c.eps
    if src is not None:
        comp, i, j, k = src
        f = {"Ex": Ex, "Ey": Ey, "Ez": Ez}[comp]
        f[i, j, k] = f[i, j, k] - dt * src_amp / c.eps[i, j, k]
    return Bloch3State(Ex, Ey, Ez, Hx, Hy, Hz)
