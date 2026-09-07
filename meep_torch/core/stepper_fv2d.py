# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Full-vector 2D stepper: all six components, 3x3 symmetric eps (W5's
polarization_grating blocker -- z-coupled permittivity breaks the TM/TE
split). d/dz = 0; periodic y (roll-based), x-CPML from cpml_profiles.

Full-shape (nx, ny) fields; Yee staggering carried implicitly by
forward/backward difference pairing (the Bloch-stepper pattern):
  Bx <- -dy_f Ez        By <- +dx Ez (CPML)     Bz <- -(dx Ey - dy_f Ex)
  Dx <- +dy_b Hz        Dy <- -dx Hz (CPML)     Dz <- +(dx Hy - dy_b Hx)
E from D via the precomputed inverse of the 3x3 eps tensor (symmetric;
components given at cell centers, applied pointwise -- no cross-position
averaging, adequate for smoothly varying tensors like the LC director).
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .pml import cpml_profiles


class FV2DCoeffs(NamedTuple):
    dt: float
    dx: float
    inv: tuple               # 6 tensors (nx, ny): ixx, ixy, ixz, iyy, iyz, izz
    b_hx: torch.Tensor       # (nx-1, 1) CPML half-x (H updates from dEz/dx etc.)
    a_hx: torch.Tensor
    k_hx: torch.Tensor
    b_ex: torch.Tensor       # (nx-2, 1) CPML interior-x (E updates)
    a_ex: torch.Tensor
    k_ex: torch.Tensor


class FV2DState(NamedTuple):
    Ex: torch.Tensor; Ey: torch.Tensor; Ez: torch.Tensor    # (nx, ny)
    Hx: torch.Tensor; Hy: torch.Tensor; Hz: torch.Tensor
    Dx: torch.Tensor; Dy: torch.Tensor; Dz: torch.Tensor
    psi_hy: torch.Tensor; psi_hz: torch.Tensor              # (nx-1, ny)
    psi_ey: torch.Tensor; psi_ez: torch.Tensor              # (nx-2, ny)


def build_fv2d_coeffs(eps6, dx: float, npml: int = 0,
                      dtype=torch.float64, device="cpu") -> FV2DCoeffs:
    """eps6 = (exx, exy, exz, eyy, eyz, ezz), each (nx, ny)."""
    exx, exy, exz, eyy, eyz, ezz = eps6
    dt = 0.5 * dx / (2 ** 0.5)
    det = (exx * (eyy * ezz - eyz * eyz) - exy * (exy * ezz - eyz * exz)
           + exz * (exy * eyz - eyy * exz))
    ixx = (eyy * ezz - eyz * eyz) / det
    ixy = (exz * eyz - exy * ezz) / det
    ixz = (exy * eyz - exz * eyy) / det
    iyy = (exx * ezz - exz * exz) / det
    iyz = (exy * exz - exx * eyz) / det
    izz = (exx * eyy - exy * exy) / det
    nx = exx.shape[0]
    b_hx, a_hx, k_hx = cpml_profiles(nx - 1, npml, 0.5, dt, dx,
                                     dtype=dtype, device=device)
    b_e, a_e, k_e = cpml_profiles(nx, npml, 0.0, dt, dx,
                                  dtype=dtype, device=device)
    return FV2DCoeffs(dt=dt, dx=dx, inv=(ixx, ixy, ixz, iyy, iyz, izz),
                      b_hx=b_hx[:, None], a_hx=a_hx[:, None], k_hx=k_hx[:, None],
                      b_ex=b_e[1:-1, None], a_ex=a_e[1:-1, None],
                      k_ex=k_e[1:-1, None])


def init_fv2d_state(nx, ny, dtype=torch.float64, device="cpu") -> FV2DState:
    z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
    return FV2DState(z(nx, ny), z(nx, ny), z(nx, ny),
                     z(nx, ny), z(nx, ny), z(nx, ny),
                     z(nx, ny), z(nx, ny), z(nx, ny),
                     z(nx - 1, ny), z(nx - 1, ny), z(nx - 2, ny), z(nx - 2, ny))


def _dyf(v, dx):
    return (torch.roll(v, -1, dims=1) - v) / dx


def _dyb(v, dx):
    return (v - torch.roll(v, 1, dims=1)) / dx


def step_fv2d(st: FV2DState, c: FV2DCoeffs, src_amp=0.0, src_mask=None,
              src_comp: str = "Ey", src2=None) -> FV2DState:
    """src2: optional second injection (comp, mask, amp) for e.g. circular
    polarization (Ey + i Ez quadrature pairs)."""
    dt, dx = c.dt, c.dx

    # --- H update (from E at n) ---
    Hx = st.Hx - dt * _dyf(st.Ez, dx)
    dEz_dx = (st.Ez[1:, :] - st.Ez[:-1, :]) / dx
    psi_hy = c.b_hx * st.psi_hy + c.a_hx * dEz_dx
    Hy = st.Hy + dt * F.pad(dEz_dx / c.k_hx + psi_hy, (0, 0, 0, 1))
    dEy_dx = (st.Ey[1:, :] - st.Ey[:-1, :]) / dx
    psi_hz = c.b_hx * st.psi_hz + c.a_hx * dEy_dx
    Hz = st.Hz - dt * (F.pad(dEy_dx / c.k_hx + psi_hz, (0, 0, 0, 1))
                       - _dyf(st.Ex, dx))

    # --- D update (from H at n+1/2) ---
    Dx = st.Dx + dt * _dyb(Hz, dx)
    dHz_dx = (Hz[1:, :] - Hz[:-1, :]) / dx
    psi_ez_new = c.b_ex * st.psi_ez + c.a_ex * dHz_dx[:-1, :]
    Dy = st.Dy - dt * F.pad(dHz_dx[:-1, :] / c.k_ex + psi_ez_new, (0, 0, 1, 1))
    dHy_dx = (Hy[1:, :] - Hy[:-1, :]) / dx
    psi_ey_new = c.b_ex * st.psi_ey + c.a_ex * dHy_dx[:-1, :]
    Dz = st.Dz + dt * (F.pad(dHy_dx[:-1, :] / c.k_ex + psi_ey_new, (0, 0, 1, 1))
                       - _dyb(Hx, dx))
    for comp, mask, amp in ([(src_comp, src_mask, src_amp)] if src_mask is not None else []) \
            + ([src2] if src2 is not None else []):
        if comp == "Ey":
            Dy = Dy - dt * amp * mask
        elif comp == "Ez":
            Dz = Dz - dt * amp * mask
        else:
            Dx = Dx - dt * amp * mask

    ixx, ixy, ixz, iyy, iyz, izz = c.inv
    Ex = ixx * Dx + ixy * Dy + ixz * Dz
    Ey = ixy * Dx + iyy * Dy + iyz * Dz
    Ez = ixz * Dx + iyz * Dy + izz * Dz
    return FV2DState(Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz,
                     psi_hy, psi_hz, psi_ey_new, psi_ez_new)
