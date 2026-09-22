# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Full-vector 2D stepper: all six components, 3x3 symmetric eps (W5's
polarization_grating blocker -- z-coupled permittivity breaks the TM/TE
split). d/dz = 0; periodic y (roll-based), x-CPML from cpml_profiles.

Full-shape (nx, ny) fields; Yee staggering carried implicitly by
forward/backward difference pairing (the Bloch-stepper pattern):
  Bx <- -dy_f Ez        By <- +dx Ez (CPML)     Bz <- -(dx Ey - dy_f Ex)
  Dx <- +dy_b Hz        Dy <- -dx Hz (CPML)     Dz <- +(dx Hy - dy_b Hx)
E from D via the precomputed inverse of the 3x3 eps tensor (symmetric). By
default the components are taken at cell centers and applied pointwise -- no
cross-position averaging, adequate for smoothly varying tensors like the LC
director. ``build_fv2d_coeffs(eps_at_nodes=True)`` instead Kottke-blends eps
onto each E component's own Yee node first; see that function.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ..geometry.smoothing import blend_tensor_tau
from .pml import cpml_profiles
from .grid import default_device


class FV2DCoeffs(NamedTuple):
    dt: float
    dx: float
    inv: tuple               # 3 six-tuples: the inverse at Ex's, Ey's, Ez's node
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


def _inv6(eps6):
    """Inverse of the symmetric 3x3, as its six independent components."""
    exx, exy, exz, eyy, eyz, ezz = eps6
    det = (exx * (eyy * ezz - eyz * eyz) - exy * (exy * ezz - eyz * exz)
           + exz * (exy * eyz - eyy * exz))
    return ((eyy * ezz - eyz * eyz) / det, (exz * eyz - exy * ezz) / det,
            (exy * eyz - exz * eyy) / det, (exx * ezz - exz * exz) / det,
            (exy * exz - exx * eyz) / det, (exx * eyy - exy * exy) / det)


def _kottke_x(eps6):
    """eps to the i+1/2 node, Kottke-blended. x is walled: the last row
    replicates, so the blend there is the identity."""
    shifted = [torch.cat([c[1:, :], c[-1:, :]], dim=0) for c in eps6]
    return blend_tensor_tau(eps6, shifted, 0)


def _kottke_y(eps6):
    """eps to the j+1/2 node. y wraps, so the blend is exact everywhere."""
    shifted = [torch.roll(c, -1, dims=1) for c in eps6]
    return blend_tensor_tau(eps6, shifted, 1)


def build_fv2d_coeffs(eps6, dx: float, npml: int = 0,
                      dtype=torch.float64, device=None,
                      eps_at_nodes: bool = False) -> FV2DCoeffs:
    """eps6 = (exx, exy, exz, eyy, eyz, ezz), each (nx, ny).

    ``eps_at_nodes`` treats eps6 as sampled at the Ez node (i, j) and moves it
    onto each E component's OWN position before inverting -- Ex at (i+1/2, j),
    Ey at (i, j+1/2), Ez where it already is -- using the KOTTKE blend, not an
    arithmetic average. That distinction is the whole thing: a plain arithmetic
    mean across an interface is first-order and measured WORSE than leaving the
    eps where it was. See `symbolic_test/kottke.py` for the derivation.
    """
    device = device or default_device()
    exx, exy, exz, eyy, eyz, ezz = eps6
    dt = 0.5 * dx / (2 ** 0.5)
    if eps_at_nodes:
        inv = (_inv6(_kottke_x(eps6)),     # Ex row
               _inv6(_kottke_y(eps6)),     # Ey row
               _inv6(eps6))                # Ez row
    else:
        one = _inv6(eps6)
        inv = (one, one, one)
    ixx, ixy, ixz, iyy, iyz, izz = inv[0]
    nx = exx.shape[0]
    b_hx, a_hx, k_hx = cpml_profiles(nx - 1, npml, 0.5, dt, dx,
                                     dtype=dtype, device=device)
    b_e, a_e, k_e = cpml_profiles(nx, npml, 0.0, dt, dx,
                                  dtype=dtype, device=device)
    return FV2DCoeffs(dt=dt, dx=dx, inv=inv,
                      b_hx=b_hx[:, None], a_hx=a_hx[:, None], k_hx=k_hx[:, None],
                      b_ex=b_e[1:-1, None], a_ex=a_e[1:-1, None],
                      k_ex=k_e[1:-1, None])


def init_fv2d_state(nx, ny, dtype=torch.float64, device=None) -> FV2DState:
    device = device or default_device()
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

    # one row per E component, each from the eps at that component's own node
    (xxx, xxy, xxz, _, _, _) = c.inv[0]
    (_, yxy, _, yyy, yyz, _) = c.inv[1]
    (_, _, zxz, _, zyz, zzz) = c.inv[2]
    Ex = xxx * Dx + xxy * Dy + xxz * Dz
    Ey = yxy * Dx + yyy * Dy + yyz * Dz
    Ez = zxz * Dx + zyz * Dy + zzz * Dz
    return FV2DState(Ex, Ey, Ez, Hx, Hy, Hz, Dx, Dy, Dz,
                     psi_hy, psi_hz, psi_ey_new, psi_ez_new)
