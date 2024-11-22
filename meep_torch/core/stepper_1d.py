# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""1D full-vector stepper (Ey, Ez, Hy, Hz along x) with gyrotropic media.

The TM/TE split cannot host gyrotropy: a bias along the propagation axis
couples the two transverse E components through the polarization ADE
(GyrotropicLorentzianSusceptibility). Yee 1D: Ey/Ez at integer x, Hy/Hz at
half x; scalar eps/mu; x-CPML.

Gyrotropic ADE (central differences, coupled 2x2 solved in closed form):
  d2P/dt2 + g dP/dt - wb x_hat x dP/dt + w0^2 P = s w0^2 E
  P+ = Ainv [ (2 - w0^2 dt^2) P + s w0^2 dt^2 E - C P- ],
  A = (1+g dt/2) I - (wb dt/2) K,  C = (1-g dt/2) I + (wb dt/2) K,
  K = [[0, -1], [1, 0]] acting on (Py, Pz). The cross-term sign fixes the
  rotation sense; the magnitude gate is sign-independent.
"""
import math
from typing import NamedTuple

import torch

from .grid import Grid
from .pml import cpml_profiles


class Coeffs1D(NamedTuple):
    dt: float
    dx: float
    eps: torch.Tensor        # (nx,)
    b_h: torch.Tensor        # (nx-1,) CPML at half x (H updates)
    a_h: torch.Tensor
    k_h: torch.Tensor
    b_e: torch.Tensor        # (nx-2,) CPML at interior integer x (D updates)
    a_e: torch.Tensor
    k_e: torch.Tensor
    gyro: tuple | None       # (w0^2 dt^2, s w0^2 dt^2, Ainv 2x2, C 2x2)


class State1D(NamedTuple):
    Ey: torch.Tensor         # (nx,)
    Ez: torch.Tensor
    Hy: torch.Tensor         # (nx-1,)
    Hz: torch.Tensor
    Dy: torch.Tensor
    Dz: torch.Tensor
    Py_cur: torch.Tensor     # (nx,) gyrotropic polarization (zeros when off)
    Pz_cur: torch.Tensor
    Py_prev: torch.Tensor
    Pz_prev: torch.Tensor
    psi_Ezx: torch.Tensor    # (nx-1,)  Hy update
    psi_Eyx: torch.Tensor    # (nx-1,)  Hz update
    psi_Hyx: torch.Tensor    # (nx-2,)  Dz update
    psi_Hzx: torch.Tensor    # (nx-2,)  Dy update


def build_1d_coeffs(grid: Grid, eps=1.0, npml: int = 0, gyro=None) -> Coeffs1D:
    dt, dx = grid.dt, grid.dx
    nx = grid.nx
    e = torch.full((nx,), float(eps), dtype=grid.dtype, device=grid.device) \
        if not isinstance(eps, torch.Tensor) else eps
    b_h, a_h, k_h = cpml_profiles(nx - 1, npml, 0.5, dt, dx,
                                  dtype=grid.dtype, device=grid.device)
    b_e_f, a_e_f, k_e_f = cpml_profiles(nx, npml, 0.0, dt, dx,
                                        dtype=grid.dtype, device=grid.device)
    g = None
    if gyro is not None:
        f0, gamma, sigma, b0 = gyro
        w0 = 2 * math.pi * f0
        gp = 2 * math.pi * gamma * dt / 2.0
        # cross-term sign pinned by the meep-oracle profile gate (rotation
        # sense); the tutorial-analytic magnitude gate is sign-independent
        h = -2 * math.pi * b0 * dt / 2.0
        det = (1 + gp) ** 2 + h * h
        # A = [[1+gp, h], [-h, 1+gp]]  (K = [[0,-1],[1,0]], A = (1+gp)I - hK)
        ainv = ((1 + gp) / det, -h / det, h / det, (1 + gp) / det)  # row-major
        c = (1 - gp, -h, h, 1 - gp)                                # C = (1-gp)I + hK
        g = (w0 * w0 * dt * dt, sigma * w0 * w0 * dt * dt, ainv, c)
    return Coeffs1D(dt=dt, dx=dx, eps=e,
                    b_h=b_h, a_h=a_h, k_h=k_h,
                    b_e=b_e_f[1:-1], a_e=a_e_f[1:-1], k_e=k_e_f[1:-1],
                    gyro=g)


def init_1d_state(grid: Grid) -> State1D:
    nx = grid.nx
    z = lambda n: grid.zeros(n)
    return State1D(z(nx), z(nx), z(nx - 1), z(nx - 1), z(nx), z(nx),
                   z(nx), z(nx), z(nx), z(nx),
                   z(nx - 1), z(nx - 1), z(nx - 2), z(nx - 2))


def step_1d(state: State1D, m: Coeffs1D, src_amp: float = 0.0,
            src_ix: int | None = None) -> State1D:
    dt, dx = m.dt, m.dx

    # --- H update (from E at n) ---
    dEz = (state.Ez[1:] - state.Ez[:-1]) / dx
    psi_Ezx = m.b_h * state.psi_Ezx + m.a_h * dEz
    Hy = state.Hy + dt * (dEz / m.k_h + psi_Ezx)

    dEy = (state.Ey[1:] - state.Ey[:-1]) / dx
    psi_Eyx = m.b_h * state.psi_Eyx + m.a_h * dEy
    Hz = state.Hz - dt * (dEy / m.k_h + psi_Eyx)

    # --- D update (from H at n+1/2), PEC x-walls ---
    dHy = (Hy[1:] - Hy[:-1]) / dx
    psi_Hyx = m.b_e * state.psi_Hyx + m.a_e * dHy
    Dz = state.Dz.clone()
    Dz[1:-1] = state.Dz[1:-1] + dt * (dHy / m.k_e + psi_Hyx)

    dHz = (Hz[1:] - Hz[:-1]) / dx
    psi_Hzx = m.b_e * state.psi_Hzx + m.a_e * dHz
    Dy = state.Dy.clone()
    Dy[1:-1] = state.Dy[1:-1] - dt * (dHz / m.k_e + psi_Hzx)
    if src_ix is not None:
        Dy[src_ix] = Dy[src_ix] + dt * src_amp

    # --- gyrotropic ADE + E from D ---
    if m.gyro is not None:
        w0dt2, sdt2, (a11, a12, a21, a22), (c11, c12, c21, c22) = m.gyro
        ry = (2 - w0dt2) * state.Py_cur + sdt2 * state.Ey \
            - (c11 * state.Py_prev + c12 * state.Pz_prev)
        rz = (2 - w0dt2) * state.Pz_cur + sdt2 * state.Ez \
            - (c21 * state.Py_prev + c22 * state.Pz_prev)
        Py = a11 * ry + a12 * rz
        Pz = a21 * ry + a22 * rz
        Ey = (Dy - Py) / m.eps
        Ez = (Dz - Pz) / m.eps
        return State1D(Ey, Ez, Hy, Hz, Dy, Dz, Py, Pz,
                       state.Py_cur, state.Pz_cur,
                       psi_Ezx, psi_Eyx, psi_Hyx, psi_Hzx)
    Ey = Dy / m.eps
    Ez = Dz / m.eps
    return State1D(Ey, Ez, Hy, Hz, Dy, Dz,
                   state.Py_cur, state.Pz_cur, state.Py_prev, state.Pz_prev,
                   psi_Ezx, psi_Eyx, psi_Hyx, psi_Hzx)


def run_1d(state: State1D, m: Coeffs1D, n_steps: int, *, src_ix: int,
           src_profile, probe_ix):
    """Step n_steps, recording (Ey, Ez) time series at each probe index."""
    import numpy as np
    series = [(np.empty(n_steps), np.empty(n_steps)) for _ in probe_ix]
    for t in range(n_steps):
        state = step_1d(state, m, src_profile.amplitude((t + 0.5) * m.dt), src_ix)
        for s, ix in zip(series, probe_ix):
            s[0][t] = state.Ey[ix].item()
            s[1][t] = state.Ez[ix].item()
    return state, series
