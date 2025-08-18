# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .grid import Grid


class TEState(NamedTuple):
    Hz: torch.Tensor       # (nx-1, ny-1) at n     (dual of TM Ez)
    Ex: torch.Tensor       # (nx-1, ny)   at n-1/2 (dual of TM -Hx)
    Ey: torch.Tensor       # (nx, ny-1)   at n-1/2 (dual of TM -Hy)
    Dx: torch.Tensor       # (nx-1, ny)
    Dy: torch.Tensor       # (nx, ny-1)
    Px_cur: torch.Tensor   # (np, nx-1, ny)
    Px_prev: torch.Tensor
    Py_cur: torch.Tensor   # (np, nx, ny-1)
    Py_prev: torch.Tensor
    psi_Eyx: torch.Tensor  # (nx-1, ny-1)
    psi_Exy: torch.Tensor  # (nx-1, ny-1)
    psi_Hzy: torch.Tensor  # (nx-1, ny-2)
    psi_Hzx: torch.Tensor  # (nx-2, ny-1)


def init_te_state(grid: Grid, n_poles: int = 0) -> TEState:
    nx, ny = grid.nx, grid.ny
    return TEState(
        Hz=grid.zeros(nx - 1, ny - 1),
        Ex=grid.zeros(nx - 1, ny),
        Ey=grid.zeros(nx, ny - 1),
        Dx=grid.zeros(nx - 1, ny),
        Dy=grid.zeros(nx, ny - 1),
        Px_cur=grid.zeros(n_poles, nx - 1, ny),
        Px_prev=grid.zeros(n_poles, nx - 1, ny),
        Py_cur=grid.zeros(n_poles, nx, ny - 1),
        Py_prev=grid.zeros(n_poles, nx, ny - 1),
        psi_Eyx=grid.zeros(nx - 1, ny - 1),
        psi_Exy=grid.zeros(nx - 1, ny - 1),
        psi_Hzy=grid.zeros(nx - 1, ny - 2),
        psi_Hzx=grid.zeros(nx - 2, ny - 1),
    )


def _avg4_to_ex(Ey: torch.Tensor) -> torch.Tensor:
    # Ey (nx, ny-1) -> value at Ex positions (nx-1, ny); zero-pad y boundaries
    Eyp = F.pad(Ey, (1, 1))                                    # (nx, ny+1)
    return 0.25 * (Eyp[:-1, :-1] + Eyp[:-1, 1:] + Eyp[1:, :-1] + Eyp[1:, 1:])


def _avg4_to_ey(Ex: torch.Tensor) -> torch.Tensor:
    # Ex (nx-1, ny) -> value at Ey positions (nx, ny-1); zero-pad x boundaries
    Exp = F.pad(Ex, (0, 0, 1, 1))                              # (nx+1, ny)
    return 0.25 * (Exp[:-1, :-1] + Exp[:-1, 1:] + Exp[1:, :-1] + Exp[1:, 1:])


def step_te(state: TEState, m, src_amp=0.0, src_mask_ey: torch.Tensor | None = None,
            src_h_amp: float = 0.0, src_hz_mask: torch.Tensor | None = None) -> TEState:
    # Update order mirrors step_tm under the duality Hz<->Ez, (Ex,Ey)<->(-Hx,-Hy):
    # the E fields (dual of TM's H) advance first from Hz(n), then Hz (dual of
    # TM's Ez) advances from E(n+1/2). This makes the discrete TE Hz update
    # graph isomorphic to the TM Ez graph in vacuum.
    dt, dx = m.dt, m.dx

    # --- D/E update (from Hz at time n) ---
    dHz_dy = (state.Hz[:, 1:] - state.Hz[:, :-1]) / dx          # (nx-1, ny-2)
    psi_Hzy = m.b_ey * state.psi_Hzy + m.a_ey * dHz_dy
    curl_x = F.pad(dHz_dy / m.k_ey + psi_Hzy, (1, 1))

    dHz_dx = (state.Hz[1:, :] - state.Hz[:-1, :]) / dx          # (nx-2, ny-1)
    psi_Hzx = m.b_ex * state.psi_Hzx + m.a_ex * dHz_dx
    curl_y = F.pad(dHz_dx / m.k_ex + psi_Hzx, (0, 0, 1, 1))
    if m.d1x is not None:
        # W5: dD/dt = curl H - sigma_d D
        Dx = m.d1x * state.Dx + m.d2x * curl_x
        Dy = m.d1y * state.Dy - m.d2y * curl_y
        if src_mask_ey is not None:
            Dy = Dy + m.d2y * src_amp * src_mask_ey
    else:
        Dx = state.Dx + dt * curl_x
        Dy = state.Dy - dt * curl_y
        if src_mask_ey is not None:
            Dy = Dy + dt * src_amp * src_mask_ey

    # --- ADE Lorentz poles: P(n+1/2) = c1 P(n-1/2) + c2 P(n-3/2) + c3 E(n-1/2) ---
    Px_new = m.c1x * state.Px_cur + m.c2x * state.Px_prev + m.c3x * state.Ex
    Py_new = m.c1y * state.Py_cur + m.c2y * state.Py_prev + m.c3y * state.Ey

    # --- E from D (instantaneous chi2/chi3 via fixed point) ---
    # D = eps*E + chi2*E^2 + chi3*|E|^2*E  ->  E = rhs/(eps + chi2*E + chi3*|E|^2)
    # chi2 here is the per-component self-term (scalar d_iii reduction); the
    # full d-tensor cross-terms are dropped -- TM is the W5.75 SHG target.
    rhs_x = Dx - Px_new.sum(0)
    rhs_y = Dy - Py_new.sum(0)
    if m.inv_xx_x is not None:
        # W5: in-plane tensor eps -- E = eps^{-1} D, cross terms Yee-averaged
        Ex = m.inv_xx_x * rhs_x + m.inv_xy_x * _avg4_to_ex(rhs_y)
        Ey = m.inv_yy_y * rhs_y + m.inv_xy_y * _avg4_to_ey(rhs_x)
    else:
        Ex = rhs_x / m.eps_x
        Ey = rhs_y / m.eps_y
    if m.has_kerr or m.has_chi2:
        for _ in range(3):
            E2x = Ex * Ex + _avg4_to_ex(Ey) ** 2
            E2y = Ey * Ey + _avg4_to_ey(Ex) ** 2
            Ex = rhs_x / (m.eps_x + m.chi2_x * Ex + m.chi3_x * E2x)
            Ey = rhs_y / (m.eps_y + m.chi2_y * Ey + m.chi3_y * E2y)

    # --- Hz update (from E at n+1/2) ---
    dEy_dx = (Ey[1:, :] - Ey[:-1, :]) / dx                      # (nx-1, ny-1)
    dEx_dy = (Ex[:, 1:] - Ex[:, :-1]) / dx                      # (nx-1, ny-1)
    psi_Eyx = m.b_hx * state.psi_Eyx + m.a_hx * dEy_dx
    psi_Exy = m.b_hy * state.psi_Exy + m.a_hy * dEx_dy
    curl_z = dEy_dx / m.k_hx + psi_Eyx - dEx_dy / m.k_hy - psi_Exy
    if m.hz1 is not None:
        # W5: dB/dt = -curl E - sigma_b B, B = mu H
        Hz = m.hz1 * state.Hz - m.hz2 * curl_z
    else:
        Hz = state.Hz - dt * curl_z

    # --- magnetic-current injection (TFSF): Hz at n+1 seen by next step's E/D ---
    if src_hz_mask is not None:
        Hz = Hz + dt * src_h_amp * src_hz_mask

    return TEState(
        Hz, Ex, Ey, Dx, Dy,
        Px_new, state.Px_cur, Py_new, state.Py_cur,
        psi_Eyx, psi_Exy, psi_Hzy, psi_Hzx,
    )
