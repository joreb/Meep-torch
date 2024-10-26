# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Bloch-periodic TM stepper (real-valued engine, two-quadrature envelope trick).

A Bloch-periodic field E(y + Ly) = exp(i k_y Ly) E(y) is written as
E(r) = e(r) exp(i k_y y) with e periodic (e(y+Ly)=e(y)). The FDTD evolves the
periodic envelope e; the Bloch phase enters the bulk as a modified y-derivative

    dE/dy -> de/dy + i k_y e

so the y-boundary is a SIMPLE periodic wrap (no phase bookkeeping) and the
engine stays real-valued by running two real quadratures e = e_c + i e_s:

    D_c = de_c/dy - k_y * e_s
    D_s = de_s/dy + k_y * e_c        (the i k_y e coupling)

x uses CPML (from build_tm_coeffs with y_pmc=True); y is periodic (wrap).
Nondispersive + Kerr; dispersive Lorentz poles are added per quadrature when
the scene has poles (passed via the same TMCoeffs c1/c2/c3 + P_cur/P_prev).
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .coeffs import TMCoeffs
from .grid import Grid


class BlochTMState(NamedTuple):
    Ez_c: torch.Tensor;  Ez_s: torch.Tensor     # (nx, ny) envelope E_z
    Hx_c: torch.Tensor;  Hx_s: torch.Tensor     # (nx, ny) periodic y-edges
    Hy_c: torch.Tensor;  Hy_s: torch.Tensor     # (nx-1, ny)
    Dz_c: torch.Tensor;  Dz_s: torch.Tensor     # (nx, ny)
    P_cur_c: torch.Tensor; P_cur_s: torch.Tensor
    P_prev_c: torch.Tensor; P_prev_s: torch.Tensor
    psi_Ezx_c: torch.Tensor; psi_Ezx_s: torch.Tensor   # x-CPML only
    psi_Hyx_c: torch.Tensor; psi_Hyx_s: torch.Tensor


def init_bloch_tm_state(grid: Grid, n_poles: int = 0) -> BlochTMState:
    nx, ny = grid.nx, grid.ny
    z = lambda *s: grid.zeros(*s)
    return BlochTMState(
        Ez_c=z(nx, ny), Ez_s=z(nx, ny), Hx_c=z(nx, ny), Hx_s=z(nx, ny),
        Hy_c=z(nx - 1, ny), Hy_s=z(nx - 1, ny), Dz_c=z(nx, ny), Dz_s=z(nx, ny),
        P_cur_c=z(n_poles, nx, ny), P_cur_s=z(n_poles, nx, ny),
        P_prev_c=z(n_poles, nx, ny), P_prev_s=z(n_poles, nx, ny),
        psi_Ezx_c=z(nx - 1, ny), psi_Ezx_s=z(nx - 1, ny),
        psi_Hyx_c=z(nx - 2, ny), psi_Hyx_s=z(nx - 2, ny),
    )


def _dy_periodic(F: torch.Tensor, dx: float) -> torch.Tensor:
    """Periodic y-derivative (no phase): (F[:, j+1] - F[:, j]) / dx with wrap."""
    return (torch.roll(F, -1, dims=1) - F) / dx


def step_bloch_tm(state: BlochTMState, m: TMCoeffs, ky: float,
                  src_c=0.0, src_s=0.0, src_mask: torch.Tensor | None = None) -> BlochTMState:
    """One TM step of the Bloch envelope (two quadratures), x-CPML, periodic y.

    Sources inject the envelope e at a line: src_c/src_s are the c/s temporal
    envelopes times the spatial mask (the caller builds the +x-traveling phase).
    """
    dt, dx = m.dt, m.dx

    # --- H update (envelope y-derivative + i k_y coupling) ---
    dec_dy = _dy_periodic(state.Ez_c, dx)
    des_dy = _dy_periodic(state.Ez_s, dx)
    Dc_y = dec_dy - ky * state.Ez_s
    Ds_y = des_dy + ky * state.Ez_c
    Hx_c = state.Hx_c - dt * Dc_y
    Hx_s = state.Hx_s - dt * Ds_y

    # x-derivative for Hy (x-CPML)
    dEz_dx = (state.Ez_c[1:, :] - state.Ez_c[:-1, :]) / dx
    psi_Ezx_c = m.b_hx * state.psi_Ezx_c + m.a_hx * dEz_dx
    Hy_c = state.Hy_c + dt * (dEz_dx / m.k_hx + psi_Ezx_c)
    dEz_dx_s = (state.Ez_s[1:, :] - state.Ez_s[:-1, :]) / dx
    psi_Ezx_s = m.b_hx * state.psi_Ezx_s + m.a_hx * dEz_dx_s
    Hy_s = state.Hy_s + dt * (dEz_dx_s / m.k_hx + psi_Ezx_s)

    # --- D update (envelope y-curl of Hx + i k_y coupling, x-curl of Hy) ---
    dhxc_dy = _dy_periodic(Hx_c, dx)
    dhxs_dy = _dy_periodic(Hx_s, dx)
    curlY_c = dhxc_dy - ky * Hx_s
    curlY_s = dhxs_dy + ky * Hx_c
    dHy_dx_c = (Hy_c[1:, :] - Hy_c[:-1, :]) / dx
    dHy_dx_s = (Hy_s[1:, :] - Hy_s[:-1, :]) / dx
    psi_Hyx_c = m.b_ex * state.psi_Hyx_c + m.a_ex * dHy_dx_c
    psi_Hyx_s = m.b_ex * state.psi_Hyx_s + m.a_ex * dHy_dx_s
    # x-CPML on the x-curl (interior x); y is full-periodic -> take the y-curl
    # at interior x too, pad x only (y_pmc-style).
    curlY_c_int = (dhxc_dy - ky * Hx_s)[1:-1, :]
    curlY_s_int = (dhxs_dy + ky * Hx_c)[1:-1, :]
    curl_c = dHy_dx_c / m.k_ex + psi_Hyx_c - curlY_c_int
    curl_s = dHy_dx_s / m.k_ex + psi_Hyx_s - curlY_s_int
    Dz_c = state.Dz_c + dt * F.pad(curl_c, (0, 0, 1, 1))
    Dz_s = state.Dz_s + dt * F.pad(curl_s, (0, 0, 1, 1))
    if src_mask is not None:
        Dz_c = Dz_c + dt * src_c * src_mask
        Dz_s = Dz_s + dt * src_s * src_mask

    # --- Lorentz ADE (per quadrature) ---
    if m.c1.numel():
        P_cur_c = m.c1 * state.P_cur_c + m.c2 * state.P_prev_c + m.c3 * state.Ez_c
        P_cur_s = m.c1 * state.P_cur_s + m.c2 * state.P_prev_s + m.c3 * state.Ez_s
    else:
        P_cur_c = state.P_cur_c
        P_cur_s = state.P_cur_s

    # --- E from D (Kerr fixed point, per quadrature) ---
    rhs_c = Dz_c - P_cur_c.sum(0)
    rhs_s = Dz_s - P_cur_s.sum(0)
    Ez_c = rhs_c / m.eps_inf
    Ez_s = rhs_s / m.eps_inf
    if m.has_kerr:
        for _ in range(3):
            Ez_c = rhs_c / (m.eps_inf + m.chi3 * Ez_c * Ez_c)
            Ez_s = rhs_s / (m.eps_inf + m.chi3 * Ez_s * Ez_s)

    return BlochTMState(Ez_c, Ez_s, Hx_c, Hx_s, Hy_c, Hy_s, Dz_c, Dz_s,
                        P_cur_c, P_cur_s, state.P_cur_c, state.P_cur_s,
                        psi_Ezx_c, psi_Ezx_s, psi_Hyx_c, psi_Hyx_s)