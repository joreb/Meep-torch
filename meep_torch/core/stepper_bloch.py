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

    CFL: the standard 2D Yee Courant bound (courant <= 1/sqrt(2)) must hold for
    the grid derivatives, AND the +i k_y bulk coupling precesses the quadratures
    at c*|k_y|, which must be temporally resolved: c*dt*|k_y| < 2 (Nyquist; c=1
    in natural units). We assert the safety margin |k_y| < 1/dt. For a planewave
    at angle theta, k_y = 2*pi*f*sin(theta); high angles / high f may need a
    smaller dt (lower courant) or a larger cell -- raise() flags it.
    """
    dt, dx = m.dt, m.dx
    if abs(ky) >= 1.0 / dt:
        raise ValueError(
            f"Bloch k_y={ky:.4f} violates temporal-Nyquist bound |k_y| < 1/dt={1/dt:.2f}; "
            f"reduce dt (lower courant) or the angle/frequency")

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

    # --- E from D (instantaneous Kerr via fixed point over the pair jointly) ---
    # Isotropic Kerr uses the COMPLEX envelope intensity |E|^2 = E_c^2 + E_s^2;
    # both quadratures share the same nonlinear denominator.
    rhs_c = Dz_c - P_cur_c.sum(0)
    rhs_s = Dz_s - P_cur_s.sum(0)
    Ez_c = rhs_c / m.eps_inf
    Ez_s = rhs_s / m.eps_inf
    if m.has_kerr:
        for _ in range(3):
            denom = m.eps_inf + m.chi3 * (Ez_c * Ez_c + Ez_s * Ez_s)
            Ez_c = rhs_c / denom
            Ez_s = rhs_s / denom

    return BlochTMState(Ez_c, Ez_s, Hx_c, Hx_s, Hy_c, Hy_s, Dz_c, Dz_s,
                        P_cur_c, P_cur_s, state.P_cur_c, state.P_cur_s,
                        psi_Ezx_c, psi_Ezx_s, psi_Hyx_c, psi_Hyx_s)

def bloch_xflux(state: BlochTMState, ix: int, dy: float) -> float:
    """Time-averaged +x Poynting flux through the column x=ix for the complex
    envelope field E = (E_c + i E_s) exp(i k_y y). meep-torch TM stores Hy with a
    sign flip vs the eigenmode convention (S_x = -Ez*Hy), so <S_x> =
    -0.5 (E_c H_c + E_s H_s). Hy lives on x-edges; average the two around ix.
    """
    Hy_c = 0.5 * (state.Hy_c[ix - 1, :] + state.Hy_c[ix, :])
    Hy_s = 0.5 * (state.Hy_s[ix - 1, :] + state.Hy_s[ix, :])
    s = (state.Ez_c[ix, :] * Hy_c + state.Ez_s[ix, :] * Hy_s)
    return -0.5 * float(s.sum()) * dy


def bloch_diffraction_orders(state: BlochTMState, ix: int, ky: float, G: float,
                              k0: float, dy: float, n_modes: int = 5):
    """Diffraction-order powers through the column x=ix for a Bloch-periodic
    field whose envelope e is periodic over Ly = 2*pi/G. The physical field
    E = e(y) exp(i k_y y); the diffraction orders are the Fourier harmonics of e
    at m*G, each a plane wave with transverse k_y_m = k_y + m*G and longitudinal
    k_x_m = sqrt(k0^2 - k_y_m^2) (propagating where real). Order power is the
    time-averaged +x Poynting flux of that harmonic, normalized to the incident
    (m=0 vacuum) flux: |a_m|^2 * (k_x_m / k_x0). Returns (orders, powers) where
    powers[m+n_modes] is the (real) order power (0 for evanescent orders).
    """
    e = (state.Ez_c[ix, :] + 1j * state.Ez_s[ix, :]).numpy()
    ny = e.shape[0]
    # Fourier harmonics of the periodic envelope: a_m = (1/ny) sum_j e_j exp(-i m G y_j)
    ys = (np.arange(ny) + 0.5) * dy
    orders = list(range(-n_modes, n_modes + 1))
    kx0 = math.sqrt(max(0.0, k0 * k0 - ky * ky))
    powers = []
    for m in orders:
        kym = ky + m * G
        a = float(np.abs(np.sum(e * np.exp(-1j * m * G * ys))) / ny)
        kxm = k0 * k0 - kym * kym
        if kxm <= 0:
            powers.append(0.0)                  # evanescent order
        else:
            powers.append(a * a * (math.sqrt(kxm) / kx0) if kx0 > 0 else 0.0)
    return orders, np.array(powers)
