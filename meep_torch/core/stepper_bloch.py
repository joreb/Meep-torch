# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Bloch-periodic TM and TE steppers (real-valued engine, two-quadrature
envelope trick).

A Bloch-periodic field E(y + Ly) = exp(i k_y Ly) E(y) is written as
E(r) = e(r) exp(i k_y y) with e periodic (e(y+Ly)=e(y)). The FDTD evolves the
periodic envelope e; the Bloch phase enters the bulk as a modified y-derivative

    dE/dy -> de/dy + i k_y e

so the y-boundary is a SIMPLE periodic wrap (no phase bookkeeping) and the
engine stays real-valued by running two real quadratures e = e_c + i e_s:

    D_c = de_c/dy - k_y * e_s
    D_s = de_s/dy + k_y * e_c        (the i k_y e coupling)

x uses CPML (from build_tm_coeffs with y_pmc=True); y is periodic (wrap).
The TM half is nondispersive + Kerr, with dispersive Lorentz poles added per
quadrature when the scene has poles (passed via the same TMCoeffs c1/c2/c3 +
P_cur/P_prev). The TE half carries neither: it divides D by the two
scalar eps fields the caller supplies.
"""
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from .coeffs import TMCoeffs
from .symbolic import (PSI_NEW, STRETCHED,
                       compiled_uniform_difference)

# Differences derived in core.symbolic rather than written here. A Bloch run
# carries two real quadratures (c/s) of a complex envelope and takes its
# periodic neighbour by `torch.roll`, so there is no target component to infer
# the direction from -- these are the direction-given form.
_FWD = compiled_uniform_difference("forward")
_BWD = compiled_uniform_difference("backward")
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
    return _FWD(F, torch.roll(F, -1, dims=1), dx)


def _dy_periodic_bwd(F: torch.Tensor, dx: float) -> torch.Tensor:
    """(F[:, j] - F[:, j-1]) / dx with wrap -- the result sits at F's own j.

    The TE half needs this and the TM half does not, because on the Yee
    lattice Hz sits at j+1/2 and Ex at j, so the derivative feeding Ex reads
    DOWN from Hz. Before the lattices were unified Hz sat at j and the two
    directions were the other way round."""
    return _BWD(F, torch.roll(F, 1, dims=1), dx)


def step_bloch_tm(state: BlochTMState, m: TMCoeffs, ky: float,
                  src_c=0.0, src_s=0.0, src_mask: torch.Tensor | None = None,
                  src_mask_s: torch.Tensor | None = None) -> BlochTMState:
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
    # i k_y coupling evaluated AT Hx's stagger position (j+1/2): midpoint
    # average of Ez -- same O(k_y dx) bias fix as the TE stepper below
    Ezm_c = 0.5 * (state.Ez_c + torch.roll(state.Ez_c, -1, dims=1))
    Ezm_s = 0.5 * (state.Ez_s + torch.roll(state.Ez_s, -1, dims=1))
    Dc_y = dec_dy - ky * Ezm_s
    Ds_y = des_dy + ky * Ezm_c
    Hx_c = state.Hx_c - dt * Dc_y
    Hx_s = state.Hx_s - dt * Ds_y

    # x-derivative for Hy (x-CPML)
    dEz_dx = _FWD(state.Ez_c[:-1, :], state.Ez_c[1:, :], dx)
    psi_Ezx_c = PSI_NEW(dEz_dx, state.psi_Ezx_c, m.b_hx, m.a_hx)
    Hy_c = state.Hy_c + dt * (STRETCHED(dEz_dx, psi_Ezx_c, m.k_hx))
    dEz_dx_s = _FWD(state.Ez_s[:-1, :], state.Ez_s[1:, :], dx)
    psi_Ezx_s = PSI_NEW(dEz_dx_s, state.psi_Ezx_s, m.b_hx, m.a_hx)
    Hy_s = state.Hy_s + dt * (STRETCHED(dEz_dx_s, psi_Ezx_s, m.k_hx))

    # --- D update (envelope y-curl of Hx + i k_y coupling, x-curl of Hy) ---
    # Yee staggering: the Hx update uses the FORWARD y-difference of Ez
    # (edge j between cells j,j+1), so the Dz y-curl dHx/dy at cell j must use the
    # BACKWARD difference (Hx[j]-Hx[j-1]) -- a forward diff here shifts the curl
    # by one cell, making the update operator non-symmetric (complex eigenvalues
    # -> growth at any dt at structured-eps interfaces).
    dhxc_dy = _BWD(Hx_c, torch.roll(Hx_c, 1, dims=1), dx)    # backward, cell node
    dhxs_dy = _BWD(Hx_s, torch.roll(Hx_s, 1, dims=1), dx)
    # i k_y coupling at Dz's position (j): midpoint average of Hx back from j+1/2
    Hxm_c = 0.5 * (Hx_c + torch.roll(Hx_c, 1, dims=1))
    Hxm_s = 0.5 * (Hx_s + torch.roll(Hx_s, 1, dims=1))
    dHy_dx_c = _FWD(Hy_c[:-1, :], Hy_c[1:, :], dx)
    dHy_dx_s = _FWD(Hy_s[:-1, :], Hy_s[1:, :], dx)
    psi_Hyx_c = PSI_NEW(dHy_dx_c, state.psi_Hyx_c, m.b_ex, m.a_ex)
    psi_Hyx_s = PSI_NEW(dHy_dx_s, state.psi_Hyx_s, m.b_ex, m.a_ex)
    # x-CPML on the x-curl (interior x); y is full-periodic -> take the y-curl
    # at interior x too, pad x only (y_pmc-style).
    curlY_c_int = (dhxc_dy - ky * Hxm_s)[1:-1, :]
    curlY_s_int = (dhxs_dy + ky * Hxm_c)[1:-1, :]
    curl_c = STRETCHED(dHy_dx_c, psi_Hyx_c, m.k_ex) - curlY_c_int
    curl_s = STRETCHED(dHy_dx_s, psi_Hyx_s, m.k_ex) - curlY_s_int
    Dz_c = state.Dz_c + dt * F.pad(curl_c, (0, 0, 1, 1))
    Dz_s = state.Dz_s + dt * F.pad(curl_s, (0, 0, 1, 1))
    if src_mask is not None:
        if src_mask_s is not None:
            # complex spatial mask (mask_c + i mask_s): full complex product
            # with the (src_c + i src_s) temporal envelope -- single-order
            # (DiffractedPlanewave) launches need the e^{i m G y} phase
            Dz_c = Dz_c - dt * (src_c * src_mask - src_s * src_mask_s)
            Dz_s = Dz_s - dt * (src_c * src_mask_s + src_s * src_mask)
        else:
            Dz_c = Dz_c - dt * src_c * src_mask
            Dz_s = Dz_s - dt * src_s * src_mask

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
    e = (state.Ez_c[ix, :] + 1j * state.Ez_s[ix, :]).cpu().numpy()
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


class BlochTEState(NamedTuple):
    """The Yee lattice, as `core.symbolic.field_shape` derives it: Hz at
    (i+1/2, j+1/2), Ex at (i+1/2, j), Ey at (i, j+1/2), on an x that is walled
    and a y that wraps. This used to be the DUAL of that -- Hz on the integer
    node -- inherited by transcribing step_bloch_tm position for position, so
    the same problem gave different-shaped output depending on which TE stepper
    ran it."""
    Hz_c: torch.Tensor;  Hz_s: torch.Tensor     # (nx-1, ny) envelope H_z
    Ex_c: torch.Tensor;  Ex_s: torch.Tensor     # (nx-1, ny)
    Ey_c: torch.Tensor;  Ey_s: torch.Tensor     # (nx, ny)
    Dx_c: torch.Tensor;  Dx_s: torch.Tensor     # (nx-1, ny)
    Dy_c: torch.Tensor;  Dy_s: torch.Tensor     # (nx, ny)
    psi_Hzx_c: torch.Tensor; psi_Hzx_s: torch.Tensor   # (nx-2, ny) Ey update
    psi_Eyx_c: torch.Tensor; psi_Eyx_s: torch.Tensor   # (nx-1, ny) Hz update


def init_bloch_te_state(grid: Grid) -> BlochTEState:
    nx, ny = grid.nx, grid.ny
    z = lambda *s: grid.zeros(*s)
    return BlochTEState(
        Hz_c=z(nx - 1, ny), Hz_s=z(nx - 1, ny),
        Ex_c=z(nx - 1, ny), Ex_s=z(nx - 1, ny),
        Ey_c=z(nx, ny), Ey_s=z(nx, ny),
        Dx_c=z(nx - 1, ny), Dx_s=z(nx - 1, ny),
        Dy_c=z(nx, ny), Dy_s=z(nx, ny),
        psi_Hzx_c=z(nx - 2, ny), psi_Hzx_s=z(nx - 2, ny),
        psi_Eyx_c=z(nx - 1, ny), psi_Eyx_s=z(nx - 1, ny),
    )


def step_bloch_te(state: BlochTEState, m: TMCoeffs, eps_x: torch.Tensor,
                  eps_y: torch.Tensor, ky: float, src_c=0.0, src_s=0.0,
                  src_mask: torch.Tensor | None = None,
                  src_mask_s: torch.Tensor | None = None) -> BlochTEState:
    """TE dual of step_bloch_tm (Hz envelope, two quadratures, x-CPML,
    periodic y). m supplies dt/dx and the x-CPML profiles (build_tm_coeffs
    with y_pmc=True); eps_x (nx, ny)/eps_y (nx-1, ny) at the E positions.
    Sources are magnetic currents into Hz (mask on the Hz grid).

    Yee ordering mirrors step_te (E first from Hz(n), then Hz from E). On the
    Yee lattice Hz is at (i+1/2, j+1/2), so the Ex update takes the BACKWARD
    periodic y-difference of Hz and the Hz y-curl takes the FORWARD difference
    of Ex. Both directions flipped when the lattice was unified; the pairing is
    what matters (W2 Wave-4 lesson) and it still holds.

    The x pad moved with them. dHz/dx now lands on Ey's INTERIOR -- tangential
    E is pinned at both PEC walls -- so the Dy update pads, and the Hz update,
    whose dEy/dx already lands on Hz's own grid, no longer does.
    """
    dt, dx = m.dt, m.dx
    if abs(ky) >= 1.0 / dt:
        raise ValueError(f"Bloch k_y={ky:.4f} violates |k_y| < 1/dt={1/dt:.2f}")

    # --- D/E update (from Hz at n) ---
    # Ex at (i+1/2, j) reads DOWN from Hz at j+1/2: backward, periodic.
    dhc = _dy_periodic_bwd(state.Hz_c, dx)
    dhs = _dy_periodic_bwd(state.Hz_s, dx)
    # i k_y coupling evaluated AT Ex's stagger position (j): midpoint average
    # of Hz -- without it the bands carry an O(k_y dx) bias
    Hzm_c = 0.5 * (state.Hz_c + torch.roll(state.Hz_c, 1, dims=1))
    Hzm_s = 0.5 * (state.Hz_s + torch.roll(state.Hz_s, 1, dims=1))
    Dx_c = state.Dx_c + dt * (dhc - ky * Hzm_s)
    Dx_s = state.Dx_s + dt * (dhs + ky * Hzm_c)
    Ex_c, Ex_s = Dx_c / eps_x, Dx_s / eps_x

    # dHz/dx lands on Ey's interior (nx-2): Hz is at i+1/2, Ey at i, and the
    # two PEC walls pin tangential E. The E-family CPML profile is the one that
    # matches that extent -- on the old dual lattice this term took b_hx, the
    # H profile, which was the tell that the lattice was accidental.
    dHz_dx_c = _BWD(state.Hz_c[1:, :], state.Hz_c[:-1, :], dx)
    dHz_dx_s = _BWD(state.Hz_s[1:, :], state.Hz_s[:-1, :], dx)
    psi_Hzx_c = PSI_NEW(dHz_dx_c, state.psi_Hzx_c, m.b_ex, m.a_ex)
    psi_Hzx_s = PSI_NEW(dHz_dx_s, state.psi_Hzx_s, m.b_ex, m.a_ex)
    pad_x = (0, 0, 1, 1)
    Dy_c = state.Dy_c - dt * F.pad(STRETCHED(dHz_dx_c, psi_Hzx_c, m.k_ex), pad_x)
    Dy_s = state.Dy_s - dt * F.pad(STRETCHED(dHz_dx_s, psi_Hzx_s, m.k_ex), pad_x)
    Ey_c, Ey_s = Dy_c / eps_y, Dy_s / eps_y

    # --- Hz update (from E at n+1/2) ---
    dexc_dy = _dy_periodic(Ex_c, dx)                   # forward, periodic
    dexs_dy = _dy_periodic(Ex_s, dx)
    Exm_c = 0.5 * (Ex_c + torch.roll(Ex_c, -1, dims=1))  # at Hz's j+1/2
    Exm_s = 0.5 * (Ex_s + torch.roll(Ex_s, -1, dims=1))
    curlY_c = dexc_dy - ky * Exm_s
    curlY_s = dexs_dy + ky * Exm_c
    dEy_dx_c = _FWD(Ey_c[:-1, :], Ey_c[1:, :], dx)      # (nx-1, ny), Hz's grid
    dEy_dx_s = _FWD(Ey_s[:-1, :], Ey_s[1:, :], dx)
    psi_Eyx_c = PSI_NEW(dEy_dx_c, state.psi_Eyx_c, m.b_hx, m.a_hx)
    psi_Eyx_s = PSI_NEW(dEy_dx_s, state.psi_Eyx_s, m.b_hx, m.a_hx)
    curl_c = STRETCHED(dEy_dx_c, psi_Eyx_c, m.k_hx) - curlY_c
    curl_s = STRETCHED(dEy_dx_s, psi_Eyx_s, m.k_hx) - curlY_s
    Hz_c = state.Hz_c - dt * curl_c
    Hz_s = state.Hz_s - dt * curl_s
    if src_mask is not None:
        if src_mask_s is not None:
            # complex spatial mask, full complex product (mirrors step_bloch_tm)
            Hz_c = Hz_c - dt * (src_c * src_mask - src_s * src_mask_s)
            Hz_s = Hz_s - dt * (src_c * src_mask_s + src_s * src_mask)
        else:
            Hz_c = Hz_c - dt * src_c * src_mask
            Hz_s = Hz_s - dt * src_s * src_mask

    return BlochTEState(Hz_c, Hz_s, Ex_c, Ex_s, Ey_c, Ey_s,
                        Dx_c, Dx_s, Dy_c, Dy_s,
                        psi_Hzx_c, psi_Hzx_s, psi_Eyx_c, psi_Eyx_s)


def bloch_te_xflux(state: BlochTEState, ix: int, dy: float) -> float:
    """Time-averaged +x flux for the TE envelope: <S_x> = +1/2 (Ey Hz)."""
    # Hz sits at i+1/2, Ey at i, so averaging Hz[ix-1] and Hz[ix] lands on
    # Ey[ix] exactly. On the old dual lattice the same average landed on
    # Ey[ix-1] instead.
    Hz_c = 0.5 * (state.Hz_c[ix - 1, :] + state.Hz_c[ix, :])
    Hz_s = 0.5 * (state.Hz_s[ix - 1, :] + state.Hz_s[ix, :])
    s = (state.Ey_c[ix, :] * Hz_c + state.Ey_s[ix, :] * Hz_s)
    return 0.5 * float(s.sum()) * dy
