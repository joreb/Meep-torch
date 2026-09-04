# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Doubly-periodic Bloch steppers, TM and TE (MPB track wave 2).

Fully periodic unit cell with Bloch phases in BOTH axes: envelope
two-quadrature trick of stepper_bloch applied to x and y, no PML.
Bloch couplings i k_x / i k_y are midpoint-averaged onto the partner's
stagger position. Rectangular lattices only (oblique basis deferred).

TM (Ez, Hx, Hy) and TE (Hz, Ex, Ey) are duals and share every helper
here; they differ in which half-step takes the forward difference and in
where epsilon sits. TM carries one E component at the cell centre, so it
uses eps as given; TE carries two on staggered edges, so it needs eps
averaged onto each (see step_bloch2_te).

For *band structures* prefer meep_torch.mpb.eigensolver: an eigensolve
returns the N lowest bands complete and ordered, whereas driving these
steppers and harvesting with harminv infers the band index from the sort
position of whatever was detected, which misnumbers silently when a mode
is missed. These remain the right tool for time-domain work on a
Bloch-periodic cell, and as an independent check on the eigensolver.

CFL: courant bound plus |k| dt < 2 (the envelope precession); asserted.
"""
import math
from typing import NamedTuple

import torch


class Bloch2State(NamedTuple):
    Ez_c: torch.Tensor;  Ez_s: torch.Tensor     # (nx, ny)
    Hx_c: torch.Tensor;  Hx_s: torch.Tensor     # (nx, ny)
    Hy_c: torch.Tensor;  Hy_s: torch.Tensor     # (nx, ny)
    Dz_c: torch.Tensor;  Dz_s: torch.Tensor


def init_bloch2_state(grid) -> Bloch2State:
    z = lambda: grid.zeros(grid.nx, grid.ny)
    return Bloch2State(z(), z(), z(), z(), z(), z(), z(), z())


def _fwd(F, axis, dx):
    return (torch.roll(F, -1, dims=axis) - F) / dx


def _bwd(F, axis, dx):
    return (F - torch.roll(F, 1, dims=axis)) / dx


def _mid_fwd(F, axis):
    return 0.5 * (F + torch.roll(F, -1, dims=axis))


def _mid_bwd(F, axis):
    return 0.5 * (F + torch.roll(F, 1, dims=axis))


def step_bloch2_tm(state: Bloch2State, eps: torch.Tensor, dt: float,
                   dx: float, kx: float, ky: float,
                   src_c=0.0, src_s=0.0,
                   src_mask: torch.Tensor | None = None) -> Bloch2State:
    if max(abs(kx), abs(ky)) >= 1.0 / dt:
        raise ValueError(f"Bloch k={max(abs(kx), abs(ky)):.3f} >= 1/dt")

    # --- H update (from Ez at n): B = -curl E per component ---
    dEzc_y, dEzs_y = _fwd(state.Ez_c, 1, dx), _fwd(state.Ez_s, 1, dx)
    Ezm_c_y, Ezm_s_y = _mid_fwd(state.Ez_c, 1), _mid_fwd(state.Ez_s, 1)
    Hx_c = state.Hx_c - dt * (dEzc_y - ky * Ezm_s_y)
    Hx_s = state.Hx_s - dt * (dEzs_y + ky * Ezm_c_y)

    dEzc_x, dEzs_x = _fwd(state.Ez_c, 0, dx), _fwd(state.Ez_s, 0, dx)
    Ezm_c_x, Ezm_s_x = _mid_fwd(state.Ez_c, 0), _mid_fwd(state.Ez_s, 0)
    Hy_c = state.Hy_c + dt * (dEzc_x - kx * Ezm_s_x)
    Hy_s = state.Hy_s + dt * (dEzs_x + kx * Ezm_c_x)

    # --- Dz update (from H at n+1/2): (d_x + i k_x) Hy - (d_y + i k_y) Hx ---
    cx_c = _bwd(Hy_c, 0, dx) - kx * _mid_bwd(Hy_s, 0)
    cx_s = _bwd(Hy_s, 0, dx) + kx * _mid_bwd(Hy_c, 0)
    cy_c = _bwd(Hx_c, 1, dx) - ky * _mid_bwd(Hx_s, 1)
    cy_s = _bwd(Hx_s, 1, dx) + ky * _mid_bwd(Hx_c, 1)
    Dz_c = state.Dz_c + dt * (cx_c - cy_c)
    Dz_s = state.Dz_s + dt * (cx_s - cy_s)
    if src_mask is not None:
        Dz_c = Dz_c + dt * src_c * src_mask
        Dz_s = Dz_s + dt * src_s * src_mask
    Ez_c = Dz_c / eps
    Ez_s = Dz_s / eps
    return Bloch2State(Ez_c, Ez_s, Hx_c, Hx_s, Hy_c, Hy_s, Dz_c, Dz_s)


class Bloch2StateTE(NamedTuple):
    Hz_c: torch.Tensor;  Hz_s: torch.Tensor     # (nx, ny)  dual of TM Ez
    Ex_c: torch.Tensor;  Ex_s: torch.Tensor     # (nx, ny)
    Ey_c: torch.Tensor;  Ey_s: torch.Tensor     # (nx, ny)
    Dx_c: torch.Tensor;  Dx_s: torch.Tensor
    Dy_c: torch.Tensor;  Dy_s: torch.Tensor


def init_bloch2_state_te(grid) -> Bloch2StateTE:
    z = lambda: grid.zeros(grid.nx, grid.ny)
    return Bloch2StateTE(z(), z(), z(), z(), z(), z(), z(), z(), z(), z())


def te_eps_edges(eps: torch.Tensor):
    """(eps_x, eps_y) for the TE E-components, periodic two-point averages.

    Ex sits half a cell along x from the cell centre where eps is sampled
    and Ey half a cell along y, so each takes the average across the cell
    face it lives on. TM needs no such step: its single Ez shares the
    centre with eps.
    """
    return _mid_bwd(eps, 0), _mid_bwd(eps, 1)


def step_bloch2_te(state: Bloch2StateTE, eps_x: torch.Tensor,
                   eps_y: torch.Tensor, dt: float, dx: float,
                   kx: float, ky: float, src_c=0.0, src_s=0.0,
                   src_mask: torch.Tensor | None = None) -> Bloch2StateTE:
    """One TE step. ``eps_x``/``eps_y`` come from :func:`te_eps_edges`.

    Mirror of step_bloch2_tm: the D half takes the backward difference and
    the Hz half the forward one, the opposite way round from TM. The
    source drives Dy, matching the TM stepper driving Dz.
    """
    if max(abs(kx), abs(ky)) >= 1.0 / dt:
        raise ValueError(f"Bloch k={max(abs(kx), abs(ky)):.3f} >= 1/dt")

    # --- D update (from Hz at n): dDx/dt = (d_y + i k_y) Hz,
    #                              dDy/dt = -(d_x + i k_x) Hz ---
    dHzc_y, dHzs_y = _bwd(state.Hz_c, 1, dx), _bwd(state.Hz_s, 1, dx)
    Hzm_c_y, Hzm_s_y = _mid_bwd(state.Hz_c, 1), _mid_bwd(state.Hz_s, 1)
    Dx_c = state.Dx_c + dt * (dHzc_y - ky * Hzm_s_y)
    Dx_s = state.Dx_s + dt * (dHzs_y + ky * Hzm_c_y)

    dHzc_x, dHzs_x = _bwd(state.Hz_c, 0, dx), _bwd(state.Hz_s, 0, dx)
    Hzm_c_x, Hzm_s_x = _mid_bwd(state.Hz_c, 0), _mid_bwd(state.Hz_s, 0)
    Dy_c = state.Dy_c - dt * (dHzc_x - kx * Hzm_s_x)
    Dy_s = state.Dy_s - dt * (dHzs_x + kx * Hzm_c_x)

    if src_mask is not None:
        Dy_c = Dy_c + dt * src_c * src_mask
        Dy_s = Dy_s + dt * src_s * src_mask

    Ex_c, Ex_s = Dx_c / eps_x, Dx_s / eps_x
    Ey_c, Ey_s = Dy_c / eps_y, Dy_s / eps_y

    # --- Hz update (from E at n+1/2): -[(d_x + i k_x) Ey - (d_y + i k_y) Ex] ---
    ax_c = _fwd(Ey_c, 0, dx) - kx * _mid_fwd(Ey_s, 0)
    ax_s = _fwd(Ey_s, 0, dx) + kx * _mid_fwd(Ey_c, 0)
    ay_c = _fwd(Ex_c, 1, dx) - ky * _mid_fwd(Ex_s, 1)
    ay_s = _fwd(Ex_s, 1, dx) + ky * _mid_fwd(Ex_c, 1)
    Hz_c = state.Hz_c - dt * (ax_c - ay_c)
    Hz_s = state.Hz_s - dt * (ax_s - ay_s)
    return Bloch2StateTE(Hz_c, Hz_s, Ex_c, Ex_s, Ey_c, Ey_s,
                         Dx_c, Dx_s, Dy_c, Dy_s)
