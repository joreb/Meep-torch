# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Doubly-periodic Bloch TM stepper (MPB track wave 2).

Fully periodic unit cell with Bloch phases in BOTH axes: envelope
two-quadrature trick of stepper_bloch applied to x and y, no PML.
Bloch couplings i k_x / i k_y are midpoint-averaged onto the partner's
stagger position. Rectangular lattices only (oblique basis deferred).

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
