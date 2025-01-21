# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Full 3D Yee stepper, wave 1 (W9): real fields, scalar eps, matched
sigma_d = sigma_b absorber shells on all faces (CPML lands in a later
wave). Courant dt = 0.5 dx / sqrt(3).

Yee positions (i, j, k integers):
  Ex(i+1/2, j, k)  Ey(i, j+1/2, k)  Ez(i, j, k+1/2)
  Hx(i, j+1/2, k+1/2)  Hy(i+1/2, j, k+1/2)  Hz(i+1/2, j+1/2, k)
"""
import math
from typing import NamedTuple

import torch

from .pml import absorber_sigma


class Coeffs3D(NamedTuple):
    dt: float
    dx: float
    eps: torch.Tensor        # (nx, ny, nz) cell-centered
    damp: dict               # per-grid (d1, d2) absorber pairs


class State3D(NamedTuple):
    Ex: torch.Tensor         # (nx-1, ny, nz)
    Ey: torch.Tensor         # (nx, ny-1, nz)
    Ez: torch.Tensor         # (nx, ny, nz-1)
    Hx: torch.Tensor         # (nx, ny-1, nz-1)
    Hy: torch.Tensor         # (nx-1, ny, nz-1)
    Hz: torch.Tensor         # (nx-1, ny-1, nz)


def _avg(v, axis):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(None, -1)
    sl1[axis] = slice(1, None)
    return 0.5 * (v[tuple(sl0)] + v[tuple(sl1)])


def build_3d_coeffs(nx: int, ny: int, nz: int, dx: float, eps=None,
                    n_abs: int = 0, dtype=torch.float64,
                    device="cpu") -> Coeffs3D:
    dt = 0.5 * dx / math.sqrt(3.0)
    e = (eps if isinstance(eps, torch.Tensor)
         else torch.ones(nx, ny, nz, dtype=dtype, device=device))
    sx = absorber_sigma(nx, n_abs, dx, dtype=dtype, device=device)
    sy = absorber_sigma(ny, n_abs, dx, dtype=dtype, device=device)
    sz = absorber_sigma(nz, n_abs, dx, dtype=dtype, device=device)
    sig = torch.maximum(torch.maximum(sx[:, None, None].expand(nx, ny, nz),
                                      sy[None, :, None].expand(nx, ny, nz)),
                        sz[None, None, :].expand(nx, ny, nz))

    def dpair(s):
        return ((1 - s * dt / 2) / (1 + s * dt / 2), dt / (1 + s * dt / 2))

    damp = {
        "ex": dpair(_avg(sig, 0)), "ey": dpair(_avg(sig, 1)), "ez": dpair(_avg(sig, 2)),
        "hx": dpair(_avg(_avg(sig, 1), 2)), "hy": dpair(_avg(_avg(sig, 0), 2)),
        "hz": dpair(_avg(_avg(sig, 0), 1)),
    }
    return Coeffs3D(dt=dt, dx=dx, eps=e, damp=damp)


def init_3d_state(nx: int, ny: int, nz: int, dtype=torch.float64,
                  device="cpu") -> State3D:
    z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
    return State3D(z(nx - 1, ny, nz), z(nx, ny - 1, nz), z(nx, ny, nz - 1),
                   z(nx, ny - 1, nz - 1), z(nx - 1, ny, nz - 1),
                   z(nx - 1, ny - 1, nz))


def _d(v, axis):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(None, -1)
    sl1[axis] = slice(1, None)
    return v[tuple(sl1)] - v[tuple(sl0)]


def step_3d(state: State3D, c: Coeffs3D, src_amp: float = 0.0,
            src=None) -> State3D:
    """src: optional ("Ez"|"Ex"|"Ey", i, j, k) soft point current."""
    dx = c.dx

    # --- H update (from E at n): dB/dt = -curl E ---
    d1, d2 = c.damp["hx"]
    Hx = d1 * state.Hx - d2 * (_d(state.Ez, 1) - _d(state.Ey, 2)) / dx
    d1, d2 = c.damp["hy"]
    Hy = d1 * state.Hy - d2 * (_d(state.Ex, 2) - _d(state.Ez, 0)) / dx
    d1, d2 = c.damp["hz"]
    Hz = d1 * state.Hz - d2 * (_d(state.Ey, 0) - _d(state.Ex, 1)) / dx

    # --- E update (from H at n+1/2): dD/dt = curl H, PEC outer faces ---
    def pad_interior(t, axes):
        pad = [0, 0, 0, 0, 0, 0]
        for a in axes:
            pad[2 * (2 - a)] = 1
            pad[2 * (2 - a) + 1] = 1
        return torch.nn.functional.pad(t, pad)

    d1, d2 = c.damp["ex"]
    eps_x = _avg(c.eps, 0)
    curl_x = pad_interior(
        (_d(Hz, 1)[:, :, 1:-1] - _d(Hy, 2)[:, 1:-1, :]) / dx, (1, 2))
    Ex = (d1 * state.Ex * eps_x + d2 * curl_x) / eps_x
    d1, d2 = c.damp["ey"]
    eps_y = _avg(c.eps, 1)
    curl_y = pad_interior(
        (_d(Hx, 2)[1:-1, :, :] - _d(Hz, 0)[:, :, 1:-1]) / dx, (0, 2))
    Ey = (d1 * state.Ey * eps_y + d2 * curl_y) / eps_y
    d1, d2 = c.damp["ez"]
    eps_z = _avg(c.eps, 2)
    curl_z = pad_interior(
        (_d(Hy, 0)[:, 1:-1, :] - _d(Hx, 1)[1:-1, :, :]) / dx, (0, 1))
    Ez = (d1 * state.Ez * eps_z + d2 * curl_z) / eps_z

    if src is not None:
        comp, i, j, k = src
        if comp == "Ez":
            Ez[i, j, k] = Ez[i, j, k] + c.dt * src_amp / eps_z[i, j, k]
        elif comp == "Ex":
            Ex[i, j, k] = Ex[i, j, k] + c.dt * src_amp / eps_x[i, j, k]
        else:
            Ey[i, j, k] = Ey[i, j, k] + c.dt * src_amp / eps_y[i, j, k]
    return State3D(Ex, Ey, Ez, Hx, Hy, Hz)
