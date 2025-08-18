# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math
from typing import NamedTuple, Sequence

import torch

from .grid import Grid
from .pml import cpml_profiles


class TMCoeffs(NamedTuple):
    dt: float
    dx: float
    eps_inf: torch.Tensor
    chi3: torch.Tensor
    has_kerr: bool
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    y_pmc: bool
    b_hy: torch.Tensor
    a_hy: torch.Tensor
    k_hy: torch.Tensor
    b_hx: torch.Tensor
    a_hx: torch.Tensor
    k_hx: torch.Tensor
    b_ex: torch.Tensor
    a_ex: torch.Tensor
    k_ex: torch.Tensor
    b_ey: torch.Tensor
    a_ey: torch.Tensor
    k_ey: torch.Tensor
    # W5 conductivity/mu (None -> stepper fast-path without them)
    d1: torch.Tensor | None = None     # (nx, ny)   Dz damping
    d2: torch.Tensor | None = None     # (nx, ny)   Dz curl/source coefficient
    h1x: torch.Tensor | None = None    # (nx, ny-1) Hx damping
    h2x: torch.Tensor | None = None    # (nx, ny-1) Hx curl coefficient (dt/mu)
    h1y: torch.Tensor | None = None    # (nx-1, ny) Hy damping
    h2y: torch.Tensor | None = None    # (nx-1, ny) Hy curl coefficient
    # W5.5 magnetic Lorentz poles (mu dispersion), duals of c1/c2/c3
    mc1x: torch.Tensor | None = None   # (nm, nx, ny-1) on the Hx grid
    mc2x: torch.Tensor | None = None
    mc3x: torch.Tensor | None = None
    mc1y: torch.Tensor | None = None   # (nm, nx-1, ny) on the Hy grid
    mc2y: torch.Tensor | None = None
    mc3y: torch.Tensor | None = None
    # W5.75 instantaneous chi2 (second-order): None -> has_chi2 fast-path
    chi2: torch.Tensor | None = None      # (nx, ny)
    has_chi2: bool = False


def _as_field(val, grid: Grid, default: float) -> torch.Tensor:
    if val is None:
        val = default
    if isinstance(val, torch.Tensor):
        if val.shape != (grid.nx, grid.ny):
            raise ValueError(f"expected shape {(grid.nx, grid.ny)}, got {tuple(val.shape)}")
        return val.to(dtype=grid.dtype, device=grid.device)
    return torch.full((grid.nx, grid.ny), float(val), dtype=grid.dtype, device=grid.device)


def build_tm_coeffs(
    grid: Grid,
    eps_inf=None,
    poles: Sequence = (),
    pole_masks: Sequence[torch.Tensor] | None = None,
    chi3=None,
    chi2=None,
    npml: int = 0,
    y_pmc: bool = False,
    sigma_d=None,
    sigma_b=None,
    mu=None,
    m_poles: Sequence = (),
    m_pole_masks: Sequence[torch.Tensor] | None = None,
) -> TMCoeffs:
    dt, dx = grid.dt, grid.dx
    eps = _as_field(eps_inf, grid, 1.0)
    x3 = _as_field(chi3, grid, 0.0)
    x2 = _as_field(chi2, grid, 0.0)
    has_kerr = bool((x3 != 0).any())
    has_chi2 = bool((x2 != 0).any())

    cond = {}
    if sigma_d is not None or sigma_b is not None or mu is not None:
        sd = _as_field(sigma_d, grid, 0.0)
        sb = _as_field(sigma_b, grid, 0.0)
        mu_ = _as_field(mu, grid, 1.0)
        cond["d1"] = (1 - sd * dt / 2) / (1 + sd * dt / 2)
        cond["d2"] = dt / (1 + sd * dt / 2)
        # H positions: Hx (nx, ny-1) averaged along y; Hy (nx-1, ny) along x
        sbx = 0.5 * (sb[:, :-1] + sb[:, 1:])
        mux = 0.5 * (mu_[:, :-1] + mu_[:, 1:])
        sby = 0.5 * (sb[:-1, :] + sb[1:, :])
        muy = 0.5 * (mu_[:-1, :] + mu_[1:, :])
        cond["h1x"] = (1 - sbx * dt / 2) / (1 + sbx * dt / 2)
        cond["h2x"] = dt / (mux * (1 + sbx * dt / 2))
        cond["h1y"] = (1 - sby * dt / 2) / (1 + sby * dt / 2)
        cond["h2y"] = dt / (muy * (1 + sby * dt / 2))

    if m_poles:
        if cond:
            raise NotImplementedError("m_poles with sigma_b/mu not combined (YAGNI)")
        nm = len(m_poles)
        for suf, shape, avg in (("x", (grid.nx, grid.ny - 1), lambda v: 0.5 * (v[:, :-1] + v[:, 1:])),
                                ("y", (grid.nx - 1, grid.ny), lambda v: 0.5 * (v[:-1, :] + v[1:, :]))):
            c1m = grid.zeros(nm, *shape)
            c2m = grid.zeros(nm, *shape)
            c3m = grid.zeros(nm, *shape)
            for k, pole in enumerate(m_poles):
                fn, gamma, sigma = pole[0], pole[1], pole[2]
                w0 = 2 * math.pi * fn
                G = 2 * math.pi * gamma
                den = 1.0 + G * dt / 2.0
                mask = (m_pole_masks[k] if m_pole_masks is not None
                        else torch.ones(grid.nx, grid.ny, dtype=grid.dtype, device=grid.device))
                c1m[k] = (2.0 - w0 * w0 * dt * dt) / den
                c2m[k] = -(1.0 - G * dt / 2.0) / den
                c3m[k] = (sigma * w0 * w0 * dt * dt / den) * avg(mask.to(grid.dtype))
            cond[f"mc1{suf}"], cond[f"mc2{suf}"], cond[f"mc3{suf}"] = c1m, c2m, c3m

    n_poles = len(poles)
    c1 = grid.zeros(n_poles, grid.nx, grid.ny)
    c2 = grid.zeros(n_poles, grid.nx, grid.ny)
    c3 = grid.zeros(n_poles, grid.nx, grid.ny)
    for k, pole in enumerate(poles):
        fn, gamma, sigma = pole[0], pole[1], pole[2]
        w0 = 2 * math.pi * fn
        G = 2 * math.pi * gamma
        den = 1.0 + G * dt / 2.0
        mask = pole_masks[k] if pole_masks is not None else torch.ones_like(eps)
        c1[k] = (2.0 - w0 * w0 * dt * dt) / den * torch.ones_like(eps)
        c2[k] = -(1.0 - G * dt / 2.0) / den * torch.ones_like(eps)
        c3[k] = (sigma * w0 * w0 * dt * dt / den) * mask.to(eps)

    if npml * 2 >= grid.nx or (npml * 2 >= grid.ny and not y_pmc):
        raise ValueError(f"npml={npml} too large for grid {grid.nx}x{grid.ny}")

    def prof(n, offset, use_pml=True):
        return cpml_profiles(
            n, npml if use_pml else 0, offset, dt, dx, dtype=grid.dtype, device=grid.device
        )

    # H updates: dEz/dy at half-y (ny-1); dEz/dx at half-x (nx-1)
    b_hy, a_hy, k_hy = prof(grid.ny - 1, 0.5, use_pml=not y_pmc)
    b_hx, a_hx, k_hx = prof(grid.nx - 1, 0.5)
    # D update: dHy/dx at interior integer x (positions 1..nx-2)
    b_ex_f, a_ex_f, k_ex_f = prof(grid.nx, 0.0)
    b_ex, a_ex, k_ex = b_ex_f[1:-1], a_ex_f[1:-1], k_ex_f[1:-1]
    # dHx/dy at integer y: interior (PEC) or full (PMC)
    b_ey_f, a_ey_f, k_ey_f = prof(grid.ny, 0.0, use_pml=not y_pmc)
    if y_pmc:
        b_ey, a_ey, k_ey = b_ey_f, a_ey_f, k_ey_f
    else:
        b_ey, a_ey, k_ey = b_ey_f[1:-1], a_ey_f[1:-1], k_ey_f[1:-1]

    return TMCoeffs(
        dt=dt, dx=dx, eps_inf=eps, chi3=x3, has_kerr=has_kerr,
        chi2=x2, has_chi2=has_chi2,
        c1=c1, c2=c2, c3=c3, y_pmc=y_pmc,
        b_hy=b_hy[None, :], a_hy=a_hy[None, :], k_hy=k_hy[None, :],
        b_hx=b_hx[:, None], a_hx=a_hx[:, None], k_hx=k_hx[:, None],
        b_ex=b_ex[:, None], a_ex=a_ex[:, None], k_ex=k_ex[:, None],
        b_ey=b_ey[None, :], a_ey=a_ey[None, :], k_ey=k_ey[None, :],
        **cond,
    )


class TECoeffs(NamedTuple):
    dt: float
    dx: float
    eps_x: torch.Tensor    # (nx-1, ny)   at Ex positions
    eps_y: torch.Tensor    # (nx, ny-1)   at Ey positions
    chi3_x: torch.Tensor   # (nx-1, ny)
    chi3_y: torch.Tensor   # (nx, ny-1)
    has_kerr: bool
    c1x: torch.Tensor      # (np, nx-1, ny)
    c2x: torch.Tensor
    c3x: torch.Tensor
    c1y: torch.Tensor      # (np, nx, ny-1)
    c2y: torch.Tensor
    c3y: torch.Tensor
    b_hx: torch.Tensor     # (nx-1, 1)  half-x, Hz update dEy/dx
    a_hx: torch.Tensor
    k_hx: torch.Tensor
    b_hy: torch.Tensor     # (1, ny-1)  half-y, Hz update dEx/dy
    a_hy: torch.Tensor
    k_hy: torch.Tensor
    b_ey: torch.Tensor     # (1, ny-2)  integer-y interior, Dx update dHz/dy
    a_ey: torch.Tensor
    k_ey: torch.Tensor
    b_ex: torch.Tensor     # (nx-2, 1)  integer-x interior, Dy update dHz/dx
    a_ex: torch.Tensor
    k_ex: torch.Tensor
    # W5 conductivity/mu (None -> stepper fast-path without them)
    d1x: torch.Tensor | None = None    # (nx-1, ny)   Dx damping
    d2x: torch.Tensor | None = None
    d1y: torch.Tensor | None = None    # (nx, ny-1)   Dy damping
    d2y: torch.Tensor | None = None
    hz1: torch.Tensor | None = None    # (nx-1, ny-1) Hz damping
    hz2: torch.Tensor | None = None    # (nx-1, ny-1) Hz curl coefficient (dt/mu)
    # W5 in-plane anisotropic eps: inverse-tensor components at E positions
    inv_xx_x: torch.Tensor | None = None   # (nx-1, ny)
    inv_xy_x: torch.Tensor | None = None   # (nx-1, ny)
    inv_yy_y: torch.Tensor | None = None   # (nx, ny-1)
    inv_xy_y: torch.Tensor | None = None   # (nx, ny-1)
    # W5.75 instantaneous chi2 (second-order): None -> has_chi2 fast-path
    chi2_x: torch.Tensor | None = None      # (nx-1, ny)
    chi2_y: torch.Tensor | None = None      # (nx, ny-1)
    has_chi2: bool = False


def _avg_to_ex(v: torch.Tensor) -> torch.Tensor:
    # cell centers (nx, ny) -> Ex positions (nx-1, ny)
    return 0.5 * (v[:-1, :] + v[1:, :])


def _avg_to_ey(v: torch.Tensor) -> torch.Tensor:
    # cell centers (nx, ny) -> Ey positions (nx, ny-1)
    return 0.5 * (v[:, :-1] + v[:, 1:])


def build_te_coeffs(
    grid: Grid,
    eps_inf=None,
    poles: Sequence = (),
    pole_masks: Sequence[torch.Tensor] | None = None,
    chi3=None,
    chi2=None,
    npml: int = 0,
    sigma_d=None,
    sigma_b=None,
    mu=None,
    eps_tensor=None,
) -> TECoeffs:
    dt, dx = grid.dt, grid.dx
    eps = _as_field(eps_inf, grid, 1.0)
    x3 = _as_field(chi3, grid, 0.0)
    x2 = _as_field(chi2, grid, 0.0)
    has_kerr = bool((x3 != 0).any())
    has_chi2 = bool((x2 != 0).any())

    eps_x, eps_y = _avg_to_ex(eps), _avg_to_ey(eps)
    chi3_x, chi3_y = _avg_to_ex(x3), _avg_to_ey(x3)
    chi2_x, chi2_y = _avg_to_ex(x2), _avg_to_ey(x2)

    cond = {}
    if eps_tensor is not None:
        # symmetric in-plane tensor (exx, exy, eyy) at cell centers ->
        # inverse components Yee-averaged to the E positions
        exx = _as_field(eps_tensor[0], grid, 1.0)
        exy = _as_field(eps_tensor[1], grid, 0.0)
        eyy = _as_field(eps_tensor[2], grid, 1.0)
        det = exx * eyy - exy * exy
        cond["inv_xx_x"] = _avg_to_ex(eyy / det)
        cond["inv_xy_x"] = _avg_to_ex(-exy / det)
        cond["inv_yy_y"] = _avg_to_ey(exx / det)
        cond["inv_xy_y"] = _avg_to_ey(-exy / det)
    if sigma_d is not None or sigma_b is not None or mu is not None:
        sd = _as_field(sigma_d, grid, 0.0)
        sb = _as_field(sigma_b, grid, 0.0)
        mu_ = _as_field(mu, grid, 1.0)
        sdx, sdy = _avg_to_ex(sd), _avg_to_ey(sd)
        cond["d1x"] = (1 - sdx * dt / 2) / (1 + sdx * dt / 2)
        cond["d2x"] = dt / (1 + sdx * dt / 2)
        cond["d1y"] = (1 - sdy * dt / 2) / (1 + sdy * dt / 2)
        cond["d2y"] = dt / (1 + sdy * dt / 2)
        # Hz positions (nx-1, ny-1): 4-point average of cell centers
        sbz = 0.25 * (sb[:-1, :-1] + sb[1:, :-1] + sb[:-1, 1:] + sb[1:, 1:])
        muz = 0.25 * (mu_[:-1, :-1] + mu_[1:, :-1] + mu_[:-1, 1:] + mu_[1:, 1:])
        cond["hz1"] = (1 - sbz * dt / 2) / (1 + sbz * dt / 2)
        cond["hz2"] = dt / (muz * (1 + sbz * dt / 2))

    n_poles = len(poles)
    c1x = grid.zeros(n_poles, grid.nx - 1, grid.ny)
    c2x = grid.zeros(n_poles, grid.nx - 1, grid.ny)
    c3x = grid.zeros(n_poles, grid.nx - 1, grid.ny)
    c1y = grid.zeros(n_poles, grid.nx, grid.ny - 1)
    c2y = grid.zeros(n_poles, grid.nx, grid.ny - 1)
    c3y = grid.zeros(n_poles, grid.nx, grid.ny - 1)
    for k, pole in enumerate(poles):
        fn, gamma, sigma = pole[0], pole[1], pole[2]
        w0 = 2 * math.pi * fn
        G = 2 * math.pi * gamma
        den = 1.0 + G * dt / 2.0
        mask = pole_masks[k] if pole_masks is not None else torch.ones_like(eps)
        mask = mask.to(eps)
        c1x[k] = (2.0 - w0 * w0 * dt * dt) / den * torch.ones_like(eps_x)
        c2x[k] = -(1.0 - G * dt / 2.0) / den * torch.ones_like(eps_x)
        c3x[k] = (sigma * w0 * w0 * dt * dt / den) * _avg_to_ex(mask)
        c1y[k] = (2.0 - w0 * w0 * dt * dt) / den * torch.ones_like(eps_y)
        c2y[k] = -(1.0 - G * dt / 2.0) / den * torch.ones_like(eps_y)
        c3y[k] = (sigma * w0 * w0 * dt * dt / den) * _avg_to_ey(mask)

    if npml * 2 >= grid.nx or npml * 2 >= grid.ny:
        raise ValueError(f"npml={npml} too large for grid {grid.nx}x{grid.ny}")

    def prof(n, offset):
        return cpml_profiles(n, npml, offset, dt, dx, dtype=grid.dtype, device=grid.device)

    # Hz update: dEy/dx at half-x (nx-1); dEx/dy at half-y (ny-1)
    b_hx, a_hx, k_hx = prof(grid.nx - 1, 0.5)
    b_hy, a_hy, k_hy = prof(grid.ny - 1, 0.5)
    # Dx update: dHz/dy at interior integer y (positions 1..ny-2)
    b_ey_f, a_ey_f, k_ey_f = prof(grid.ny, 0.0)
    b_ey, a_ey, k_ey = b_ey_f[1:-1], a_ey_f[1:-1], k_ey_f[1:-1]
    # Dy update: dHz/dx at interior integer x (positions 1..nx-2)
    b_ex_f, a_ex_f, k_ex_f = prof(grid.nx, 0.0)
    b_ex, a_ex, k_ex = b_ex_f[1:-1], a_ex_f[1:-1], k_ex_f[1:-1]

    return TECoeffs(
        dt=dt, dx=dx,
        eps_x=eps_x, eps_y=eps_y, chi3_x=chi3_x, chi3_y=chi3_y, has_kerr=has_kerr,
        chi2_x=chi2_x, chi2_y=chi2_y, has_chi2=has_chi2,
        c1x=c1x, c2x=c2x, c3x=c3x, c1y=c1y, c2y=c2y, c3y=c3y,
        b_hx=b_hx[:, None], a_hx=a_hx[:, None], k_hx=k_hx[:, None],
        b_hy=b_hy[None, :], a_hy=a_hy[None, :], k_hy=k_hy[None, :],
        b_ey=b_ey[None, :], a_ey=a_ey[None, :], k_ey=k_ey[None, :],
        b_ex=b_ex[:, None], a_ex=a_ex[:, None], k_ex=k_ex[:, None],
        **cond,
    )
