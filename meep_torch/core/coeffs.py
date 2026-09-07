# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math
from typing import NamedTuple, Sequence

import torch

from .grid import Grid
from .pml import cpml_profiles, side_counts


def _axis_sides(npml):
    """npml as ((x_lo, x_hi), (y_lo, y_hi)) cell counts; an int means the
    same layer on every edge, a pair of pairs sets each edge (0 = PEC)."""
    if isinstance(npml, (tuple, list)):
        px, py = npml
        return side_counts(px), side_counts(py)
    return (int(npml), int(npml)), (int(npml), int(npml))


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
    # mirror half-domain at the LOW edge of an axis: 0 = off, +1 = even
    # (PMC-like wall, dEz/dn = 0), -1 = odd (Ez ghost = -Ez across the cut;
    # H holds the on-plane row/col). build_tm_coeffs y_mirror/x_mirror.
    y_mirror_lo: int = 0
    x_mirror_lo: int = 0
    # C2 (Rotate2 about z) fold at the LOW-y cut: 0 = off, else the user
    # phase +-1. Shares every shape/PML rule with an odd y mirror; only the
    # ghost differs -- Ez[i, -1] = phase * Ez[nx-1-i, 0] (reversed in x)
    # rather than phase * Ez[i, 0]. See stepper_tm.step_db.
    y_rot2_lo: int = 0
    # C4 (Rotate4 about z) fold: 0 = off, else the user phase +-1. Cuts BOTH
    # low edges (a quadrant, so it needs a square grid) and its ghosts are
    # transposes -- Ez[i, -1] = phase * Ez[0, i], Ez[-1, j] = phase * Ez[j, 0]
    # -- because C4 carries the y cut onto the x cut.
    rot4_lo: int = 0


def _as_field(val, grid: Grid, default: float) -> torch.Tensor:
    if val is None:
        val = default
    if isinstance(val, torch.Tensor):
        if val.shape != (grid.nx, grid.ny):
            raise ValueError(f"expected shape {(grid.nx, grid.ny)}, got {tuple(val.shape)}")
        return val.to(dtype=grid.dtype, device=grid.device)
    return torch.full((grid.nx, grid.ny), float(val), dtype=grid.dtype, device=grid.device)


def _pole_recurrence(pole, dt):
    """ADE recurrence scalars for one dispersive pole.

    pole = (fn, gamma, sigma) or (fn, gamma, sigma, fc): the restoring
    frequency w0 = 2 pi*fn drives c1/c2; the coupling frequency
    wc = 2 pi*(fc or fn) drives c3. fc defaults to fn (Lorentz, the
    existing 3-tuple behaviour); fc != fn gives a Drude pole
    (fn=0, fc=fp -> no restoring, coupling at the plasma frequency).
    Returns (c1, c2, c3) scalars; c3 still needs the spatial mask.
    """
    fn, gamma, sigma = pole[0], pole[1], pole[2]
    fc = pole[3] if len(pole) > 3 else fn
    w0 = 2 * math.pi * fn
    wc = 2 * math.pi * fc
    G = 2 * math.pi * gamma
    den = 1.0 + G * dt / 2.0
    c1 = (2.0 - w0 * w0 * dt * dt) / den
    c2 = -(1.0 - G * dt / 2.0) / den
    c3 = sigma * wc * wc * dt * dt / den
    return c1, c2, c3


def _axis_pole_coeffs(grid, dt, eps_ref, shared, axis_poles, shared_masks,
                      axis_masks, shape, avg):
    """Build (c1, c2, c3) for one TE E-axis: shared poles act on BOTH axes,
    axis_poles act on THIS axis only. Masks: shared_masks for the shared
    poles, axis_masks for the axis-only poles (None -> uniform). `avg`
    Yee-averages the cell-center mask onto the E-node shape."""
    allp = list(shared) + list(axis_poles)
    n = len(allp)
    c1 = grid.zeros(n, *shape)
    c2 = grid.zeros(n, *shape)
    c3 = grid.zeros(n, *shape)
    ns = len(shared)
    for k, pole in enumerate(allp):
        cv1, cv2, cv3 = _pole_recurrence(pole, dt)
        if k < ns:
            m = shared_masks[k] if shared_masks is not None else None
        else:
            m = axis_masks[k - ns] if axis_masks is not None else None
        mask = (m if m is not None else torch.ones_like(eps_ref)).to(eps_ref)
        c1[k] = cv1
        c2[k] = cv2
        c3[k] = cv3 * avg(mask)
    return c1, c2, c3


def build_tm_coeffs(
    grid: Grid,
    eps_inf=None,
    poles: Sequence = (),
    pole_masks: Sequence[torch.Tensor] | None = None,
    chi3=None,
    chi2=None,
    npml=0,          # int (all edges) or ((x_lo, x_hi), (y_lo, y_hi)) cells
    y_pmc: bool = False,
    sigma_d=None,
    sigma_b=None,
    mu=None,
    m_poles: Sequence = (),
    m_pole_masks: Sequence[torch.Tensor] | None = None,
    y_mirror: int = 0,
    x_mirror: int = 0,
    y_rot2: int = 0,
    rot4: int = 0,
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
        cv1, cv2, cv3 = _pole_recurrence(pole, dt)
        mask = pole_masks[k] if pole_masks is not None else torch.ones_like(eps)
        c1[k] = cv1 * torch.ones_like(eps)
        c2[k] = cv2 * torch.ones_like(eps)
        c3[k] = cv3 * mask.to(eps)

    px, py = _axis_sides(npml)
    # A symmetry cut replaces the low-side PML: the kept half spans from the
    # cut plane to the far edge, so only the FAR edge carries an absorber.
    # Drop the low count before the fits-in-the-grid check -- otherwise a
    # halved cell is measured against the full cell's two-sided PML budget
    # (metal-cavity-ldos.py: npml=((50,50),(50,50)) vs a 200x100 half grid).
    if y_mirror or y_rot2 or rot4:
        py = (0, py[1])
    if x_mirror or rot4:
        px = (0, px[1])
    if sum(px) >= grid.nx or (sum(py) >= grid.ny and not y_pmc):
        raise ValueError(f"npml={npml} too large for grid {grid.nx}x{grid.ny}")

    def prof(n, offset, pair, use_pml=True):
        return cpml_profiles(
            n, pair if use_pml else 0, offset, dt, dx, dtype=grid.dtype, device=grid.device
        )

    if (y_mirror not in (0, +1, -1) or x_mirror not in (0, +1, -1)
            or y_rot2 not in (0, +1, -1) or rot4 not in (0, +1, -1)):
        raise ValueError(f"symmetry parity must be 0/+1/-1, got y={y_mirror} "
                         f"x={x_mirror} rot2={y_rot2} rot4={rot4}")
    if y_mirror and y_rot2:
        raise ValueError("y_mirror and y_rot2 both cut the low-y edge")
    if rot4 and (y_mirror or x_mirror or y_rot2):
        raise ValueError("rot4 cuts both low edges on its own")
    if rot4 and grid.nx != grid.ny:
        raise ValueError(f"rot4 needs a square grid, got {grid.nx}x{grid.ny}")
    if y_pmc and (y_mirror or x_mirror or y_rot2 or rot4):
        raise ValueError("symmetry cuts and y_pmc are mutually exclusive")
    # every cut whose on-plane H row is live carries no conductivity/mu
    if (y_mirror == -1 or x_mirror == -1 or y_rot2 or rot4) and ("h1x" in cond or m_poles):
        raise NotImplementedError(
            "odd mirror / Rotate2 / Rotate4 with sigma_b/mu/m_poles: the on-plane H row "
            "has no conductivity/mu coefficients (standard H-grid shapes only)")

    def kill_lo(b, a, k, nlo):
        # a mirror wall replaces the low-side PML: identity profile there
        b[:nlo] = 0.0
        a[:nlo] = 0.0
        k[:nlo] = 1.0

    # H updates: dEz/dy at half-y (ny-1); dEz/dx at half-x (nx-1). The odd
    # mirror's on-plane H row lives in TMState.Hx0/Hy0 (no PML at the cut),
    # so the interior profiles are the same for both parities.
    b_hy, a_hy, k_hy = prof(grid.ny - 1, 0.5, py, use_pml=not y_pmc)
    b_hx, a_hx, k_hx = prof(grid.nx - 1, 0.5, px)
    if y_mirror or y_rot2 or rot4:
        kill_lo(b_hy, a_hy, k_hy, py[0])
    if x_mirror or rot4:
        kill_lo(b_hx, a_hx, k_hx, px[0])
    # D update: dHy/dx at integer x -- interior (PEC walls) or, with an
    # x mirror, cols 0..nx-2 (cut col live, far col PEC behind its PML)
    b_ex_f, a_ex_f, k_ex_f = prof(grid.nx, 0.0, px)
    if x_mirror or rot4:
        kill_lo(b_ex_f, a_ex_f, k_ex_f, px[0])
        b_ex, a_ex, k_ex = b_ex_f[:-1], a_ex_f[:-1], k_ex_f[:-1]
    else:
        b_ex, a_ex, k_ex = b_ex_f[1:-1], a_ex_f[1:-1], k_ex_f[1:-1]
    # dHx/dy at integer y: interior (PEC) or full (PMC)
    b_ey_f, a_ey_f, k_ey_f = prof(grid.ny, 0.0, py, use_pml=not y_pmc)
    if y_pmc:
        b_ey, a_ey, k_ey = b_ey_f, a_ey_f, k_ey_f
    elif y_mirror or y_rot2 or rot4:
        # fdtdx-style mirror: wall at the LOW-y cut, PML high-y only; the D
        # update keeps rows 0..ny-2 (cut row live, top row PEC behind PML)
        kill_lo(b_ey_f, a_ey_f, k_ey_f, py[0])
        b_ey, a_ey, k_ey = b_ey_f[:-1], a_ey_f[:-1], k_ey_f[:-1]
    else:
        b_ey, a_ey, k_ey = b_ey_f[1:-1], a_ey_f[1:-1], k_ey_f[1:-1]

    return TMCoeffs(
        dt=dt, dx=dx, eps_inf=eps, chi3=x3, has_kerr=has_kerr,
        chi2=x2, has_chi2=has_chi2,
        c1=c1, c2=c2, c3=c3, y_pmc=y_pmc,
        y_mirror_lo=y_mirror, x_mirror_lo=x_mirror, y_rot2_lo=y_rot2,
        rot4_lo=rot4,
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
    # Oriented (tensor) poles: cross-drive of Px by Ey / Py by Ex.
    # None -> stepper fast-path (grid-aligned poles only).
    c3xy: torch.Tensor | None = None        # (np_x, nx-1, ny)
    c3yx: torch.Tensor | None = None        # (np_y, nx, ny-1)


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
    npml=0,          # int (all edges) or ((x_lo, x_hi), (y_lo, y_hi)) cells
    sigma_d=None,
    sigma_b=None,
    mu=None,
    eps_tensor=None,
    poles_x: Sequence = (),
    poles_y: Sequence = (),
    pole_masks_x: Sequence[torch.Tensor] | None = None,
    pole_masks_y: Sequence[torch.Tensor] | None = None,
    poles_tensor: Sequence = (),
    pole_masks_tensor: Sequence[torch.Tensor] | None = None,
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

    # Per-axis poles: shared `poles` act on BOTH Ex and Ey; poles_x/poles_y act
    # on one axis only (per-axis Drude -> hyperbolic media). Empty by default
    # -> c1x/c1y both have len(poles), matching the pre-per-axis behaviour.
    c1x, c2x, c3x = _axis_pole_coeffs(
        grid, dt, eps, poles, poles_x, pole_masks, pole_masks_x,
        (grid.nx - 1, grid.ny), _avg_to_ex)
    c1y, c2y, c3y = _axis_pole_coeffs(
        grid, dt, eps, poles, poles_y, pole_masks, pole_masks_y,
        (grid.nx, grid.ny - 1), _avg_to_ey)

    # Oriented (tensor) poles: pole = (fn, gamma, (sxx, sxy, syy)) or
    # (fn, gamma, S, fc). The in-plane susceptibility tensor chi(f)*S
    # (S = u u^T for an oriented Lorentz oscillator) appends one pole to
    # EACH axis stack; the off-diagonal sxy drives Px from Yee-averaged Ey
    # (c3xy) and Py from Ex (c3yx) -- non-grid-aligned optic axes.
    c3xy = c3yx = None
    if len(poles_tensor):
        npx0, npy0 = c1x.shape[0], c1y.shape[0]
        tc1x = grid.zeros(len(poles_tensor), grid.nx - 1, grid.ny)
        tc2x, tc3x, tc3xy = (torch.zeros_like(tc1x) for _ in range(3))
        tc1y = grid.zeros(len(poles_tensor), grid.nx, grid.ny - 1)
        tc2y, tc3y, tc3yx = (torch.zeros_like(tc1y) for _ in range(3))
        for k, pole in enumerate(poles_tensor):
            fn, gamma, (sxx, sxy, syy) = pole[0], pole[1], pole[2]
            fc = pole[3] if len(pole) > 3 else fn
            cv1, cv2, cv3 = _pole_recurrence((fn, gamma, 1.0, fc), dt)
            m = (pole_masks_tensor[k] if pole_masks_tensor is not None
                 else torch.ones_like(eps)).to(eps)
            mx, my = _avg_to_ex(m), _avg_to_ey(m)
            tc1x[k], tc2x[k] = cv1, cv2
            tc1y[k], tc2y[k] = cv1, cv2
            tc3x[k], tc3xy[k] = cv3 * sxx * mx, cv3 * sxy * mx
            tc3y[k], tc3yx[k] = cv3 * syy * my, cv3 * sxy * my
        c1x = torch.cat([c1x, tc1x]); c2x = torch.cat([c2x, tc2x])
        c3x = torch.cat([c3x, tc3x])
        c1y = torch.cat([c1y, tc1y]); c2y = torch.cat([c2y, tc2y])
        c3y = torch.cat([c3y, tc3y])
        c3xy = torch.cat([grid.zeros(npx0, grid.nx - 1, grid.ny), tc3xy])
        c3yx = torch.cat([grid.zeros(npy0, grid.nx, grid.ny - 1), tc3yx])
        cond["c3xy"], cond["c3yx"] = c3xy, c3yx

    px, py = _axis_sides(npml)
    if sum(px) >= grid.nx or sum(py) >= grid.ny:
        raise ValueError(f"npml={npml} too large for grid {grid.nx}x{grid.ny}")

    def prof(n, offset, pair):
        return cpml_profiles(n, pair, offset, dt, dx, dtype=grid.dtype, device=grid.device)

    # Hz update: dEy/dx at half-x (nx-1); dEx/dy at half-y (ny-1)
    b_hx, a_hx, k_hx = prof(grid.nx - 1, 0.5, px)
    b_hy, a_hy, k_hy = prof(grid.ny - 1, 0.5, py)
    # Dx update: dHz/dy at interior integer y (positions 1..ny-2)
    b_ey_f, a_ey_f, k_ey_f = prof(grid.ny, 0.0, py)
    b_ey, a_ey, k_ey = b_ey_f[1:-1], a_ey_f[1:-1], k_ey_f[1:-1]
    # Dy update: dHz/dx at interior integer x (positions 1..nx-2)
    b_ex_f, a_ex_f, k_ex_f = prof(grid.nx, 0.0, px)
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
