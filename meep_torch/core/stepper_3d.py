# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Full 3D Yee stepper (W9): real fields, scalar eps. Wave 1 shipped a
matched-sigma absorber shell; wave 2 adds a true CFS-PML (CPML) and the
material D/B-conductivity fast-path, mirroring the 2D TM stepper. Courant
dt = 0.5 dx / sqrt(3).

Yee positions (i, j, k integers):
  Ex(i+1/2, j, k)  Ey(i, j+1/2, k)  Ez(i, j, k+1/2)
  Hx(i, j+1/2, k+1/2)  Hy(i+1/2, j, k+1/2)  Hz(i+1/2, j+1/2, k)

Boundary mode: ``npml > 0`` selects CPML (12 psi memory variables, one per
(component, derivative-axis)); otherwise the wave-1 matched-sigma absorber
(``n_abs > 0``) is used and the psi fields stay None (zero overhead).
Material ``sigma_d`` (damps E) / ``sigma_b`` (damps H) compose into the
same d1/d2 fast-path as the 2D stepper and are independent of the boundary
mode (a lossy slab inside a CPML cell works).
"""
import math
from typing import NamedTuple

import torch

from .coeffs import _pole_recurrence
from .pml import absorber_sigma, cpml_profiles


class Coeffs3D(NamedTuple):
    dt: float
    dx: float
    eps: torch.Tensor        # (nx, ny, nz) cell-centered
    damp: dict              # per-grid (d1, d2): material+absorber conductivity
    cpml: dict | None       # per-axis (b, a, k) profiles, None in absorber mode
    pols: dict | None = None  # "x"/"y"/"z" -> (c1, c2, c3) ADE pole stacks


class State3D(NamedTuple):
    Ex: torch.Tensor         # (nx-1, ny, nz)
    Ey: torch.Tensor         # (nx, ny-1, nz)
    Ez: torch.Tensor         # (nx, ny, nz-1)
    Hx: torch.Tensor         # (nx, ny-1, nz-1)
    Hy: torch.Tensor         # (nx-1, ny, nz-1)
    Hz: torch.Tensor         # (nx-1, ny-1, nz)
    # CPML memory variables (None in absorber mode). 6 H-update (E-derivative)
    # + 6 E-update (H-derivative); one per (component, derivative-axis).
    psi_Ez_y: torch.Tensor | None = None   # (nx, ny-1, nz-1)  Hx <- dEz/dy
    psi_Ey_z: torch.Tensor | None = None   # (nx, ny-1, nz-1)  Hx <- dEy/dz
    psi_Ex_z: torch.Tensor | None = None   # (nx-1, ny, nz-1)  Hy <- dEx/dz
    psi_Ez_x: torch.Tensor | None = None   # (nx-1, ny, nz-1)  Hy <- dEz/dx
    psi_Ey_x: torch.Tensor | None = None   # (nx-1, ny-1, nz)   Hz <- dEy/dx
    psi_Ex_y: torch.Tensor | None = None   # (nx-1, ny-1, nz)   Hz <- dEx/dy
    psi_Hz_y: torch.Tensor | None = None   # (nx-1, ny-2, nz)   Ex <- dHz/dy
    psi_Hy_z: torch.Tensor | None = None   # (nx-1, ny, nz-2)   Ex <- dHy/dz
    psi_Hx_z: torch.Tensor | None = None   # (nx, ny-1, nz-2)   Ey <- dHx/dz
    psi_Hz_x: torch.Tensor | None = None   # (nx-2, ny-1, nz)   Ey <- dHz/dx
    psi_Hy_x: torch.Tensor | None = None   # (nx-2, ny, nz-1)   Ez <- dHy/dx
    psi_Hx_y: torch.Tensor | None = None   # (nx, ny-2, nz-1)   Ez <- dHx/dy
    # ADE polarization buffers (allocated only when n_poles > 0): per E comp a
    # (n_poles, *comp_shape) cur/prev pair for P(n+1) = c1 P(n) + c2 P(n-1) +
    # c3 E(n).
    Px_cur: torch.Tensor | None = None     # (np, nx-1, ny, nz)
    Px_prev: torch.Tensor | None = None
    Py_cur: torch.Tensor | None = None     # (np, nx, ny-1, nz)
    Py_prev: torch.Tensor | None = None
    Pz_cur: torch.Tensor | None = None     # (np, nx, ny, nz-1)
    Pz_prev: torch.Tensor | None = None


def _avg(v, axis):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(None, -1)
    sl1[axis] = slice(1, None)
    return 0.5 * (v[tuple(sl0)] + v[tuple(sl1)])


def _d(v, axis):
    sl0 = [slice(None)] * 3
    sl1 = [slice(None)] * 3
    sl0[axis] = slice(None, -1)
    sl1[axis] = slice(1, None)
    return v[tuple(sl1)] - v[tuple(sl0)]


def _pad_interior(t, axes):
    """Zero-pad one cell on both sides of each axis in ``axes`` (PEC outer
    edge for the tangential E components)."""
    pad = [0, 0, 0, 0, 0, 0]
    for a in axes:
        pad[2 * (2 - a)] = 1
        pad[2 * (2 - a) + 1] = 1
    return torch.nn.functional.pad(t, pad)


def _prof(n, npml, offset, dt, dx, dtype, device):
    return cpml_profiles(n, npml, offset, dt, dx, dtype=dtype, device=device)


def _bc(v, ax):
    """Reshape a 1D profile ``v`` to live along axis ``ax`` (length on ``ax``,
    1 elsewhere) for broadcasting against a 3D field."""
    sh = [1, 1, 1]
    sh[ax] = v.shape[0]
    return v.reshape(sh)


def _ident(n, ax, dtype, device):
    """Identity CPML profile (b=0, a=0, k=1): the recursion ``psi'=b*psi+a*d``
    then keeps ``psi=0`` and ``ce=d/k+psi=d`` -- a plain derivative. Used on
    axes handled by the matched-sigma absorber in mixed mode."""
    sh = [1, 1, 1]
    sh[ax] = n
    z = torch.zeros(sh, dtype=dtype, device=device)
    return z, z.clone(), torch.ones(sh, dtype=dtype, device=device)


def build_3d_coeffs(nx: int, ny: int, nz: int, dx: float, eps=None,
                    npml: int = 0, n_abs: int = 0, absorber_axes=(),
                    sigma_d=None, sigma_b=None, poles=(), pole_masks=None,
                    dtype=torch.float64, device="cpu", absorber_m: int = 2,
                    absorber_R0: float = 1e-15) -> Coeffs3D:
    """Boundary modes: full CPML (``npml>0``, ``n_abs=0``), full matched-sigma
    absorber (``npml=0``, ``n_abs>0``), or mixed (``npml>0`` and ``n_abs>0``):
    CPML on the axes not in ``absorber_axes``, matched-sigma absorber on the
    axes in ``absorber_axes`` (via an identity CPML profile -- plain derivative
    -- plus the absorber sigma in ``sig_bound``). Mixed mode isolates one
    boundary for a clean CPML-vs-absorber comparison (e.g. a z-uniform
    planewave with ``absorber_axes=(2,)`` sees only the z-boundary; the x/y
    CPML is inert for the uniform transverse field)."""
    dt = 0.5 * dx / math.sqrt(3.0)
    e = (eps if isinstance(eps, torch.Tensor)
         else torch.ones(nx, ny, nz, dtype=dtype, device=device))
    dims = (nx, ny, nz)
    mixed = npml > 0 and n_abs > 0
    ax_set = set(absorber_axes) if mixed else set()

    # --- boundary profiles / sigma ---
    if npml > 0:
        cpml = {}
        for ax, kh, ke in ((0, "hx", "ex"), (1, "hy", "ey"), (2, "hz", "ez")):
            n = dims[ax]
            if ax in ax_set:                       # absorber axis: plain derivative
                bH, aH, kH = _ident(n - 1, ax, dtype, device)
                bE, aE, kE = _ident(n - 2, ax, dtype, device)
            else:                                  # CPML axis
                b, a, k = _prof(n - 1, npml, 0.5, dt, dx, dtype, device)
                bH, aH, kH = _bc(b, ax), _bc(a, ax), _bc(k, ax)
                b, a, k = _prof(n, npml, 0.0, dt, dx, dtype, device)
                bE, aE, kE = _bc(b[1:-1], ax), _bc(a[1:-1], ax), _bc(k[1:-1], ax)
            cpml[kh] = (bH, aH, kH)
            cpml[ke] = (bE, aE, kE)
        if mixed:                                  # absorber sigma on absorber axes
            sig_bound = torch.zeros(nx, ny, nz, dtype=dtype, device=device)
            for ax in absorber_axes:
                s = absorber_sigma(dims[ax], n_abs, dx, m=absorber_m,
                                   R0=absorber_R0, dtype=dtype, device=device)
                sig_bound = torch.maximum(sig_bound, _bc(s, ax).expand(nx, ny, nz))
        else:
            sig_bound = torch.zeros(nx, ny, nz, dtype=dtype, device=device)
    else:
        cpml = None
        if n_abs > 0:
            sx = absorber_sigma(nx, n_abs, dx, m=absorber_m, R0=absorber_R0,
                                dtype=dtype, device=device)
            sy = absorber_sigma(ny, n_abs, dx, m=absorber_m, R0=absorber_R0,
                                dtype=dtype, device=device)
            sz = absorber_sigma(nz, n_abs, dx, m=absorber_m, R0=absorber_R0,
                                dtype=dtype, device=device)
            sig_bound = torch.maximum(torch.maximum(sx[:, None, None],
                                                    sy[None, :, None]),
                                      sz[None, None, :])
        else:
            sig_bound = torch.zeros(nx, ny, nz, dtype=dtype, device=device)

    # --- material D/B-conductivity (composes with the boundary absorber;
    # in CPML mode sig_bound is zero, so this is material-only) ---
    sd = (sigma_d if isinstance(sigma_d, torch.Tensor)
          else torch.zeros(nx, ny, nz, dtype=dtype, device=device))
    sb = (sigma_b if isinstance(sigma_b, torch.Tensor)
          else torch.zeros(nx, ny, nz, dtype=dtype, device=device))
    sig_e = sig_bound + sd
    sig_h = sig_bound + sb

    def dpair(s):
        return ((1 - s * dt / 2) / (1 + s * dt / 2), dt / (1 + s * dt / 2))

    damp = {
        "ex": dpair(_avg(sig_e, 0)), "ey": dpair(_avg(sig_e, 1)),
        "ez": dpair(_avg(sig_e, 2)),
        "hx": dpair(_avg(_avg(sig_h, 1), 2)),
        "hy": dpair(_avg(_avg(sig_h, 0), 2)),
        "hz": dpair(_avg(_avg(sig_h, 0), 1)),
    }

    # --- ADE poles (Lorentz/Drude, same recurrence scalars as the 2D
    # stepper via coeffs._pole_recurrence): per E comp a (c1, c2, c3) stack.
    # c1/c2 are pole scalars shaped (np, 1, 1, 1); c3 = cv3 * Yee-averaged
    # mask on the comp grid, (np, *comp_shape). Masks are cell-centered
    # (nx, ny, nz); keep them off the PML that damps their axis -- the
    # recurrence is not PML-split, so a pole active on a PML-damped E feeds
    # an un-damped P back into the damped E (growth mode, see the 2D
    # test_peraxis_dispersion note). ---
    pols = None
    if len(poles):
        pols = {}
        for key, shape, ax in (("x", (nx - 1, ny, nz), 0),
                               ("y", (nx, ny - 1, nz), 1),
                               ("z", (nx, ny, nz - 1), 2)):
            np_ = len(poles)
            c3 = torch.zeros(np_, *shape, dtype=dtype, device=device)
            vs = []
            for p, pole in enumerate(poles):
                v1, v2, v3 = _pole_recurrence(pole, dt)
                m = (pole_masks[p] if pole_masks is not None else
                     torch.ones(nx, ny, nz, dtype=dtype, device=device))
                c3[p] = v3 * _avg(m.to(dtype=dtype, device=device), ax)
                vs.append((v1, v2))
            c1 = torch.tensor([v[0] for v in vs], dtype=dtype,
                              device=device).reshape(np_, 1, 1, 1)
            c2 = torch.tensor([v[1] for v in vs], dtype=dtype,
                              device=device).reshape(np_, 1, 1, 1)
            pols[key] = (c1, c2, c3)
    return Coeffs3D(dt=dt, dx=dx, eps=e, damp=damp, cpml=cpml, pols=pols)


def init_3d_state(nx: int, ny: int, nz: int, npml: int = 0, n_poles: int = 0,
                  dtype=torch.float64, device="cpu") -> State3D:
    z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
    psi = {}
    if npml > 0:
        psi = dict(
            psi_Ez_y=z(nx, ny - 1, nz - 1), psi_Ey_z=z(nx, ny - 1, nz - 1),
            psi_Ex_z=z(nx - 1, ny, nz - 1), psi_Ez_x=z(nx - 1, ny, nz - 1),
            psi_Ey_x=z(nx - 1, ny - 1, nz), psi_Ex_y=z(nx - 1, ny - 1, nz),
            psi_Hz_y=z(nx - 1, ny - 2, nz), psi_Hy_z=z(nx - 1, ny, nz - 2),
            psi_Hx_z=z(nx, ny - 1, nz - 2), psi_Hz_x=z(nx - 2, ny - 1, nz),
            psi_Hy_x=z(nx - 2, ny, nz - 1), psi_Hx_y=z(nx, ny - 2, nz - 1),
        )
    if n_poles:
        psi.update(
            Px_cur=z(n_poles, nx - 1, ny, nz), Px_prev=z(n_poles, nx - 1, ny, nz),
            Py_cur=z(n_poles, nx, ny - 1, nz), Py_prev=z(n_poles, nx, ny - 1, nz),
            Pz_cur=z(n_poles, nx, ny, nz - 1), Pz_prev=z(n_poles, nx, ny, nz - 1),
        )
    return State3D(
        z(nx - 1, ny, nz), z(nx, ny - 1, nz), z(nx, ny, nz - 1),
        z(nx, ny - 1, nz - 1), z(nx - 1, ny, nz - 1), z(nx - 1, ny - 1, nz),
        **psi,
    )


def _e_pole(E_old, P_cur, P_new, d1, d2, eps_c, curl):
    """Folded-D E update with ADE poles: D = eps_inf E + sum P, so
    E(n+1) = (d1 (eps_inf E^n + sum P^n) + d2 curl - sum P^(n+1)) / eps_inf.
    With no poles this reduces to the plain update below."""
    return (d1 * (E_old * eps_c + P_cur.sum(0)) + d2 * curl
            - P_new.sum(0)) / eps_c


def step_3d(state: State3D, c: Coeffs3D, src_amp: float = 0.0,
            src=None) -> State3D:
    """One leapfrog step. ``src`` is either ``("Ez"|"Ex"|"Ey", i, j, k)`` for
    a soft point current or ``("Ez"|"Ex"|"Ey", mask)`` for an area/plane
    soft current (mask matches the component's node shape)."""
    dx = c.dx
    cp = c.cpml

    # ADE pole advance from E^n (meep's update_pols runs after step_db(D)
    # but before update_eh(E); the source injects after, so P precesses on
    # the pre-source E -- computing it here from the incoming E is exact).
    pols = c.pols
    pn = pout = None
    if pols is not None:
        pn = {
            key: pols[key][0] * getattr(state, f"P{key}_cur")
            + pols[key][1] * getattr(state, f"P{key}_prev")
            + pols[key][2] * getattr(state, "E" + key)
            for key in "xyz"
        }
        pout = {f"P{key}_cur": pn[key] for key in "xyz"}
        pout.update({f"P{key}_prev": getattr(state, f"P{key}_cur")
                    for key in "xyz"})

    if cp is None:
        # --- wave-1 absorber path (no CPML memory variables) ---
        d1, d2 = c.damp["hx"]
        Hx = d1 * state.Hx - d2 * (_d(state.Ez, 1) - _d(state.Ey, 2)) / dx
        d1, d2 = c.damp["hy"]
        Hy = d1 * state.Hy - d2 * (_d(state.Ex, 2) - _d(state.Ez, 0)) / dx
        d1, d2 = c.damp["hz"]
        Hz = d1 * state.Hz - d2 * (_d(state.Ey, 0) - _d(state.Ex, 1)) / dx
        eps_x = _avg(c.eps, 0)
        curl_x = _pad_interior(
            (_d(Hz, 1)[:, :, 1:-1] - _d(Hy, 2)[:, 1:-1, :]) / dx, (1, 2))
        d1, d2 = c.damp["ex"]
        Ex = (_e_pole(state.Ex, state.Px_cur, pn["x"], d1, d2, eps_x, curl_x)
              if pols is not None else
              (d1 * state.Ex * eps_x + d2 * curl_x) / eps_x)
        eps_y = _avg(c.eps, 1)
        curl_y = _pad_interior(
            (_d(Hx, 2)[1:-1, :, :] - _d(Hz, 0)[:, :, 1:-1]) / dx, (0, 2))
        d1, d2 = c.damp["ey"]
        Ey = (_e_pole(state.Ey, state.Py_cur, pn["y"], d1, d2, eps_y, curl_y)
              if pols is not None else
              (d1 * state.Ey * eps_y + d2 * curl_y) / eps_y)
        eps_z = _avg(c.eps, 2)
        curl_z = _pad_interior(
            (_d(Hy, 0)[:, 1:-1, :] - _d(Hx, 1)[1:-1, :, :]) / dx, (0, 1))
        d1, d2 = c.damp["ez"]
        Ez = (_e_pole(state.Ez, state.Pz_cur, pn["z"], d1, d2, eps_z, curl_z)
              if pols is not None else
              (d1 * state.Ez * eps_z + d2 * curl_z) / eps_z)
        return _inject_src(State3D(Ex, Ey, Ez, Hx, Hy, Hz, **(pout or {})),
                           c, src_amp, src, eps_x, eps_y, eps_z)

    # --- CPML path ---
    bHx, aHx, kHx = cp["hx"]; bHy, aHy, kHy = cp["hy"]; bHz, aHz, kHz = cp["hz"]
    bEx, aEx, kEx = cp["ex"]; bEy, aEy, kEy = cp["ey"]; bEz, aEz, kEz = cp["ez"]

    # H update (dB/dt = -curl E); each derivative gets its axis CPML.
    dEz_dy = _d(state.Ez, 1) / dx
    psi_Ez_y = bHy * state.psi_Ez_y + aHy * dEz_dy
    ce_y = dEz_dy / kHy + psi_Ez_y
    dEy_dz = _d(state.Ey, 2) / dx
    psi_Ey_z = bHz * state.psi_Ey_z + aHz * dEy_dz
    ce_z = dEy_dz / kHz + psi_Ey_z
    d1, d2 = c.damp["hx"]
    Hx = d1 * state.Hx - d2 * (ce_y - ce_z)

    dEx_dz = _d(state.Ex, 2) / dx
    psi_Ex_z = bHz * state.psi_Ex_z + aHz * dEx_dz
    ce_z2 = dEx_dz / kHz + psi_Ex_z
    dEz_dx = _d(state.Ez, 0) / dx
    psi_Ez_x = bHx * state.psi_Ez_x + aHx * dEz_dx
    ce_x = dEz_dx / kHx + psi_Ez_x
    d1, d2 = c.damp["hy"]
    Hy = d1 * state.Hy - d2 * (ce_z2 - ce_x)

    dEy_dx = _d(state.Ey, 0) / dx
    psi_Ey_x = bHx * state.psi_Ey_x + aHx * dEy_dx
    ce_x2 = dEy_dx / kHx + psi_Ey_x
    dEx_dy = _d(state.Ex, 1) / dx
    psi_Ex_y = bHy * state.psi_Ex_y + aHy * dEx_dy
    ce_y2 = dEx_dy / kHy + psi_Ex_y
    d1, d2 = c.damp["hz"]
    Hz = d1 * state.Hz - d2 * (ce_x2 - ce_y2)

    # E update (dD/dt = curl H); PEC outer edge via _pad_interior, CPML on
    # the in-plane derivatives.
    eps_x = _avg(c.eps, 0)
    dHz_dy = _d(Hz, 1) / dx
    psi_Hz_y = bEy * state.psi_Hz_y + aEy * dHz_dy
    ch_y = dHz_dy / kEy + psi_Hz_y
    dHy_dz = _d(Hy, 2) / dx
    psi_Hy_z = bEz * state.psi_Hy_z + aEz * dHy_dz
    ch_z = dHy_dz / kEz + psi_Hy_z
    curl_x = _pad_interior(ch_y[:, :, 1:-1] - ch_z[:, 1:-1, :], (1, 2))
    d1, d2 = c.damp["ex"]
    Ex = (_e_pole(state.Ex, state.Px_cur, pn["x"], d1, d2, eps_x, curl_x)
          if pols is not None else
          (d1 * state.Ex * eps_x + d2 * curl_x) / eps_x)

    eps_y = _avg(c.eps, 1)
    dHx_dz = _d(Hx, 2) / dx
    psi_Hx_z = bEz * state.psi_Hx_z + aEz * dHx_dz
    ch_z2 = dHx_dz / kEz + psi_Hx_z
    dHz_dx = _d(Hz, 0) / dx
    psi_Hz_x = bEx * state.psi_Hz_x + aEx * dHz_dx
    ch_x = dHz_dx / kEx + psi_Hz_x
    curl_y = _pad_interior(ch_z2[1:-1, :, :] - ch_x[:, :, 1:-1], (0, 2))
    d1, d2 = c.damp["ey"]
    Ey = (_e_pole(state.Ey, state.Py_cur, pn["y"], d1, d2, eps_y, curl_y)
          if pols is not None else
          (d1 * state.Ey * eps_y + d2 * curl_y) / eps_y)

    eps_z = _avg(c.eps, 2)
    dHy_dx = _d(Hy, 0) / dx
    psi_Hy_x = bEx * state.psi_Hy_x + aEx * dHy_dx
    ch_x2 = dHy_dx / kEx + psi_Hy_x
    dHx_dy = _d(Hx, 1) / dx
    psi_Hx_y = bEy * state.psi_Hx_y + aEy * dHx_dy
    ch_y2 = dHx_dy / kEy + psi_Hx_y
    curl_z = _pad_interior(ch_x2[:, 1:-1, :] - ch_y2[1:-1, :, :], (0, 1))
    d1, d2 = c.damp["ez"]
    Ez = (d1 * state.Ez * eps_z + d2 * curl_z) / eps_z

    new = State3D(
        Ex, Ey, Ez, Hx, Hy, Hz,
        psi_Ez_y, psi_Ey_z, psi_Ex_z, psi_Ez_x, psi_Ey_x, psi_Ex_y,
        psi_Hz_y, psi_Hy_z, psi_Hx_z, psi_Hz_x, psi_Hy_x, psi_Hx_y,
    )
    return _inject_src(new, c, src_amp, src, eps_x, eps_y, eps_z)


def _inject_src(state, c, src_amp, src, eps_x, eps_y, eps_z):
    if src is None or src_amp == 0.0:
        return state
    comp = src[0]
    if isinstance(src[1], torch.Tensor):           # area/plane soft source
        mask = src[1]
        if comp == "Ez":
            Ez = state.Ez + c.dt * src_amp * mask / eps_z
            return state._replace(Ez=Ez)
        if comp == "Ex":
            Ex = state.Ex + c.dt * src_amp * mask / eps_x
            return state._replace(Ex=Ex)
        Ey = state.Ey + c.dt * src_amp * mask / eps_y
        return state._replace(Ey=Ey)
    if len(src) == 3:                              # plane-indexed (comp, ix,
        ix, plane = src[1], src[2]                 # plane) -- lean-stepper
        if comp == "Ez":                           # form, no full-volume mask
            Ez = state.Ez.clone()
            Ez[ix] = Ez[ix] + c.dt * src_amp * plane / eps_z[ix]
            return state._replace(Ez=Ez)
        if comp == "Ex":
            Ex = state.Ex.clone()
            Ex[ix] = Ex[ix] + c.dt * src_amp * plane / eps_x[ix]
            return state._replace(Ex=Ex)
        Ey = state.Ey.clone()
        Ey[ix] = Ey[ix] + c.dt * src_amp * plane / eps_y[ix]
        return state._replace(Ey=Ey)
    i, j, k = src[1], src[2], src[3]               # soft point source
    if comp == "Ez":
        state.Ez[i, j, k] = state.Ez[i, j, k] + c.dt * src_amp / eps_z[i, j, k]
    elif comp == "Ex":
        state.Ex[i, j, k] = state.Ex[i, j, k] + c.dt * src_amp / eps_x[i, j, k]
    else:
        state.Ey[i, j, k] = state.Ey[i, j, k] + c.dt * src_amp / eps_y[i, j, k]
    return state