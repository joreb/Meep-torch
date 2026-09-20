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
from .pml import absorber_sigma, cpml_profiles, side_counts
from .symbolic import (PSI_NEW, STRETCHED,
                       compiled_curl_composition)

#: meep's grouping as `stepper_tm` and `stepper_te` write it:
#: `((d0/k0 + psi0) - d1/k1) - psi1`. This file used to write
#: `(d0/k0 + psi0) - (d1/k1 + psi1)` instead -- the same algebra, a
#: different association, and 1.4e-14 apart. Neither grouping is meep's
#: (meep has no CPML at all), so the tie-break is baselines: tm/te carry
#: the solve_cw tolerance history and core.factory is bitwise through
#: this composition on TM and on all three TE components.
_COMPOSE = compiled_curl_composition(n_terms=2)


class Coeffs3D(NamedTuple):
    dt: float
    dx: float
    eps: torch.Tensor        # (nx, ny, nz) cell-centered
    damp: dict              # per-grid (d1, d2): material+absorber conductivity
    cpml: dict | None       # per-axis (b, a, k) profiles, None in absorber mode
    pols: dict | None = None  # "x"/"y"/"z" -> (c1, c2, c3) ADE pole stacks
    pmc: tuple = ()          # axes with PMC walls instead of PEC (see _term_pmc)


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
    # The six E-update memories are allocated at the extent step_3d writes,
    # which is smaller than the raw derivative and depends on the PMC axes --
    # see _psi_e_shapes. Ex <- dHz/dy, dHy/dz; Ey <- dHx/dz, dHz/dx;
    # Ez <- dHy/dx, dHx/dy.
    psi_Hz_y: torch.Tensor | None = None
    psi_Hy_z: torch.Tensor | None = None
    psi_Hx_z: torch.Tensor | None = None
    psi_Hz_x: torch.Tensor | None = None
    psi_Hy_x: torch.Tensor | None = None
    psi_Hx_y: torch.Tensor | None = None
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


def _pad_interior(t, axes, pmc=()):
    """Zero-pad one cell on both sides of each axis in ``axes`` (PEC outer
    edge for the tangential E components). Axes in ``pmc`` are skipped --
    ``_term_pmc`` already delivered them full-length."""
    pad = [0, 0, 0, 0, 0, 0]
    for a in axes:
        if a in pmc:
            continue
        pad[2 * (2 - a)] = 1
        pad[2 * (2 - a) + 1] = 1
    return torch.nn.functional.pad(t, pad)


def _trim_full(t, padaxes, fullax, pmc):
    """Interior trim of an E-update curl term along its full-length axis.
    The written region of a tangential E component stops one cell short of
    each PEC wall; a PMC wall keeps it alive, so a pmc axis is not trimmed.
    Applied to the raw derivative, BEFORE the psi recursion, so the CPML
    memory is allocated at the extent the step actually writes. The trim is
    always on an axis the recursion's coefficients are uniform along (they
    vary along the derivative axis), so it commutes with the recursion
    bitwise."""
    sl = [slice(None)] * 3
    for a in padaxes:
        if a == fullax and a not in pmc:
            sl[a] = slice(1, -1)
    return t[tuple(sl)]


def _pad_pmc_kappa(t, padaxes, fullax, pmc):
    """:func:`_pad_pmc` for a kappa profile, which pads with ONE rather than zero.

    The E-update terms are composed as ``s0 - d/kappa - psi`` so that this file
    groups its curls the way `stepper_tm` and `stepper_te` do. That needs the
    derivative, its kappa and its psi padded to a common shape instead of the
    stretched value being padded whole.

    Zero is the right pad for ``d`` and ``psi`` for the reason `_pad_pmc`
    already gives -- the derivative along a pmc axis is identically zero for the
    transverse-uniform fields this pad exists for. The matching pad for kappa is
    ONE, so the padded region contributes ``0/1 + 0 = 0``: exactly what the
    zero-padded stretched value contributed before. Padding kappa with zero
    instead would divide by it.

    Each kappa is 1-D along its own derivative axis and broadcast elsewhere, so
    padding it on the pmc axis keeps it aligned with the padded derivative.
    """
    pad = [0, 0, 0, 0, 0, 0]
    for a in padaxes:
        if a != fullax and a in pmc:
            pad[2 * (2 - a)] = pad[2 * (2 - a) + 1] = 1
    return (torch.nn.functional.pad(t, pad, value=1.0) if any(pad) else t)


def _pad_pmc(t, padaxes, fullax, pmc):
    """PMC wall handling for one E-update curl term (2D ``y_pmc`` mirror:
    tangential E live at the wall, H=0 ghost beyond it). The reduced-axis
    term -- the derivative along the pmc axis -- is zero-padded out to the
    wall; that pad is exact for transverse-uniform fields, whose derivative
    along the pmc axis is identically zero. Non-pmc axes are left as-is for
    ``_pad_interior``. With ``pmc=()`` this is the identity."""
    pad = [0, 0, 0, 0, 0, 0]
    for a in padaxes:
        if a != fullax and a in pmc:
            pad[2 * (2 - a)] = pad[2 * (2 - a) + 1] = 1
    return torch.nn.functional.pad(t, pad) if any(pad) else t


def _term_pmc(t, padaxes, fullax, pmc):
    """Trim then pad, for the absorber path where there is no psi recursion
    to sit between the two halves."""
    return _pad_pmc(_trim_full(t, padaxes, fullax, pmc), padaxes, fullax, pmc)


def _psi_e_shapes(nx, ny, nz, pmc=()):
    """Shapes of the six E-update CPML memories: the source H component's
    grid, one shorter on the derivative axis and two shorter on the axis the
    interior trim removes -- what ``step_3d`` writes, nothing more. Before
    this was derived the arrays carried the untrimmed extent and the psi
    recursion ran on two slabs that ``_trim_full`` then discarded."""
    H = {"x": (nx, ny - 1, nz - 1), "y": (nx - 1, ny, nz - 1),
         "z": (nx - 1, ny - 1, nz)}
    #  name          source H, derivative axis, full axis
    spec = (("psi_Hz_y", "z", 1, 2), ("psi_Hy_z", "y", 2, 1),
            ("psi_Hx_z", "x", 2, 0), ("psi_Hz_x", "z", 0, 2),
            ("psi_Hy_x", "y", 0, 1), ("psi_Hx_y", "x", 1, 0))
    out = {}
    for name, comp, dax, fax in spec:
        s = list(H[comp])
        s[dax] -= 1
        if fax not in pmc:
            s[fax] -= 2
        out[name] = tuple(s)
    return out


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


def _axis_sides3(v):
    """Per-axis (lo, hi) cell counts: an int means the same layer on all
    six faces, a 3-sequence of int-or-(lo, hi) sets each axis (0 = PEC)."""
    if isinstance(v, (tuple, list)):
        return tuple(side_counts(p) for p in v)
    return (side_counts(v),) * 3


def build_3d_coeffs(nx: int, ny: int, nz: int, dx: float, eps=None,
                    npml=0, n_abs=0, absorber_axes=(),
                    sigma_d=None, sigma_b=None, poles=(), pole_masks=None,
                    dtype=torch.float64, device="cpu", absorber_m: int = 2,
                    absorber_R0: float = 1e-15, pmc_axes=()) -> Coeffs3D:
    """Boundary modes: full CPML (``npml>0``, ``n_abs=0``), full matched-sigma
    absorber (``npml=0``, ``n_abs>0``), or mixed (``npml>0`` and ``n_abs>0``):
    CPML on the axes not in ``absorber_axes``, matched-sigma absorber on the
    axes in ``absorber_axes`` (via an identity CPML profile -- plain derivative
    -- plus the absorber sigma in ``sig_bound``). Mixed mode isolates one
    boundary for a clean CPML-vs-absorber comparison (e.g. a z-uniform
    planewave with ``absorber_axes=(2,)`` sees only the z-boundary; the x/y
    CPML is inert for the uniform transverse field). ``pmc_axes`` swaps the
    PEC walls on those axes for PMC (tangential E live at the wall, identity
    CPML profile -- see ``_term_pmc``), keeping a transverse-uniform
    planewave exactly uniform on a tiny transverse grid.

    ``npml`` / ``n_abs``: an int puts the same layer on every face; a
    3-sequence of int-or-(lo, hi) pairs sets each axis end independently
    (0 = plain PEC wall there, mp.PML direction/side semantics)."""
    dt = 0.5 * dx / math.sqrt(3.0)
    e = (eps if isinstance(eps, torch.Tensor)
         else torch.ones(nx, ny, nz, dtype=dtype, device=device))
    dims = (nx, ny, nz)
    npml_ax = _axis_sides3(npml)
    nabs_ax = _axis_sides3(n_abs)
    has_pml = any(l or h for l, h in npml_ax)
    has_abs = any(l or h for l, h in nabs_ax)
    mixed = has_pml and has_abs
    ax_set = set(absorber_axes) if mixed else set()

    # --- boundary profiles / sigma ---
    if has_pml:
        cpml = {}
        for ax, kh, ke in ((0, "hx", "ex"), (1, "hy", "ey"), (2, "hz", "ez")):
            n = dims[ax]
            if ax in ax_set or ax in pmc_axes:     # absorber/PMC axis: plain derivative
                bH, aH, kH = _ident(n - 1, ax, dtype, device)
                bE, aE, kE = _ident(n - 2, ax, dtype, device)
            else:                                  # CPML axis
                b, a, k = _prof(n - 1, npml_ax[ax], 0.5, dt, dx, dtype, device)
                bH, aH, kH = _bc(b, ax), _bc(a, ax), _bc(k, ax)
                b, a, k = _prof(n, npml_ax[ax], 0.0, dt, dx, dtype, device)
                bE, aE, kE = _bc(b[1:-1], ax), _bc(a[1:-1], ax), _bc(k[1:-1], ax)
            cpml[kh] = (bH, aH, kH)
            cpml[ke] = (bE, aE, kE)
        sig_bound = torch.zeros(nx, ny, nz, dtype=dtype, device=device)
        if mixed:                                  # absorber sigma on absorber axes
            for ax in absorber_axes:
                s = absorber_sigma(dims[ax], nabs_ax[ax], dx, m=absorber_m,
                                   R0=absorber_R0, dtype=dtype, device=device)
                sig_bound = torch.maximum(sig_bound, _bc(s, ax).expand(nx, ny, nz))
    else:
        cpml = None
        if has_abs:
            sx = absorber_sigma(nx, nabs_ax[0], dx, m=absorber_m, R0=absorber_R0,
                                dtype=dtype, device=device)
            sy = absorber_sigma(ny, nabs_ax[1], dx, m=absorber_m, R0=absorber_R0,
                                dtype=dtype, device=device)
            sz = absorber_sigma(nz, nabs_ax[2], dx, m=absorber_m, R0=absorber_R0,
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
    return Coeffs3D(dt=dt, dx=dx, eps=e, damp=damp, cpml=cpml, pols=pols,
                    pmc=tuple(pmc_axes))


def init_3d_state(nx: int, ny: int, nz: int, npml: int = 0, n_poles: int = 0,
                  dtype=torch.float64, device="cpu", pmc=()) -> State3D:
    """``pmc`` must match ``build_3d_coeffs``'s ``pmc_axes``: a PMC wall keeps
    the tangential E alive, which widens the E-update CPML memories by one
    cell at each end of that axis (see ``_psi_e_shapes``)."""
    z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)
    psi = {}
    if npml > 0:
        psi = dict(
            psi_Ez_y=z(nx, ny - 1, nz - 1), psi_Ey_z=z(nx, ny - 1, nz - 1),
            psi_Ex_z=z(nx - 1, ny, nz - 1), psi_Ez_x=z(nx - 1, ny, nz - 1),
            psi_Ey_x=z(nx - 1, ny - 1, nz), psi_Ex_y=z(nx - 1, ny - 1, nz),
            **{k: z(*shape) for k, shape in _psi_e_shapes(nx, ny, nz, pmc).items()},
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
    """One leapfrog step. ``src`` is ``("Ez"|"Ex"|"Ey", i, j, k)`` for a soft
    point current, ``("Ez"|"Ex"|"Ey", mask)`` for an area/plane soft current
    (mask matches the component's node shape), or ``(comp, ix, plane)`` for the
    plane-indexed form the lean stepper uses, which injects into one x slice
    without materializing a full-volume mask."""
    dx = c.dx
    cp = c.cpml
    pmc = c.pmc

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
            (_term_pmc(_d(Hz, 1), (1, 2), 2, pmc)
             - _term_pmc(_d(Hy, 2), (1, 2), 1, pmc)) / dx, (1, 2), pmc)
        d1, d2 = c.damp["ex"]
        Ex = (_e_pole(state.Ex, state.Px_cur, pn["x"], d1, d2, eps_x, curl_x)
              if pols is not None else
              (d1 * state.Ex * eps_x + d2 * curl_x) / eps_x)
        eps_y = _avg(c.eps, 1)
        curl_y = _pad_interior(
            (_term_pmc(_d(Hx, 2), (0, 2), 0, pmc)
             - _term_pmc(_d(Hz, 0), (0, 2), 2, pmc)) / dx, (0, 2), pmc)
        d1, d2 = c.damp["ey"]
        Ey = (_e_pole(state.Ey, state.Py_cur, pn["y"], d1, d2, eps_y, curl_y)
              if pols is not None else
              (d1 * state.Ey * eps_y + d2 * curl_y) / eps_y)
        eps_z = _avg(c.eps, 2)
        curl_z = _pad_interior(
            (_term_pmc(_d(Hy, 0), (0, 1), 1, pmc)
             - _term_pmc(_d(Hx, 1), (0, 1), 0, pmc)) / dx, (0, 1), pmc)
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
    psi_Ez_y = PSI_NEW(dEz_dy, state.psi_Ez_y, bHy, aHy)
    ce_y = STRETCHED(dEz_dy, psi_Ez_y, kHy)
    dEy_dz = _d(state.Ey, 2) / dx
    psi_Ey_z = PSI_NEW(dEy_dz, state.psi_Ey_z, bHz, aHz)
    d1, d2 = c.damp["hx"]
    Hx = d1 * state.Hx - d2 * _COMPOSE(ce_y, dEy_dz, kHz, psi_Ey_z)

    dEx_dz = _d(state.Ex, 2) / dx
    psi_Ex_z = PSI_NEW(dEx_dz, state.psi_Ex_z, bHz, aHz)
    ce_z2 = STRETCHED(dEx_dz, psi_Ex_z, kHz)
    dEz_dx = _d(state.Ez, 0) / dx
    psi_Ez_x = PSI_NEW(dEz_dx, state.psi_Ez_x, bHx, aHx)
    d1, d2 = c.damp["hy"]
    Hy = d1 * state.Hy - d2 * _COMPOSE(ce_z2, dEz_dx, kHx, psi_Ez_x)

    dEy_dx = _d(state.Ey, 0) / dx
    psi_Ey_x = PSI_NEW(dEy_dx, state.psi_Ey_x, bHx, aHx)
    ce_x2 = STRETCHED(dEy_dx, psi_Ey_x, kHx)
    dEx_dy = _d(state.Ex, 1) / dx
    psi_Ex_y = PSI_NEW(dEx_dy, state.psi_Ex_y, bHy, aHy)
    d1, d2 = c.damp["hz"]
    Hz = d1 * state.Hz - d2 * _COMPOSE(ce_x2, dEx_dy, kHy, psi_Ex_y)

    # E update (dD/dt = curl H); PEC outer edge via _pad_interior, CPML on
    # the in-plane derivatives.
    eps_x = _avg(c.eps, 0)
    dHz_dy = _trim_full(_d(Hz, 1), (1, 2), 2, pmc) / dx
    psi_Hz_y = PSI_NEW(dHz_dy, state.psi_Hz_y, bEy, aEy)
    ch_y = STRETCHED(dHz_dy, psi_Hz_y, kEy)
    dHy_dz = _trim_full(_d(Hy, 2), (1, 2), 1, pmc) / dx
    psi_Hy_z = PSI_NEW(dHy_dz, state.psi_Hy_z, bEz, aEz)
    curl_x = _pad_interior(
        _COMPOSE(_pad_pmc(ch_y, (1, 2), 2, pmc),
                 _pad_pmc(dHy_dz, (1, 2), 1, pmc),
                 _pad_pmc_kappa(kEz, (1, 2), 1, pmc),
                 _pad_pmc(psi_Hy_z, (1, 2), 1, pmc)), (1, 2), pmc)
    d1, d2 = c.damp["ex"]
    Ex = (_e_pole(state.Ex, state.Px_cur, pn["x"], d1, d2, eps_x, curl_x)
          if pols is not None else
          (d1 * state.Ex * eps_x + d2 * curl_x) / eps_x)

    eps_y = _avg(c.eps, 1)
    dHx_dz = _trim_full(_d(Hx, 2), (0, 2), 0, pmc) / dx
    psi_Hx_z = PSI_NEW(dHx_dz, state.psi_Hx_z, bEz, aEz)
    ch_z2 = STRETCHED(dHx_dz, psi_Hx_z, kEz)
    dHz_dx = _trim_full(_d(Hz, 0), (0, 2), 2, pmc) / dx
    psi_Hz_x = PSI_NEW(dHz_dx, state.psi_Hz_x, bEx, aEx)
    curl_y = _pad_interior(
        _COMPOSE(_pad_pmc(ch_z2, (0, 2), 0, pmc),
                 _pad_pmc(dHz_dx, (0, 2), 2, pmc),
                 _pad_pmc_kappa(kEx, (0, 2), 2, pmc),
                 _pad_pmc(psi_Hz_x, (0, 2), 2, pmc)), (0, 2), pmc)
    d1, d2 = c.damp["ey"]
    Ey = (_e_pole(state.Ey, state.Py_cur, pn["y"], d1, d2, eps_y, curl_y)
          if pols is not None else
          (d1 * state.Ey * eps_y + d2 * curl_y) / eps_y)

    eps_z = _avg(c.eps, 2)
    dHy_dx = _trim_full(_d(Hy, 0), (0, 1), 1, pmc) / dx
    psi_Hy_x = PSI_NEW(dHy_dx, state.psi_Hy_x, bEx, aEx)
    ch_x2 = STRETCHED(dHy_dx, psi_Hy_x, kEx)
    dHx_dy = _trim_full(_d(Hx, 1), (0, 1), 0, pmc) / dx
    psi_Hx_y = PSI_NEW(dHx_dy, state.psi_Hx_y, bEy, aEy)
    curl_z = _pad_interior(
        _COMPOSE(_pad_pmc(ch_x2, (0, 1), 1, pmc),
                 _pad_pmc(dHx_dy, (0, 1), 0, pmc),
                 _pad_pmc_kappa(kEy, (0, 1), 0, pmc),
                 _pad_pmc(psi_Hx_y, (0, 1), 0, pmc)), (0, 1), pmc)
    d1, d2 = c.damp["ez"]
    Ez = (_e_pole(state.Ez, state.Pz_cur, pn["z"], d1, d2, eps_z, curl_z)
          if pols is not None else
          (d1 * state.Ez * eps_z + d2 * curl_z) / eps_z)

    new = State3D(
        Ex, Ey, Ez, Hx, Hy, Hz,
        psi_Ez_y, psi_Ey_z, psi_Ex_z, psi_Ez_x, psi_Ey_x, psi_Ex_y,
        psi_Hz_y, psi_Hy_z, psi_Hx_z, psi_Hz_x, psi_Hy_x, psi_Hx_y,
        **(pout or {}),
    )
    return _inject_src(new, c, src_amp, src, eps_x, eps_y, eps_z)


def _inject_src(state, c, src_amp, src, eps_x, eps_y, eps_z):
    """meep's ``step_source``, in the three shapes this stepper accepts.

    THE SIGN IS DERIVED, not chosen. ``symbolic.source_coefficient`` reads it
    off Ampere's law -- ``dD/dt = curl H - J`` gives ``-1/eps`` -- and it is
    meep's single ``f[c][i] -= real(A)`` in `fields_chunk::step_source`. Two of
    these three branches used to ADD, which is a dipole radiating with the
    opposite phase: it runs, it converges, and the only gate that can see it is
    the LDOS, which comes out negative.

    THE CELL MEASURE IS DERIVED TOO, from the source's extent. A current
    density concentrated to zero extent along an axis carries one factor of
    ``1/dx`` for that axis, so a point (three delta directions) carries
    ``1/dx^3`` and an x-indexed plane (one) carries ``1/dx``. meep applies the
    identical rule one factor at a time, ``data.amp *= gv.a`` per zero-extent
    direction (`src/sources.cpp`), and `sources.LineSource` derives the 2D case
    the same way. Without it the field a dipole produces moves with resolution.

    The MASK form is left alone deliberately: an array cannot say which of its
    axes the source is a delta in -- `test_w9_wave2` passes a z-slab of ones,
    which is a plane, while another caller may pass a filled volume -- so the
    measure stays with the caller who knows the extent.

    The indexed forms write IN PLACE, which is safe for a reason that is worth
    stating rather than rediscovering: both call sites hand this function a
    `State3D` whose E components were just built by the update, so the tensor
    being written is never the one the caller still holds. Cloning it would
    cost a full-field allocation every step to guard a hazard the call sites
    already rule out -- and `test_a_3d_step_does_not_mutate_its_input` is what
    keeps that true.
    """
    if src is None or src_amp == 0.0:
        return state
    comp = src[0]
    eps = {"Ex": eps_x, "Ey": eps_y, "Ez": eps_z}[comp]
    if isinstance(src[1], torch.Tensor):           # area/plane soft source
        mask = src[1]                              # caller owns the measure
        f = getattr(state, comp) - c.dt * src_amp * mask / eps
        return state._replace(**{comp: f})
    if len(src) == 3:                              # plane-indexed (comp, ix,
        ix, plane = src[1], src[2]                 # plane), delta in x only
        f = getattr(state, comp)
        f[ix] = f[ix] - c.dt * src_amp * plane / (c.dx * eps[ix])
        return state._replace(**{comp: f})
    i, j, k = src[1], src[2], src[3]               # soft point, delta in three
    f = getattr(state, comp)                       # freshly built above, so
    f[i, j, k] = (f[i, j, k]                       # writing it is not the
                  - c.dt * src_amp / (c.dx ** 3 * eps[i, j, k]))
    return state._replace(**{comp: f})
