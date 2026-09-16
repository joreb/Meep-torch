# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .coeffs import TMCoeffs
from .grid import Grid
import sympy as sp
from .symbolic import (PSI_NEW, STRETCHED, compiled_curl_composition,
                       compiled_difference)

# Yee differences derived in symbolic.py rather than written here. Which of
# these is a forward and which a backward difference is a consequence of the
# two components' half-cell offsets, not a choice -- see that module. The
# compiled form is over a common denominator, so it is the same float
# operation meep performs: one division of a difference, not two divisions.
_dEz_dy = compiled_difference("Ez", "y", "Hx")   # forward
_dEz_dx = compiled_difference("Ez", "x", "Hy")   # forward
_dHx_dy = compiled_difference("Hx", "y", "Ez")   # backward
_dHy_dx = compiled_difference("Hy", "x", "Ez")   # backward

# The curl composition, in meep's term order. The order lives in
# core.symbolic now, shared by every stepper that needs it, instead of being a
# line of arithmetic per stepper that nothing checks.
_COMPOSE = compiled_curl_composition(n_terms=2)

class TMState(NamedTuple):
    Ez: torch.Tensor       # (nx, ny)        E_z at time n
    Hx: torch.Tensor       # (nx, ny-1)      at n-1/2
    Hy: torch.Tensor       # (nx-1, ny)      at n-1/2
    Dz: torch.Tensor       # (nx, ny)
    P_cur: torch.Tensor    # (n_poles, nx, ny)
    P_prev: torch.Tensor   # (n_poles, nx, ny)
    psi_Ezy: torch.Tensor  # (nx, ny-1)      CPML memory, dEz/dy in Hx update
    psi_Ezx: torch.Tensor  # (nx-1, ny)      dEz/dx in Hy update
    psi_Hyx: torch.Tensor  # (nx_e, ny_e): live Ez region -- interior, or up
    psi_Hxy: torch.Tensor  # to the wall under y_pmc / a mirror (init_tm_state)
    # W5.5 magnetic Lorentz poles (allocated only when n_mpoles > 0)
    Bx: torch.Tensor | None = None       # (nx, ny-1)
    By: torch.Tensor | None = None       # (nx-1, ny)
    Mx_cur: torch.Tensor | None = None   # (nm, nx, ny-1)
    Mx_prev: torch.Tensor | None = None
    My_cur: torch.Tensor | None = None   # (nm, nx-1, ny)
    My_prev: torch.Tensor | None = None
    # odd-mirror on-plane H rows (y_mirror/x_mirror == -1): the H node ON the
    # cut plane is live (driven by 2 Ez[0]/dx via the odd ghost), kept apart
    # so Hx/Hy keep their standard shapes for every consumer
    Hx0: torch.Tensor | None = None      # (nx,)  Hx at the low-y cut
    Hy0: torch.Tensor | None = None      # (ny,)  Hy at the low-x cut



def _replaced(state, **fields):
    """``state._replace(**fields)``, but traceable by ``torch.compile``.

    ``NamedTuple._replace`` raises a bare ``AssertionError`` inside dynamo
    (torch 2.9.1, ``side_effects.codegen_save_tempvars``), so every stepper
    using it was uncompilable -- which mattered more than it sounds. Measured
    on this repo: the generic axis-generic steppers compile and gain 2.3-3.8x,
    so an UNCOMPILABLE hand-written stepper is the slow path once compile is on
    the table. ``_make`` and direct construction both trace fine, and
    ``_fields`` is a static tuple so dynamo unrolls this comprehension.

    Keeps ``_replace``'s shape at the call sites: these states carry ~20
    fields, and writing every one out ten times would bury the two that change.
    """
    return type(state)._make(
        [fields.get(name, value) for name, value in zip(state._fields, state)])

def init_tm_state(grid: Grid, n_poles: int = 0, y_pmc: bool = False,
                  n_mpoles: int = 0, y_mirror: int = 0,
                  x_mirror: int = 0, y_rot2: int = 0,
                  rot4: int = 0) -> TMState:
    nx, ny = grid.nx, grid.ny
    # psi_Hxy rows: PEC interior ny-2; both-wall PMC ny; a low-y cut (mirror,
    # Rotate2 or Rotate4) ny-1. Rotate4 cuts both low edges.
    ny_e = ny if y_pmc else (ny - 1 if (y_mirror or y_rot2 or rot4) else ny - 2)
    nx_e = nx - 1 if (x_mirror or rot4) else nx - 2
    mag = {}
    if n_mpoles:
        mag = dict(
            Bx=grid.zeros(nx, ny - 1), By=grid.zeros(nx - 1, ny),
            Mx_cur=grid.zeros(n_mpoles, nx, ny - 1),
            Mx_prev=grid.zeros(n_mpoles, nx, ny - 1),
            My_cur=grid.zeros(n_mpoles, nx - 1, ny),
            My_prev=grid.zeros(n_mpoles, nx - 1, ny),
        )
    return TMState(
        Ez=grid.zeros(nx, ny),
        Hx=grid.zeros(nx, ny - 1),
        Hy=grid.zeros(nx - 1, ny),
        Dz=grid.zeros(nx, ny),
        P_cur=grid.zeros(n_poles, nx, ny),
        P_prev=grid.zeros(n_poles, nx, ny),
        psi_Ezy=grid.zeros(nx, ny - 1),
        psi_Ezx=grid.zeros(nx - 1, ny),
        psi_Hyx=grid.zeros(nx_e, ny_e),
        psi_Hxy=grid.zeros(nx_e, ny_e),
        # the C2 fold keeps the on-plane Hx row live for BOTH parities: the
        # cut row maps onto itself reversed in x, it is not forced to zero
        Hx0=grid.zeros(nx) if (y_mirror == -1 or y_rot2 or rot4) else None,
        Hy0=grid.zeros(ny) if (x_mirror == -1 or rot4) else None,
        **mag,
    )


# --- the cut-plane ghost, from the symmetry rather than per case -------------
#
# A fold confines the solution to an eigenspace of the symmetry operator:
# R f = lam f. That relation holds POINTWISE at every node, ghost nodes
# included, and it can be read in two equivalent directions,
#
#     f(x) = conj(lam) M   f(P^-1 x)          [pre-image form]
#     f(x) = lam       M^T f(P   x)          [image form]
#
# so the ghost just beyond a cut is whichever of the two reaches a node inside
# the kept domain. For an INVOLUTION (mirror, C2) P = P^-1 and lam = conj(lam),
# the two coincide, and the ghost is the cut row itself times the phase. C4 is
# not an involution, so the two low edges take OPPOSITE powers of P: the low-y
# ghost reads the first column and the low-x ghost the first row, which is the
# "ghost comes from the other cut line, transposed" fact -- forced, not extra.
#
# M drops out here: M_E = 1 for Ez under every in-plane rotation and both
# in-plane mirrors, because they all fix the z axis (component_transform
# confirms it). Only the phase and the gather survive.
#
# The GATHER stays index algebra deliberately. "The C2 partner is this row
# reversed in x" and C4's transposed column are tensor index operations, not
# expressions sympy carries. What the derivation supplies is which gather and
# which phase power -- and the collapse those cases tempt you into
# (2*Ez0/h odd, 0 even, Ez0*(1-p)/h for C2) is a special case that holds only
# where the flip maps an element onto itself, not the general form.
#
# NOTE ON THE PHASE. The two edges carry lam and conj(lam). They agree only
# because the engine restricts to lam in {+1, -1}: the C4 irreps are
# {1, i, -1, -i}, and at +-i a single phase on both edges would be wrong. The
# restriction is real (a real field is never in one circular sector alone, so
# +-i buys nothing over a C2-odd half fold), but the shortcut below depends on
# it, so it is named rather than left load-bearing and silent.
_CUT_GATHER = {
    # (low-y ghost, low-x ghost), each as a function of the folded Ez
    "mirror": (lambda Ez: Ez[:, 0], lambda Ez: Ez[0, :]),          # P = P^-1
    "rot2": (lambda Ez: Ez[:, 0].flip(0), None),                   # C2, y cut
    "rot4": (lambda Ez: Ez[0, :], lambda Ez: Ez[:, 0]),            # P, then P^-1
}


def _cut_ghost(m: TMCoeffs, axis: str):
    """``(phase, gather)`` for the ghost beyond the low-``axis`` cut plane."""
    if m.rot4_lo:
        kind, phase = "rot4", m.rot4_lo
    elif axis == "y" and m.y_rot2_lo:
        kind, phase = "rot2", m.y_rot2_lo
    else:
        # the only remaining live cut-plane node is an ODD mirror; the even
        # one has no ghost at all (H = 0 is padded in, see step_db's D phase)
        kind, phase = "mirror", -1
    gather = _CUT_GATHER[kind][0 if axis == "y" else 1]
    assert gather is not None, f"{kind} has no low-{axis} cut plane"
    return phase, gather


# --- meep fields::step phases (src/step.cpp) ---------------------------------
# step_tm is split at meep's phase boundaries; the math lines are the original
# fused kernel's, only moved. Each phase maps TMState -> TMState.

def step_db(state: TMState, m: TMCoeffs, ft: str) -> TMState:
    """meep step_db(ft): ft="B" advances B from curl E, ft="D" advances D from
    curl H. CPML recursive-convolution psi updates and sigma damping live
    here (meep applies its sigma in step_db). Without magnetic poles B = mu H
    is folded, so the "B" phase writes H directly (mu in h2*; update_eh(H) is
    then the identity)."""
    dt, dx = m.dt, m.dx
    if ft == "B":
        # --- B/H update (from Ez at time n) ---
        dEz_dy = _dEz_dy(state.Ez[:, :-1], state.Ez[:, 1:], dx)   # (nx, ny-1)
        psi_Ezy = PSI_NEW(dEz_dy, state.psi_Ezy, m.b_hy, m.a_hy)
        curl_hx = STRETCHED(dEz_dy, psi_Ezy, m.k_hy)
        dEz_dx = _dEz_dx(state.Ez[:-1, :], state.Ez[1:, :], dx)   # (nx-1, ny)
        psi_Ezx = PSI_NEW(dEz_dx, state.psi_Ezx, m.b_hx, m.a_hx)
        curl_hy = STRETCHED(dEz_dx, psi_Ezx, m.k_hx)
        # odd-mirror on-plane H nodes: odd ghost Ez[-1] = -Ez[0] across the
        # cut -> dEz/dn = 2 Ez[0]/dx there (no PML at the cut, no psi)
        mir = {}
        # ghost Ez just beyond each cut, then dEz/dn = (Ez_cut - ghost)/dx.
        #   odd mirror  ghost = -Ez_cut                 -> 2 Ez_cut/dx
        #   Rotate2     ghost = phase * Ez_cut reversed in x (the C2 partner
        #               of the cut row is the cut row itself)
        #   Rotate4     ghost = phase * the OTHER cut's line, transposed: C4
        #               carries the y cut onto the x cut, so the row below
        #               the y cut is the first column and vice versa
        if state.Hx0 is not None:
            phase, gather = _cut_ghost(m, "y")
            mir["Hx0"] = state.Hx0 - dt * _dEz_dy(
                phase * gather(state.Ez), state.Ez[:, 0], dx)
        if state.Hy0 is not None:
            phase, gather = _cut_ghost(m, "x")
            mir["Hy0"] = state.Hy0 + dt * _dEz_dx(
                phase * gather(state.Ez), state.Ez[0, :], dx)
        if m.mc1x is not None:
            # W5.5 mu-Lorentz: B integrates the curl; update_eh(H) forms H = B - sum M
            return _replaced(state, psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                                  Bx=state.Bx - dt * curl_hx,
                                  By=state.By + dt * curl_hy, **mir)
        if m.h1x is not None:
            # W5: dB/dt = -curl E - sigma_b B, B = mu H (fused: mu inside h2*)
            return _replaced(state, psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                                  Hx=m.h1x * state.Hx - m.h2x * curl_hx,
                                  Hy=m.h1y * state.Hy + m.h2y * curl_hy, **mir)
        return _replaced(state, psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                              Hx=state.Hx - dt * curl_hx,
                              Hy=state.Hy + dt * curl_hy, **mir)

    # --- D update (from H at n+1/2) ---
    # Ghost/plane H beyond each wall, then difference over the live Ez
    # region: interior (PEC walls), all rows (y-PMC), or rows/cols from the
    # cut plane up (mirror: even = H=0 ghost, odd = the Hx0/Hy0 plane node).
    Hx, Hy = state.Hx, state.Hy
    if m.y_pmc:
        Hx = F.pad(Hx, (1, 1))                                    # PMC: H=0 beyond wall
        pad_y, ys = (0, 0), slice(None)
    elif m.y_mirror_lo == +1:
        Hx = F.pad(Hx, (1, 0))                                    # even: H=0 ghost at cut
        pad_y, ys = (0, 1), slice(None, -1)
    elif m.y_mirror_lo == -1 or m.y_rot2_lo or m.rot4_lo:
        Hx = torch.cat([state.Hx0[:, None], Hx], dim=1)           # odd/C2: plane node live
        pad_y, ys = (0, 1), slice(None, -1)
    else:
        pad_y, ys = (1, 1), slice(1, -1)
    if m.x_mirror_lo == +1:
        Hy = F.pad(Hy, (0, 0, 1, 0))
        pad_x, xs = (0, 1), slice(None, -1)
    elif m.x_mirror_lo == -1 or m.rot4_lo:
        Hy = torch.cat([state.Hy0[None, :], Hy], dim=0)
        pad_x, xs = (0, 1), slice(None, -1)
    else:
        pad_x, xs = (1, 1), slice(1, -1)
    dHx_dy = _dHx_dy(Hx[xs, 1:], Hx[xs, :-1], dx)
    dHy_dx = _dHy_dx(Hy[1:, ys], Hy[:-1, ys], dx)
    pad_spec = pad_y + pad_x

    psi_Hyx = PSI_NEW(dHy_dx, state.psi_Hyx, m.b_ex, m.a_ex)
    psi_Hxy = PSI_NEW(dHx_dy, state.psi_Hxy, m.b_ey, m.a_ey)
    # meep's association, (((d1/k1 + psi1) - d2/k2) - psi2), now emitted by
    # the derivation rather than typed here: compiled_curl_composition is
    # handed the term order and builds it as nested binary Adds, which is the
    # one structure that survives sympy's printer. Grouping it as
    # (d1/k1 + psi1) - (d2/k2 + psi2) is the same algebra and different floats,
    # so this is not style -- it is what makes the step bitwise with meep.
    curl = _COMPOSE(STRETCHED(dHy_dx, psi_Hyx, m.k_ex),
                    dHx_dy, m.k_ey, psi_Hxy)
    if m.d1 is not None:
        # W5: dD/dt = curl H - sigma_d D
        Dz = m.d1 * state.Dz + m.d2 * F.pad(curl, pad_spec)
    else:
        Dz = state.Dz + dt * F.pad(curl, pad_spec)
    return _replaced(state, Dz=Dz, psi_Hyx=psi_Hyx, psi_Hxy=psi_Hxy)


def step_source(state: TMState, m: TMCoeffs, ft: str, amp=0.0,
                src_mask: torch.Tensor | None = None,
                src_hy_mask: torch.Tensor | None = None,
                src_hx_mask: torch.Tensor | None = None) -> TMState:
    """meep step_source(ft): soft current injection -- electric currents into
    D (ft="D"; scaled by d2 under sigma_d, like the fused kernel), magnetic
    TFSF currents into B/H (ft="B").

    Sign: the current is SUBTRACTED. meep's step.cpp step_source does
    ``f[c][0][i] -= real(current * dt)`` for magnetic B_stuff exactly as for
    electric D_stuff, because Ampere's law is dD/dt = curl H - J. Injecting
    with + instead gives a field of the right magnitude and the wrong
    polarity, which every quadratic observable (flux, energy, |E|^2, harminv
    frequency and Q) is blind to -- so the whole test suite passed with it
    backwards until a signed field trace was compared against the oracle.
    All steppers subtract; do not "fix" one to match a stale golden."""
    dt = m.dt
    if ft == "B":
        # into B when magnetic poles allocate it (meep injects into B);
        # otherwise straight into the folded H
        Hy, Hx = (state.By, state.Bx) if state.Bx is not None else (state.Hy, state.Hx)
        if src_hy_mask is not None:
            Hy = Hy - dt * amp * src_hy_mask
        if src_hx_mask is not None:
            Hx = Hx - dt * amp * src_hx_mask
        if state.Bx is not None:
            return _replaced(state, Bx=Hx, By=Hy)
        return _replaced(state, Hx=Hx, Hy=Hy)
    if src_mask is None:
        return state
    scale = m.d2 if m.d1 is not None else dt
    return _replaced(state, Dz=state.Dz - scale * amp * src_mask)


def update_pols(state: TMState, m: TMCoeffs) -> TMState:
    """meep update_pols(E): ADE Lorentz recurrence
    P(n+1) = c1 P(n) + c2 P(n-1) + c3 Ez(n)."""
    P_new = m.c1 * state.P_cur + m.c2 * state.P_prev + m.c3 * state.Ez
    return _replaced(state, P_cur=P_new, P_prev=state.P_cur)


def update_eh(state: TMState, m: TMCoeffs, ft: str) -> TMState:
    """meep update_eh(ft): ft="H" forms H = B - sum M (magnetic Lorentz ADE;
    identity when B = mu H is folded into step_db). ft="E" solves E from
    D - P (instantaneous chi2/chi3 via fixed point -- fusion load-bearing,
    the phase boundary sits at meep's)."""
    if ft == "H":
        if m.mc1x is None:
            return state
        # W5.5: M precesses off H(n-1/2), H = B - sum M (mu_inf = 1)
        Mx_new = m.mc1x * state.Mx_cur + m.mc2x * state.Mx_prev + m.mc3x * state.Hx
        My_new = m.mc1y * state.My_cur + m.mc2y * state.My_prev + m.mc3y * state.Hy
        return _replaced(state, Hx=state.Bx - Mx_new.sum(0),
                              Hy=state.By - My_new.sum(0),
                              Mx_cur=Mx_new, Mx_prev=state.Mx_cur,
                              My_cur=My_new, My_prev=state.My_cur)
    # D = eps*E + chi2*E^2 + chi3*E^3  ->  E = rhs / (eps + chi2*E + chi3*E^2)
    rhs = state.Dz - state.P_cur.sum(0)
    Ez = rhs / m.eps_inf
    if m.has_kerr or m.has_chi2:
        for _ in range(3):
            Ez = rhs / (m.eps_inf + m.chi2 * Ez + m.chi3 * Ez * Ez)
    return _replaced(state, Ez=Ez)


def step_tm(
    state: TMState,
    m: TMCoeffs,
    src_amp=0.0,
    src_mask: torch.Tensor | None = None,
    src_h_amp: float = 0.0,
    src_hy_mask: torch.Tensor | None = None,
    src_hx_mask: torch.Tensor | None = None,
) -> TMState:
    """One TM step, composed of the phases in meep's fields::step order:
    step_db(B) -> step_source(B) -> update_eh(H) -> step_db(D) ->
    step_source(D) -> update_pols -> update_eh(E)."""
    state = step_db(state, m, "B")
    if src_hy_mask is not None or src_hx_mask is not None:
        state = step_source(state, m, "B", src_h_amp,
                            src_hy_mask=src_hy_mask, src_hx_mask=src_hx_mask)
    state = update_eh(state, m, "H")
    state = step_db(state, m, "D")
    state = step_source(state, m, "D", src_amp, src_mask=src_mask)
    state = update_pols(state, m)
    return update_eh(state, m, "E")
