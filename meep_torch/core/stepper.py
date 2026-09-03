# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .coeffs import TMCoeffs
from .grid import Grid


class TMState(NamedTuple):
    Ez: torch.Tensor       # (nx, ny)        E_z at time n
    Hx: torch.Tensor       # (nx, ny-1)      at n-1/2
    Hy: torch.Tensor       # (nx-1, ny)      at n-1/2
    Dz: torch.Tensor       # (nx, ny)
    P_cur: torch.Tensor    # (n_poles, nx, ny)
    P_prev: torch.Tensor   # (n_poles, nx, ny)
    psi_Ezy: torch.Tensor  # (nx, ny-1)      CPML memory, dEz/dy in Hx update
    psi_Ezx: torch.Tensor  # (nx-1, ny)      dEz/dx in Hy update
    psi_Hyx: torch.Tensor  # (nx-2, ny-2) or (nx-2, ny) if y_pmc
    psi_Hxy: torch.Tensor  # same shape as psi_Hyx
    # W5.5 magnetic Lorentz poles (allocated only when n_mpoles > 0)
    Bx: torch.Tensor | None = None       # (nx, ny-1)
    By: torch.Tensor | None = None       # (nx-1, ny)
    Mx_cur: torch.Tensor | None = None   # (nm, nx, ny-1)
    Mx_prev: torch.Tensor | None = None
    My_cur: torch.Tensor | None = None   # (nm, nx-1, ny)
    My_prev: torch.Tensor | None = None


def init_tm_state(grid: Grid, n_poles: int = 0, y_pmc: bool = False,
                  n_mpoles: int = 0, y_mirror: int = 0) -> TMState:
    nx, ny = grid.nx, grid.ny
    # psi_Hxy rows: PEC interior ny-2; both-wall PMC ny; even-mirror lo ny-1
    ny_e = ny if y_pmc else (ny - 1 if y_mirror == +1 else ny - 2)
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
        psi_Hyx=grid.zeros(nx - 2, ny_e),
        psi_Hxy=grid.zeros(nx - 2, ny_e),
        **mag,
    )


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
        dEz_dy = (state.Ez[:, 1:] - state.Ez[:, :-1]) / dx        # (nx, ny-1)
        psi_Ezy = m.b_hy * state.psi_Ezy + m.a_hy * dEz_dy
        curl_hx = dEz_dy / m.k_hy + psi_Ezy
        dEz_dx = (state.Ez[1:, :] - state.Ez[:-1, :]) / dx        # (nx-1, ny)
        psi_Ezx = m.b_hx * state.psi_Ezx + m.a_hx * dEz_dx
        curl_hy = dEz_dx / m.k_hx + psi_Ezx
        if m.mc1x is not None:
            # W5.5 mu-Lorentz: B integrates the curl; update_eh(H) forms H = B - sum M
            return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                                  Bx=state.Bx - dt * curl_hx,
                                  By=state.By + dt * curl_hy)
        if m.h1x is not None:
            # W5: dB/dt = -curl E - sigma_b B, B = mu H (fused: mu inside h2*)
            return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                                  Hx=m.h1x * state.Hx - m.h2x * curl_hx,
                                  Hy=m.h1y * state.Hy + m.h2y * curl_hy)
        return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                              Hx=state.Hx - dt * curl_hx,
                              Hy=state.Hy + dt * curl_hy)

    # --- D update (from H at n+1/2) ---
    Hx, Hy = state.Hx, state.Hy
    if m.y_pmc:
        Hx_pad = F.pad(Hx, (1, 1))                                # PMC: H=0 beyond wall
        dHx_dy = (Hx_pad[1:-1, 1:] - Hx_pad[1:-1, :-1]) / dx      # (nx-2, ny)
        dHy_dx = (Hy[1:, :] - Hy[:-1, :]) / dx                    # (nx-2, ny)
        pad_spec = (0, 0, 1, 1)                                   # pad x only
    elif m.y_mirror_lo:
        # even-mirror cut at j=0: PMC wall low (H=0 ghost), PEC top
        Hx_pad = F.pad(Hx, (1, 0))                                # low ghost only
        dHx_dy = (Hx_pad[1:-1, 1:] - Hx_pad[1:-1, :-1]) / dx      # (nx-2, ny-1)
        dHy_dx = (Hy[1:, :-1] - Hy[:-1, :-1]) / dx                # (nx-2, ny-1)
        pad_spec = (0, 1, 1, 1)                                   # pad top y + x
    else:
        dHx_dy = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dx              # (nx-2, ny-2)
        dHy_dx = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx              # (nx-2, ny-2)
        pad_spec = (1, 1, 1, 1)

    psi_Hyx = m.b_ex * state.psi_Hyx + m.a_ex * dHy_dx
    psi_Hxy = m.b_ey * state.psi_Hxy + m.a_ey * dHx_dy
    curl = dHy_dx / m.k_ex + psi_Hyx - dHx_dy / m.k_ey - psi_Hxy
    if m.d1 is not None:
        # W5: dD/dt = curl H - sigma_d D
        Dz = m.d1 * state.Dz + m.d2 * F.pad(curl, pad_spec)
    else:
        Dz = state.Dz + dt * F.pad(curl, pad_spec)
    return state._replace(Dz=Dz, psi_Hyx=psi_Hyx, psi_Hxy=psi_Hxy)


def step_source(state: TMState, m: TMCoeffs, ft: str, amp=0.0,
                src_mask: torch.Tensor | None = None,
                src_hy_mask: torch.Tensor | None = None,
                src_hx_mask: torch.Tensor | None = None) -> TMState:
    """meep step_source(ft): soft current injection -- electric currents into
    D (ft="D"; scaled by d2 under sigma_d, like the fused kernel), magnetic
    TFSF currents into B/H (ft="B")."""
    dt = m.dt
    if ft == "B":
        # into B when magnetic poles allocate it (meep injects into B);
        # otherwise straight into the folded H
        Hy, Hx = (state.By, state.Bx) if state.Bx is not None else (state.Hy, state.Hx)
        if src_hy_mask is not None:
            Hy = Hy + dt * amp * src_hy_mask
        if src_hx_mask is not None:
            Hx = Hx + dt * amp * src_hx_mask
        if state.Bx is not None:
            return state._replace(Bx=Hx, By=Hy)
        return state._replace(Hx=Hx, Hy=Hy)
    if src_mask is None:
        return state
    scale = m.d2 if m.d1 is not None else dt
    return state._replace(Dz=state.Dz + scale * amp * src_mask)


def update_pols(state: TMState, m: TMCoeffs) -> TMState:
    """meep update_pols(E): ADE Lorentz recurrence
    P(n+1) = c1 P(n) + c2 P(n-1) + c3 Ez(n)."""
    P_new = m.c1 * state.P_cur + m.c2 * state.P_prev + m.c3 * state.Ez
    return state._replace(P_cur=P_new, P_prev=state.P_cur)


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
        return state._replace(Hx=state.Bx - Mx_new.sum(0),
                              Hy=state.By - My_new.sum(0),
                              Mx_cur=Mx_new, Mx_prev=state.Mx_cur,
                              My_cur=My_new, My_prev=state.My_cur)
    # D = eps*E + chi2*E^2 + chi3*E^3  ->  E = rhs / (eps + chi2*E + chi3*E^2)
    rhs = state.Dz - state.P_cur.sum(0)
    Ez = rhs / m.eps_inf
    if m.has_kerr or m.has_chi2:
        for _ in range(3):
            Ez = rhs / (m.eps_inf + m.chi2 * Ez + m.chi3 * Ez * Ez)
    return state._replace(Ez=Ez)


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
