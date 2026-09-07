# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .grid import Grid


class TEState(NamedTuple):
    Hz: torch.Tensor       # (nx-1, ny-1) at n     (dual of TM Ez)
    Ex: torch.Tensor       # (nx-1, ny)   at n-1/2 (dual of TM -Hx)
    Ey: torch.Tensor       # (nx, ny-1)   at n-1/2 (dual of TM -Hy)
    Dx: torch.Tensor       # (nx-1, ny)
    Dy: torch.Tensor       # (nx, ny-1)
    Px_cur: torch.Tensor   # (np, nx-1, ny)
    Px_prev: torch.Tensor
    Py_cur: torch.Tensor   # (np, nx, ny-1)
    Py_prev: torch.Tensor
    psi_Eyx: torch.Tensor  # (nx-1, ny-1)
    psi_Exy: torch.Tensor  # (nx-1, ny-1)
    psi_Hzy: torch.Tensor  # (nx-1, ny-2)
    psi_Hzx: torch.Tensor  # (nx-2, ny-1)


def init_te_state(grid: Grid, n_poles: int = 0, n_poles_y: int | None = None) -> TEState:
    """Allocate TE state. n_poles sizes the Ex-axis polarization buffers
    (Px_cur/Px_prev); n_poles_y sizes the Ey-axis buffers (Py_cur/Py_prev)
    and defaults to n_poles (shared-poles case). Per-axis dispersion passes
    differing counts (poles + poles_x vs poles + poles_y)."""
    nx, ny = grid.nx, grid.ny
    if n_poles_y is None:
        n_poles_y = n_poles
    return TEState(
        Hz=grid.zeros(nx - 1, ny - 1),
        Ex=grid.zeros(nx - 1, ny),
        Ey=grid.zeros(nx, ny - 1),
        Dx=grid.zeros(nx - 1, ny),
        Dy=grid.zeros(nx, ny - 1),
        Px_cur=grid.zeros(n_poles, nx - 1, ny),
        Px_prev=grid.zeros(n_poles, nx - 1, ny),
        Py_cur=grid.zeros(n_poles_y, nx, ny - 1),
        Py_prev=grid.zeros(n_poles_y, nx, ny - 1),
        psi_Eyx=grid.zeros(nx - 1, ny - 1),
        psi_Exy=grid.zeros(nx - 1, ny - 1),
        psi_Hzy=grid.zeros(nx - 1, ny - 2),
        psi_Hzx=grid.zeros(nx - 2, ny - 1),
    )


def _avg4_to_ex(Ey: torch.Tensor) -> torch.Tensor:
    # Ey (nx, ny-1) -> value at Ex positions (nx-1, ny); zero-pad y boundaries
    Eyp = F.pad(Ey, (1, 1))                                    # (nx, ny+1)
    return 0.25 * (Eyp[:-1, :-1] + Eyp[:-1, 1:] + Eyp[1:, :-1] + Eyp[1:, 1:])


def _avg4_to_ey(Ex: torch.Tensor) -> torch.Tensor:
    # Ex (nx-1, ny) -> value at Ey positions (nx, ny-1); zero-pad x boundaries
    Exp = F.pad(Ex, (0, 0, 1, 1))                              # (nx+1, ny)
    return 0.25 * (Exp[:-1, :-1] + Exp[:-1, 1:] + Exp[1:, :-1] + Exp[1:, 1:])


# --- meep fields::step phases (src/step.cpp) ---------------------------------
# step_te is split at meep's phase boundaries; the math lines are the original
# fused kernel's, only moved. The engine's TM-TE duality (Hz<->Ez,
# (Ex,Ey)<->(-Hx,-Hy)) staggers Hz on integer steps, so a TE step enters
# meep's phase cycle at the D half: step_db(D) -> step_source(D) ->
# update_pols -> update_eh(E), then step_db(B) -> step_source(B) ->
# update_eh(H). This keeps the discrete TE Hz update graph isomorphic to the
# TM Ez graph (and every TE monitor/oracle convention) intact.

def step_db(state: TEState, m, ft: str) -> TEState:
    """meep step_db(ft): ft="D" advances Dx/Dy from curl Hz, ft="B" advances
    Hz from curl E (B = mu H folded; sigma_b inside hz1/hz2). CPML psi
    updates and sigma damping live here (meep applies its sigma in step_db)."""
    dt, dx = m.dt, m.dx
    if ft == "D":
        # --- D update (from Hz at time n) ---
        dHz_dy = (state.Hz[:, 1:] - state.Hz[:, :-1]) / dx      # (nx-1, ny-2)
        psi_Hzy = m.b_ey * state.psi_Hzy + m.a_ey * dHz_dy
        curl_x = F.pad(dHz_dy / m.k_ey + psi_Hzy, (1, 1))

        dHz_dx = (state.Hz[1:, :] - state.Hz[:-1, :]) / dx      # (nx-2, ny-1)
        psi_Hzx = m.b_ex * state.psi_Hzx + m.a_ex * dHz_dx
        curl_y = F.pad(dHz_dx / m.k_ex + psi_Hzx, (0, 0, 1, 1))
        if m.y_mirror_lo == +1:
            # even y-mirror: the on-plane Ex row is live (tangential E,
            # even); the odd Hz ghost across the cut gives dHz/dy = 2 Hz[0]/dx
            # there. Row 0 sits outside the psi arrays (no PML at the cut).
            # For phase -1 the on-plane Ex row is odd, stays zero, and the
            # zero-pad ghost above is exact -- no substitution.
            curl_x = torch.cat([2.0 * state.Hz[:, :1] / dx, curl_x[:, 1:]], dim=1)
        if m.x_mirror_lo == +1:
            # even x-mirror: same structure on the x axis (Ey tangential)
            curl_y = torch.cat([2.0 * state.Hz[:1, :] / dx, curl_y[1:, :]], dim=0)
        if m.d1x is not None:
            # W5: dD/dt = curl H - sigma_d D
            Dx = m.d1x * state.Dx + m.d2x * curl_x
            Dy = m.d1y * state.Dy - m.d2y * curl_y
        else:
            Dx = state.Dx + dt * curl_x
            Dy = state.Dy - dt * curl_y
        return state._replace(Dx=Dx, Dy=Dy, psi_Hzy=psi_Hzy, psi_Hzx=psi_Hzx)

    # --- Hz update (from E at n+1/2) ---
    dEy_dx = (state.Ey[1:, :] - state.Ey[:-1, :]) / dx          # (nx-1, ny-1)
    dEx_dy = (state.Ex[:, 1:] - state.Ex[:, :-1]) / dx          # (nx-1, ny-1)
    psi_Eyx = m.b_hx * state.psi_Eyx + m.a_hx * dEy_dx
    psi_Exy = m.b_hy * state.psi_Exy + m.a_hy * dEx_dy
    curl_z = dEy_dx / m.k_hx + psi_Eyx - dEx_dy / m.k_hy - psi_Exy
    if m.hz1 is not None:
        # W5: dB/dt = -curl E - sigma_b B, B = mu H
        Hz = m.hz1 * state.Hz - m.hz2 * curl_z
    else:
        Hz = state.Hz - dt * curl_z
    return state._replace(Hz=Hz, psi_Eyx=psi_Eyx, psi_Exy=psi_Exy)


def step_source(state: TEState, m, ft: str, amp=0.0,
                src_mask: torch.Tensor | None = None,
                src_hz_mask: torch.Tensor | None = None) -> TEState:
    """meep step_source(ft): soft current injection -- electric currents into
    Dy (ft="D"; scaled by d2y under sigma_d, like the fused kernel), the
    magnetic TFSF current into Hz (ft="B")."""
    dt = m.dt
    if ft == "B":
        if src_hz_mask is None:
            return state
        return state._replace(Hz=state.Hz - dt * amp * src_hz_mask)
    if src_mask is None:
        return state
    scale = m.d2y if m.d1x is not None else dt
    return state._replace(Dy=state.Dy - scale * amp * src_mask)


def update_pols(state: TEState, m) -> TEState:
    """meep update_pols(E): per-axis ADE Lorentz recurrence
    P(n+1/2) = c1 P(n-1/2) + c2 P(n-3/2) + c3 E(n-1/2)."""
    Px_new = m.c1x * state.Px_cur + m.c2x * state.Px_prev + m.c3x * state.Ex
    Py_new = m.c1y * state.Py_cur + m.c2y * state.Py_prev + m.c3y * state.Ey
    if m.c3xy is not None:
        # oriented poles: off-diagonal chi drives Px from Ey (and Py from Ex)
        Px_new = Px_new + m.c3xy * _avg4_to_ex(state.Ey)
        Py_new = Py_new + m.c3yx * _avg4_to_ey(state.Ex)
    return state._replace(Px_cur=Px_new, Px_prev=state.Px_cur,
                          Py_cur=Py_new, Py_prev=state.Py_cur)


def update_eh(state: TEState, m, ft: str) -> TEState:
    """meep update_eh(ft): ft="E" solves Ex/Ey from D - P (tensor eps and the
    instantaneous chi2/chi3 fixed point -- fusion load-bearing, the phase
    boundary sits at meep's). ft="H" is the identity (B = mu H folded into
    step_db)."""
    if ft == "H":
        return state
    # D = eps*E + chi2*E^2 + chi3*|E|^2*E  ->  E = rhs/(eps + chi2*E + chi3*|E|^2)
    # chi2 here is the per-component self-term (scalar d_iii reduction); the
    # full d-tensor cross-terms are dropped -- TM is the W5.75 SHG target.
    rhs_x = state.Dx - state.Px_cur.sum(0)
    rhs_y = state.Dy - state.Py_cur.sum(0)
    if m.inv_xx_x is not None:
        # W5: in-plane tensor eps -- E = eps^{-1} D, cross terms Yee-averaged
        Ex = m.inv_xx_x * rhs_x + m.inv_xy_x * _avg4_to_ex(rhs_y)
        Ey = m.inv_yy_y * rhs_y + m.inv_xy_y * _avg4_to_ey(rhs_x)
    else:
        Ex = rhs_x / m.eps_x
        Ey = rhs_y / m.eps_y
    if m.has_kerr or m.has_chi2:
        for _ in range(3):
            E2x = Ex * Ex + _avg4_to_ex(Ey) ** 2
            E2y = Ey * Ey + _avg4_to_ey(Ex) ** 2
            Ex = rhs_x / (m.eps_x + m.chi2_x * Ex + m.chi3_x * E2x)
            Ey = rhs_y / (m.eps_y + m.chi2_y * Ey + m.chi3_y * E2y)
    return state._replace(Ex=Ex, Ey=Ey)


def step_te(state: TEState, m, src_amp=0.0, src_mask_ey: torch.Tensor | None = None,
            src_h_amp: float = 0.0, src_hz_mask: torch.Tensor | None = None) -> TEState:
    """One TE step, composed of meep's phases entered at the D half (see the
    phase-block comment above): step_db(D) -> step_source(D) -> update_pols
    -> update_eh(E) -> step_db(B) -> step_source(B) -> update_eh(H)."""
    state = step_db(state, m, "D")
    state = step_source(state, m, "D", src_amp, src_mask=src_mask_ey)
    state = update_pols(state, m)
    state = update_eh(state, m, "E")
    state = step_db(state, m, "B")
    if src_hz_mask is not None:
        state = step_source(state, m, "B", src_h_amp, src_hz_mask=src_hz_mask)
    return update_eh(state, m, "H")
