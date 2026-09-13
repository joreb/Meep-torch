# SPDX-License-Identifier: MIT
# Drop-in 2D TMz integrator for the Y-splitter where the Yee curl is a devito
# Operator (symbolic forward derivatives) instead of torch slices. Like
# fdtd_pylops_2d, everything else -- CPML psi, source injection, E-solve,
# step_tm phase order -- is meep_torch's, reused verbatim, so the S-parameters
# match the reference tightly; only the curl kernel is devito's.
#
# Linux-only: devito needs libc posix_memalign (absent on Windows). The test
# importorskips it. Per step the relevant field is copied into a devito
# Function, a prebuilt Operator evaluates the forward differences, and the
# results are read back into torch for the identical psi/H/D update. This is
# NOT the devito batched-run model (meep_torch's DFT monitors need the field
# each step), so it is a correctness comparison, not a perf one.
#
# ponytail: devito can't run on this box, so this is best-effort code for the
# user's Linux+CUDA machine. If devito's fdkind='forward' derivative differs
# from the torch forward difference by a sign or factor, the S-params diverge
# and the gate fails -- tune the derivative kind on Linux. The pylops twin
# (validated bitwise) is the proof the drop-in approach is sound.
import numpy as np
import torch

from meep_torch.core.stepper_tm import step_source, update_eh, update_pols

from . import splitter_setup as SS

NX, NY, DX = SS.grid.nx, SS.grid.ny, SS.grid.dx

try:
    from devito import Grid, Function, Eq, Operator  # noqa: F401
    _DEVITO = True
except ImportError:
    _DEVITO = False

# Built lazily on first step (needs devito + the grid).
_op = None


def _build():
    """Prebuild a devito Operator that computes the four forward differences:
    dEz/dy, dEz/dx (from Ez) and dHx/dy, dHy/dx (from Hx, Hy). Inputs/outputs
    are devito Functions over a (NX, NY) grid; the caller copies torch fields
    in and reads the outputs back. fdkind='forward' selects (f[i+1]-f[i])/dx,
    matching meep's forward difference exactly."""
    from devito import Grid, Function, Eq, Operator
    grid = Grid(shape=(NX, NY), extent=(NX * DX, NY * DX))
    # inputs
    Ez = Function(name="Ez", grid=grid, space_order=1)
    Hx = Function(name="Hx", grid=grid, space_order=1)   # lives on (NX, NY-1)
    Hy = Function(name="Hy", grid=grid, space_order=1)   # lives on (NX-1, NY)
    # outputs (full-grid; trimmed/sliced in torch after readback, like pylops)
    dez_dy = Function(name="dez_dy", grid=grid, space_order=0)
    dez_dx = Function(name="dez_dx", grid=grid, space_order=0)
    dhx_dy = Function(name="dhx_dy", grid=grid, space_order=0)
    dhy_dx = Function(name="dhy_dx", grid=grid, space_order=0)
    eqs = [
        Eq(dez_dy, Ez.dy(fdkind="forward")),
        Eq(dez_dx, Ez.dx(fdkind="forward")),
        Eq(dhx_dy, Hx.dy(fdkind="forward")),
        Eq(dhy_dx, Hy.dx(fdkind="forward")),
    ]
    return {"grid": grid, "Ez": Ez, "Hx": Hx, "Hy": Hy,
            "out": (dez_dy, dez_dx, dhx_dy, dhy_dx),
            "op": Operator(eqs)}


def _set(fn, field):
    arr = field.detach().cpu().numpy().astype(np.float64)
    fn.data[:] = 0.0
    fn.data[tuple(slice(0, s) for s in arr.shape)] = arr


def _devito_curl_E(Ez):
    """B phase: set Ez, run the op, return (dEz/dy, dEz/dx) as torch tensors."""
    global _op
    if _op is None:
        _op = _build()
    _set(_op["Ez"], Ez)
    _op["op"].apply()
    dez_dy, dez_dx, _, _ = _op["out"]
    dy = torch.from_numpy(np.asarray(dez_dy.data)[:NX, :NY - 1].copy())
    dx = torch.from_numpy(np.asarray(dez_dx.data)[:NX - 1, :NY].copy())
    return dy.to(Ez.dtype), dx.to(Ez.dtype)


def _devito_curl_H(Hx, Hy):
    """D phase: set Hx+Hy, run the op, return (dHx/dy full, dHy/dx full)."""
    global _op
    if _op is None:
        _op = _build()
    _set(_op["Hx"], Hx)
    _set(_op["Hy"], Hy)
    _op["op"].apply()
    _, _, dhx_dy, dhy_dx = _op["out"]
    dy = torch.from_numpy(np.asarray(dhx_dy.data)[:NX, :NY - 2].copy())
    dx = torch.from_numpy(np.asarray(dhy_dx.data)[:NX - 2, :NY].copy())
    return dy.to(Hx.dtype), dx.to(Hy.dtype)


def step_db_devito(state, m, ft):
    if m.y_pmc or m.y_mirror_lo or m.x_mirror_lo or m.y_rot2_lo or m.rot4_lo:
        raise NotImplementedError("devito step_db: plain branch only.")
    if m.mc1x is not None or m.h1x is not None or m.d1 is not None:
        raise NotImplementedError("devito step_db: magnetic poles / sigma "
                                  "unsupported (splitter has none).")
    import torch.nn.functional as F
    dt = m.dt
    if ft == "B":
        dEz_dy, dEz_dx = _devito_curl_E(state.Ez)
        psi_Ezy = m.b_hy * state.psi_Ezy + m.a_hy * dEz_dy
        curl_hx = dEz_dy / m.k_hy + psi_Ezy
        psi_Ezx = m.b_hx * state.psi_Ezx + m.a_hx * dEz_dx
        curl_hy = dEz_dx / m.k_hx + psi_Ezx
        return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                              Hx=state.Hx - dt * curl_hx,
                              Hy=state.Hy + dt * curl_hy)

    dHx_dy_full, dHy_dx_full = _devito_curl_H(state.Hx, state.Hy)
    dHx_dy = dHx_dy_full[1:-1, :]
    dHy_dx = dHy_dx_full[:, 1:-1]
    psi_Hyx = m.b_ex * state.psi_Hyx + m.a_ex * dHy_dx
    psi_Hxy = m.b_ey * state.psi_Hxy + m.a_ey * dHx_dy
    curl = dHy_dx / m.k_ex + psi_Hyx - dHx_dy / m.k_ey - psi_Hxy
    Dz = state.Dz + dt * F.pad(curl, (1, 1, 1, 1))
    return state._replace(Dz=Dz, psi_Hyx=psi_Hyx, psi_Hxy=psi_Hxy)


def step_devito(state, m, src_amp=0.0, src_mask=None, src_h_amp=0.0,
                src_hy_mask=None, src_hx_mask=None):
    """step_tm with step_db swapped for the devito-curl version."""
    state = step_db_devito(state, m, "B")
    if src_hy_mask is not None or src_hx_mask is not None:
        state = step_source(state, m, "B", src_h_amp,
                            src_hy_mask=src_hy_mask, src_hx_mask=src_hx_mask)
    state = update_eh(state, m, "H")
    state = step_db_devito(state, m, "D")
    state = step_source(state, m, "D", src_amp, src_mask=src_mask)
    state = update_pols(state, m)
    return update_eh(state, m, "E")