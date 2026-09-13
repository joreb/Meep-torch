# SPDX-License-Identifier: MIT
# Drop-in 2D TMz integrator for the Y-splitter where the Yee curl is a pylops
# LinearOperator (a sparse forward-difference matrix) instead of torch slices.
# Everything else -- the CPML psi recursion, source injection, E-solve, and the
# step_tm phase order -- is byte-identical to meep_torch's stepper_tm, so the
# S-parameters match the meep_torch reference tightly: only the curl kernel
# differs, and a forward difference is the same number in any backend.
#
# step_pylops has the same signature as step_tm and drops into the same run
# loop (splitter_setup.run_with_step), reusing ModeSourceTM + FluxMonitorTM +
# decompose. State stays torch; the curl matvec goes through numpy (CPU) / cupy
# (GPU) via pylops.MatrixMult. The four diff operators are built once at import
# for the splitter grid.
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from meep_torch.core.stepper_tm import (  # reuse these phases verbatim
    step_source, update_eh, update_pols,
)

from . import splitter_setup as SS
from .fdtd_pylops import _fwd_x, _fwd_y  # validated, correct layout

try:
    import pylops
    _PYLOPS = True
except ImportError:  # pragma: no cover
    _PYLOPS = False

NX, NY, DX = SS.grid.nx, SS.grid.ny, SS.grid.dx

# Four forward-difference operators, one per (field, axis), matching the
# field shapes: Ez (NX,NY), Hx (NX, NY-1), Hy (NX-1, NY). Layout is C-order
# (index = i*ny + j), built with kron like fdtd_pylops.run_pylops.
_Ix, _Iy = sp.identity(NX, format="csr"), sp.identity(NY, format="csr")
# B phase: Ez (NX, NY) -> dEz/dy (NX, NY-1), dEz/dx (NX-1, NY)
_Dyf_ez = sp.kron(_Ix, _fwd_y(NY, DX), format="csr")     # (NX*(NY-1), NX*NY)
_Dxf_ez = sp.kron(_fwd_x(NX, DX), _Iy, format="csr")     # ((NX-1)*NY, NX*NY)
# D phase: Hx (NX, NY-1) -> dHx/dy (NX, NY-2); Hy (NX-1, NY) -> dHy/dx (NX-2, NY).
# Hy's y-extent is NY (not NY-1), so the x-diff kronecker uses I_NY.
_Dyf_hx = sp.kron(_Ix, _fwd_y(NY - 1, DX), format="csr")      # (NX*(NY-2), NX*(NY-1))
_Dxf_hy = sp.kron(_fwd_x(NX - 1, DX), _Iy, format="csr")      # ((NX-2)*NY, (NX-1)*NY)

if _PYLOPS:
    _OP = {
        "dyz": pylops.MatrixMult(_Dyf_ez, dtype="float64"),
        "dxz": pylops.MatrixMult(_Dxf_ez, dtype="float64"),
        "dyh": pylops.MatrixMult(_Dyf_hx, dtype="float64"),
        "dxh": pylops.MatrixMult(_Dxf_hy, dtype="float64"),
    }
else:  # scipy fallback so the module imports without pylops (tests importorskip)
    _OP = {
        "dyz": _Dyf_ez, "dxz": _Dxf_ez, "dyh": _Dyf_hx, "dxh": _Dxf_hy,
    }


def _apply(op, field, out_shape):
    """op (pylops LinearOperator or scipy sparse) acts on the flattened
    C-order field; result reshaped to out_shape, returned as a torch tensor
    on the field's dtype."""
    flat = field.detach().cpu().numpy().reshape(-1)
    res = op * flat
    return torch.from_numpy(np.asarray(res).reshape(out_shape)).to(field.dtype)


def step_db_pylops(state, m, ft):
    """meep step_db with the curl differences from pylops. Plain-branch only
    (the splitter has no mirror/PMC/conductivity/magnetic poles); raises if a
    non-plain feature is on, so we never silently diverge from meep's step_db."""
    if m.y_pmc or m.y_mirror_lo or m.x_mirror_lo or m.y_rot2_lo or m.rot4_lo:
        raise NotImplementedError("pylops step_db: only the plain (no-mirror) "
                                  "branch; the splitter is plain.")
    dt, dx = m.dt, m.dx
    if ft == "B":
        dEz_dy = _apply(_OP["dyz"], state.Ez, (NX, NY - 1))
        psi_Ezy = m.b_hy * state.psi_Ezy + m.a_hy * dEz_dy
        curl_hx = dEz_dy / m.k_hy + psi_Ezy
        dEz_dx = _apply(_OP["dxz"], state.Ez, (NX - 1, NY))
        psi_Ezx = m.b_hx * state.psi_Ezx + m.a_hx * dEz_dx
        curl_hy = dEz_dx / m.k_hx + psi_Ezx
        if m.mc1x is not None or m.h1x is not None:
            raise NotImplementedError("pylops step_db: magnetic poles/sigma_b "
                                      "unsupported (splitter has none).")
        return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                              Hx=state.Hx - dt * curl_hx,
                              Hy=state.Hy + dt * curl_hy)

    # D phase: differences on the interior (plain: slice(1,-1) on each axis),
    # computed by applying the full-field diff then slicing -- equivalent to
    # meep's (Hx[xs,1:]-Hx[xs,:-1]) with xs=slice(1,-1).
    dHx_dy_full = _apply(_OP["dyh"], state.Hx, (NX, NY - 2))
    dHx_dy = dHx_dy_full[1:-1, :]                              # (NX-2, NY-2)
    dHy_dx_full = _apply(_OP["dxh"], state.Hy, (NX - 2, NY))
    dHy_dx = dHy_dx_full[:, 1:-1]                              # (NX-2, NY-2)
    psi_Hyx = m.b_ex * state.psi_Hyx + m.a_ex * dHy_dx
    psi_Hxy = m.b_ey * state.psi_Hxy + m.a_ey * dHx_dy
    curl = dHy_dx / m.k_ex + psi_Hyx - dHx_dy / m.k_ey - psi_Hxy
    if m.d1 is not None:
        raise NotImplementedError("pylops step_db: sigma_d unsupported "
                                  "(splitter has none).")
    Dz = state.Dz + dt * F.pad(curl, (1, 1, 1, 1))
    return state._replace(Dz=Dz, psi_Hyx=psi_Hyx, psi_Hxy=psi_Hxy)


def step_pylops(state, m, src_amp=0.0, src_mask=None, src_h_amp=0.0,
                src_hy_mask=None, src_hx_mask=None):
    """step_tm with step_db swapped for the pylops-curl version. Phase order,
    source injection, E-solve and pole update are meep's, reused verbatim."""
    state = step_db_pylops(state, m, "B")
    if src_hy_mask is not None or src_hx_mask is not None:
        state = step_source(state, m, "B", src_h_amp,
                            src_hy_mask=src_hy_mask, src_hx_mask=src_hx_mask)
    state = update_eh(state, m, "H")
    state = step_db_pylops(state, m, "D")
    state = step_source(state, m, "D", src_amp, src_mask=src_mask)
    state = update_pols(state, m)
    return update_eh(state, m, "E")