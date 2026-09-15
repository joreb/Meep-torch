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
# decompose. State stays torch; the forward curl matvec goes through numpy via
# pylops.MatrixMult (the backward uses the raw scipy transpose of the same
# matrix -- see _Diff). The four diff operators are built once at import for
# the splitter grid.
#
# Differentiable: the numpy round-trip is wrapped in a torch.autograd.Function
# (_Diff) whose backward applies the transpose of the same sparse matrix -- a
# forward difference is linear, so A^T IS its adjoint, no approximation. That
# makes step_pylops usable inside the c_y_splitter adjoint optimization via
# _make_runner(step_pylops, TMState); see optimize_splitter --stepper pylops.
# The adjoint is gated by a dot-product test (<Ax,y> == <x,A^T y>) in
# test_splitter_2d.py, plus an end-to-end d(loss)/d(eps) comparison against
# step_tm.
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

try:                       # package import (pytest, -m) ...
    from .fdtd_pylops import _fwd_x, _fwd_y   # validated, correct layout
    from .stepper_sympy import step_source, update_eh, update_pols
except ImportError:        # ... or plain script run (path invocation)
    from fdtd_pylops import _fwd_x, _fwd_y
    from stepper_sympy import step_source, update_eh, update_pols

try:
    import pylops
    _PYLOPS = True
except ImportError:  # pragma: no cover
    _PYLOPS = False

_CACHE: dict[tuple, dict] = {}


def mats(nx, ny, dx):
    """The four forward-difference operators for this grid, built once per
    (nx, ny, dx) and cached.

    Built LAZILY from the shapes the stepper is actually handed, rather than
    bound at import from splitter_setup. That is what lets this module import
    no meep_torch and no splitter_setup -- and it removes the gdsfactory head
    exec that used to run merely by importing this file, which is what made
    two `gf.Component("y_splitter_default_sbend")` calls collide in one
    process (see optimize_splitter's CELL_LINE).

    Field shapes: Ez (nx, ny), Hx (nx, ny-1), Hy (nx-1, ny). Layout is C-order
    (index = i*ny + j), built with kron like fdtd_pylops.run_pylops.

    Each entry is (A, A_transposed_as_csr, pylops_operator). A^T is stored as
    its own CSR rather than a lazy `.T` because transposing CSR yields CSC and
    the backward pass would convert on every matvec."""
    key = (nx, ny, float(dx))
    if key not in _CACHE:
        Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
        # B phase: Ez (nx, ny) -> dEz/dy (nx, ny-1), dEz/dx (nx-1, ny)
        raw = {
            "dyz": sp.kron(Ix, _fwd_y(ny, dx), format="csr"),
            "dxz": sp.kron(_fwd_x(nx, dx), Iy, format="csr"),
            # D phase: Hx -> dHx/dy (nx, ny-2); Hy -> dHy/dx (nx-2, ny).
            # Hy's y-extent is ny (not ny-1), so its x-diff kron uses I_ny.
            "dyh": sp.kron(Ix, _fwd_y(ny - 1, dx), format="csr"),
            "dxh": sp.kron(_fwd_x(nx - 1, dx), Iy, format="csr"),
        }
        _CACHE[key] = {
            k: (m, sp.csr_matrix(m.T),
                pylops.MatrixMult(m, dtype="float64") if _PYLOPS else m)
            for k, m in raw.items()}
    return _CACHE[key]

try:
    import cupy as _cp
    import cupyx.scipy.sparse as _xsp
    _CUPY = True
except ImportError:  # pragma: no cover
    _CUPY = False

_GPU_MATS: dict[tuple, dict] = {}


def _gpu_mats(dev, nx, ny, dx):
    """cupyx CSR copies of A and A^T, built once per (device, grid).

    A^T gets its own CSR instead of a lazy `.T`: transposing CSR yields CSC,
    and cupyx would convert format on every matvec -- 4 matvecs per step,
    8805 steps per simulation. Built lazily, so a CPU-only run never touches
    the GPU."""
    idx = dev.index if dev.index is not None else _cp.cuda.Device().id
    key = (idx, nx, ny, float(dx))
    if key not in _GPU_MATS:
        with _cp.cuda.Device(idx):
            _GPU_MATS[key] = {
                k: (_xsp.csr_matrix(m), _xsp.csr_matrix(mt))
                for k, (m, mt, _) in mats(nx, ny, dx).items()}
    return _GPU_MATS[key]


def _matvec_gpu(mat, field, out_shape):
    """A @ field entirely on the device.

    DLPack shares the pointer both ways -- no PCIe hop, which is the whole
    point: the host round-trip in the numpy path costs two 2 MB transfers per
    matvec. `.double()` is a no-op on the float64 state and a device-side cast
    otherwise, so it never reintroduces a host copy."""
    x = _cp.from_dlpack(field.detach().contiguous().double()).reshape(-1)
    y = mat @ x
    return torch.from_dlpack(y).reshape(out_shape).to(dtype=field.dtype)


def _on_gpu(field) -> bool:
    return _CUPY and field.is_cuda


class _Diff(torch.autograd.Function):
    """Forward: y = A x through pylops. Backward: grad_x = A^T grad_y.

    The numpy round-trip is what severs autograd, so it has to live inside an
    autograd.Function; A is a constant (geometry-independent) sparse matrix, so
    the backward is just the transpose matvec -- exact, not an approximation."""

    @staticmethod
    def forward(ctx, field, key, out_shape, grid):
        ctx.key, ctx.grid = key, grid
        ctx.in_shape = tuple(field.shape)
        if _on_gpu(field):
            return _matvec_gpu(_gpu_mats(field.device, *grid)[key][0],
                               field, out_shape)
        flat = field.detach().cpu().double().numpy().reshape(-1)
        res = np.asarray(mats(*grid)[key][2] * flat).reshape(out_shape)
        return torch.from_numpy(res).to(dtype=field.dtype, device=field.device)

    @staticmethod
    def backward(ctx, grad_out):
        if _on_gpu(grad_out):
            g = _matvec_gpu(_gpu_mats(grad_out.device, *ctx.grid)[ctx.key][1],
                            grad_out, ctx.in_shape)
            return g, None, None, None
        flat = grad_out.detach().cpu().double().numpy().reshape(-1)
        res = (mats(*ctx.grid)[ctx.key][1] @ flat).reshape(ctx.in_shape)
        g = torch.from_numpy(res).to(dtype=grad_out.dtype, device=grad_out.device)
        return g, None, None, None


def _apply(key, field, out_shape, grid):
    """Differentiable forward difference: A[key] applied to the flattened
    C-order field, reshaped to out_shape, back as a torch tensor. `grid` is
    (nx, ny, dx), carried through so the operators stay lazily built."""
    return _Diff.apply(field, key, out_shape, grid)


def step_db_pylops(state, m, ft):
    """meep step_db with the curl differences from pylops. Plain-branch only
    (the splitter has no mirror/PMC/conductivity/magnetic poles); raises if a
    non-plain feature is on, so we never silently diverge from meep's step_db."""
    if m.y_pmc or m.y_mirror_lo or m.x_mirror_lo or m.y_rot2_lo or m.rot4_lo:
        raise NotImplementedError("pylops step_db: only the plain (no-mirror) "
                                  "branch; the splitter is plain.")
    dt, dx = m.dt, m.dx
    # grid read off the state, not bound at import -- see mats()
    NX, NY = state.Ez.shape
    grid = (NX, NY, dx)
    if ft == "B":
        dEz_dy = _apply("dyz", state.Ez, (NX, NY - 1), grid)
        psi_Ezy = m.b_hy * state.psi_Ezy + m.a_hy * dEz_dy
        curl_hx = dEz_dy / m.k_hy + psi_Ezy
        dEz_dx = _apply("dxz", state.Ez, (NX - 1, NY), grid)
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
    dHx_dy_full = _apply("dyh", state.Hx, (NX, NY - 2), grid)
    dHx_dy = dHx_dy_full[1:-1, :]                              # (NX-2, NY-2)
    dHy_dx_full = _apply("dxh", state.Hy, (NX - 2, NY), grid)
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