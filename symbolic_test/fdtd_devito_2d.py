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
# Validated on Linux (devito 4.8.23): with the grid extent and dtype set below,
# a 3-step state diff against step_tm is ~1e-16, i.e. the devito forward
# derivative is bitwise the torch forward difference. The pylops twin is the
# second, independent proof the drop-in approach is sound.
#
# Differentiable, in the style of slimgroup/Devito4PyTorch (see
# reference/Devito4PyTorch): the devito round-trip lives inside
# torch.autograd.Functions whose backward calls devito's ADJOINT operator, not
# a sparse transpose. Devito does not derive adjoints automatically -- as in
# Devito4PyTorch's AcousticWaveSolver.jacobian_adjoint, the adjoint stencil is
# written by hand and then gated numerically.
#
# The adjoint of the forward difference D: f -> (f[i+1]-f[i])/h is
#   (D^T g)[i] = (g[i-1] - g[i])/h
# with g zero outside its valid extent. _build() emits that as a second
# Operator (an explicit one-index shift -- see the note there on why
# fdkind="backward" does not do it), and test_splitter_2d gates it with a
# dot-product test <Ax, y> == <x, A^T y> against the sparse twin in
# fdtd_pylops_2d, plus an end-to-end d(loss)/d(eps) match against step_tm.
import numpy as np
import torch

try:                       # package import (pytest, -m) ...
    from .stepper_sympy import step_source, update_eh, update_pols
except ImportError:        # ... or plain script run (path invocation)
    from stepper_sympy import step_source, update_eh, update_pols

try:
    from devito import Grid, Function, Eq, Operator  # noqa: F401
    _DEVITO = True
except ImportError:
    _DEVITO = False

# Operators, one set per (nx, ny, dx), built on first use. Keyed rather than
# a single global so this module binds no grid at import -- which is what lets
# it drop the splitter_setup import (and the gdsfactory head exec that came
# with it; two of those in one process is what made the cell names collide).
_OPS: dict[tuple, dict] = {}


def _ops(nx, ny, dx):
    key = (nx, ny, float(dx))
    if key not in _OPS:
        _OPS[key] = _build(nx, ny, dx)
    return _OPS[key]


def _build(NX, NY, DX):
    """Prebuild a devito Operator that computes the four forward differences:
    dEz/dy, dEz/dx (from Ez) and dHx/dy, dHy/dx (from Hx, Hy). Inputs/outputs
    are devito Functions over a (NX, NY) grid; the caller copies torch fields
    in and reads the outputs back. fdkind='forward' selects (f[i+1]-f[i])/dx,
    matching meep's forward difference exactly."""
    from devito import Grid, Function, Eq, Operator
    # extent is the distance between the FIRST and LAST point, so devito's
    # derived spacing is extent/(shape-1): passing NX*DX makes every derivative
    # off by NX/(NX-1) (~0.12% here) -- small enough to sneak past the S-param
    # gate while still being wrong. dtype=float64 matches the torch state; the
    # devito default is float32 and would cap the agreement at ~1e-7.
    grid = Grid(shape=(NX, NY), extent=((NX - 1) * DX, (NY - 1) * DX),
                dtype=np.float64)
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
    # adjoint: (D^T g)[i] = (g[i-1]-g[i])/h = -(backward difference of g).
    # One (Gy, Gx) pair serves both phases -- the B phase feeds it dEz/dy and
    # dEz/dx, the D phase dHx/dy and dHy/dx; only the output slice differs.
    Gy = Function(name="Gy", grid=grid, space_order=1)
    Gx = Function(name="Gx", grid=grid, space_order=1)
    Ay = Function(name="Ay", grid=grid, space_order=0)
    Ax = Function(name="Ax", grid=grid, space_order=0)
    # NB: fdkind="backward" is NOT the stencil we need -- at space_order=1
    # devito emits the same one-sided (f[i+1]-f[i])/h for both fdkinds (checked
    # numerically), so -Gy.dy(fdkind="backward") is -(forward diff), off by one
    # index. Write the shift explicitly instead; the halo zeroed by _set makes
    # both boundary rows come out exactly as A^T's do.
    xd, yd = grid.dimensions
    adj = Operator([Eq(Ay, (Gy.subs(yd, yd - 1) - Gy) / yd.spacing),
                    Eq(Ax, (Gx.subs(xd, xd - 1) - Gx) / xd.spacing)])
    return {"grid": grid, "Ez": Ez, "Hx": Hx, "Hy": Hy,
            "out": (dez_dy, dez_dx, dhx_dy, dhy_dx),
            "op": Operator(eqs),
            "Gy": Gy, "Gx": Gx, "adj_out": (Ay, Ax), "adj": adj}


def _set(fn, field):
    """Copy a torch field into the top-left corner of a full-grid devito
    Function, everything else zero. The halo is zeroed too, not just the
    domain: the adjoint's backward difference reads the low-side halo at index
    0, and a stale value there would silently corrupt the first row/column."""
    arr = field.detach().cpu().numpy().astype(np.float64)
    fn.data_with_halo[:] = 0.0
    fn.data[tuple(slice(0, s) for s in arr.shape)] = arr


def _adjoint(g_dy, g_dx, dy_shape, dx_shape, grid):
    """Run the devito adjoint Operator on (g_dy, g_dx) and return the two
    transposed-difference results sliced to dy_shape / dx_shape. `grid` is
    (nx, ny, dx)."""
    _op = _ops(*grid)
    _set(_op["Gy"], g_dy)
    _set(_op["Gx"], g_dx)
    _op["adj"].apply()
    Ay, Ax = _op["adj_out"]
    ay = torch.from_numpy(np.asarray(Ay.data)[:dy_shape[0], :dy_shape[1]].copy())
    ax = torch.from_numpy(np.asarray(Ax.data)[:dx_shape[0], :dx_shape[1]].copy())
    return (ay.to(dtype=g_dy.dtype, device=g_dy.device),
            ax.to(dtype=g_dx.dtype, device=g_dx.device))


class _CurlE(torch.autograd.Function):
    """B phase: Ez -> (dEz/dy, dEz/dx) via one devito Operator apply."""

    @staticmethod
    def forward(ctx, Ez, grid):
        _op = _ops(*grid)
        ctx.dev, ctx.dt_, ctx.grid = Ez.device, Ez.dtype, grid
        NX, NY = grid[0], grid[1]
        ctx.nx, ctx.ny = NX, NY
        _set(_op["Ez"], Ez)
        _op["op"].apply()
        dez_dy, dez_dx, _, _ = _op["out"]
        dy = torch.from_numpy(np.asarray(dez_dy.data)[:NX, :NY - 1].copy())
        dx = torch.from_numpy(np.asarray(dez_dx.data)[:NX - 1, :NY].copy())
        return (dy.to(dtype=Ez.dtype, device=Ez.device),
                dx.to(dtype=Ez.dtype, device=Ez.device))

    @staticmethod
    def backward(ctx, g_dy, g_dx):
        # both adjoints land on Ez's own (NX, NY) shape and add
        NX, NY = ctx.nx, ctx.ny
        ay, ax = _adjoint(g_dy, g_dx, (NX, NY), (NX, NY), ctx.grid)
        return ay + ax, None


class _CurlH(torch.autograd.Function):
    """D phase: (Hx, Hy) -> (dHx/dy, dHy/dx) via one devito Operator apply."""

    @staticmethod
    def forward(ctx, Hx, Hy, grid):
        _op = _ops(*grid)
        ctx.grid = grid
        NX, NY = grid[0], grid[1]
        ctx.nx, ctx.ny = NX, NY
        _set(_op["Hx"], Hx)
        _set(_op["Hy"], Hy)
        _op["op"].apply()
        _, _, dhx_dy, dhy_dx = _op["out"]
        dy = torch.from_numpy(np.asarray(dhx_dy.data)[:NX, :NY - 2].copy())
        dx = torch.from_numpy(np.asarray(dhy_dx.data)[:NX - 2, :NY].copy())
        return (dy.to(dtype=Hx.dtype, device=Hx.device),
                dx.to(dtype=Hy.dtype, device=Hy.device))

    @staticmethod
    def backward(ctx, g_dy, g_dx):
        # grad_Hx has Hx's shape (NX, NY-1); grad_Hy has Hy's (NX-1, NY)
        NX, NY = ctx.nx, ctx.ny
        gx, gy = _adjoint(g_dy, g_dx, (NX, NY - 1), (NX - 1, NY), ctx.grid)
        return gx, gy, None


def _devito_curl_E(Ez, grid):
    return _CurlE.apply(Ez, grid)


def _devito_curl_H(Hx, Hy, grid):
    return _CurlH.apply(Hx, Hy, grid)


def step_db_devito(state, m, ft):
    if m.y_pmc or m.y_mirror_lo or m.x_mirror_lo or m.y_rot2_lo or m.rot4_lo:
        raise NotImplementedError("devito step_db: plain branch only.")
    if m.mc1x is not None or m.h1x is not None or m.d1 is not None:
        raise NotImplementedError("devito step_db: magnetic poles / sigma "
                                  "unsupported (splitter has none).")
    import torch.nn.functional as F
    dt = m.dt
    # grid read off the state, not bound at import -- see _ops()
    NX, NY = state.Ez.shape
    grid = (NX, NY, m.dx)
    if ft == "B":
        dEz_dy, dEz_dx = _devito_curl_E(state.Ez, grid)
        psi_Ezy = m.b_hy * state.psi_Ezy + m.a_hy * dEz_dy
        curl_hx = dEz_dy / m.k_hy + psi_Ezy
        psi_Ezx = m.b_hx * state.psi_Ezx + m.a_hx * dEz_dx
        curl_hy = dEz_dx / m.k_hx + psi_Ezx
        return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                              Hx=state.Hx - dt * curl_hx,
                              Hy=state.Hy + dt * curl_hy)

    dHx_dy_full, dHy_dx_full = _devito_curl_H(state.Hx, state.Hy, grid)
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