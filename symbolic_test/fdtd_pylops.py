# SPDX-License-Identifier: MIT
# 2D TMz FDTD built with pylops as a sparse LinearOperator, run on the
# straight-waveguide scenario from scenario.py. GPU via cupy.sparse; CPU via
# scipy.sparse. pylops is a lazy import -- this module loads without it; only
# run_pylops() needs it.
#
# One leapfrog step is collapsed into a single sparse matrix M acting on the
# state u = [Ez; Hx; Hy] (H relabeled to integer steps):
#   Hx^{n+1} = Hx - dt * Dy_fwd Ez
#   Hy^{n+1} = Hy + dt * Dx_fwd Ez
#   Ez^{n+1} = Ez + (dt/eps)*(Dx_bwd Hy - Dy_bwd Hx)
#                  + (dt^2/eps)*(Dx_bwd Dx_fwd + Dy_bwd Dy_fwd) Ez - (dt/eps) J
# i.e. u^{n+1} = M u^n + b*amp(t). pylops MatrixMult(M) applies it; cupy
# backend puts M and u on the GPU. PML: row damping exp(-sigma*dt). This is
# the compact (Lax-Wendroff) FDTD matrix, equivalent to the leapfrog above.
import math

import numpy as np
import scipy.sparse as sp

def _scenario():
    """scenario imported lazily, inside the functions that need it.

    At module scope it would drag meep_torch in transitively (scenario builds
    the reference with meep_torch.core), and fdtd_pylops_2d imports _fwd_x /
    _fwd_y from here -- those are plain scipy matrix builders and must not
    cost a meep_torch import."""
    try:                   # package import (pytest, -m) ...
        from . import scenario as S
    except ImportError:    # ... or plain script run (path invocation)
        import scenario as S
    return S


def _fwd_y(ny: int, dx: float) -> sp.csr_matrix:
    """(ny-1, ny) forward difference in y: (f[j+1]-f[j])/dx."""
    rows = np.repeat(np.arange(ny - 1), 2)
    cols = np.column_stack([np.arange(ny - 1), np.arange(ny - 1) + 1]).ravel()
    data = np.tile([-1.0, 1.0], ny - 1) / dx
    return sp.csr_matrix((data, (rows, cols)), shape=(ny - 1, ny))


def _fwd_x(nx: int, dx: float) -> sp.csr_matrix:
    """(nx-1, nx) forward difference in x: (f[i+1]-f[i])/dx."""
    r = np.repeat(np.arange(nx - 1), 2)
    c = np.column_stack([np.arange(nx - 1), np.arange(nx - 1) + 1]).ravel()
    d = np.tile([-1.0, 1.0], nx - 1) / dx
    return sp.csr_matrix((d, (r, c)), shape=(nx - 1, nx))


def _bwd_x(nx: int, dx: float) -> sp.csr_matrix:
    """(nx, nx-1) backward difference in x with zero ghost at both ends.
    Row 0:        +1/dx at col 0        (ghost f[-1]=0)
    Row i (1..nx-2): -1/dx at col i-1, +1/dx at col i
    Row nx-1:     -1/dx at col nx-2     (ghost f[nx-1]=0)"""
    r = np.concatenate([[0], np.repeat(np.arange(1, nx - 1), 2), [nx - 1]])
    c = np.concatenate([[0],
                        np.column_stack([np.arange(nx - 2), np.arange(1, nx - 1)]).ravel(),
                        [nx - 2]])
    d = np.concatenate([[1.0],
                        np.tile([-1.0, 1.0], nx - 2),
                        [-1.0]]) / dx
    return sp.csr_matrix((d, (r, c)), shape=(nx, nx - 1))


def _bwd_y(ny: int, dx: float) -> sp.csr_matrix:
    """(ny, ny-1) backward difference in y, zero ghost at both ends."""
    r = np.concatenate([[0], np.repeat(np.arange(1, ny - 1), 2), [ny - 1]])
    c = np.concatenate([[0],
                        np.column_stack([np.arange(ny - 2), np.arange(1, ny - 1)]).ravel(),
                        [ny - 2]])
    d = np.concatenate([[1.0], np.tile([-1.0, 1.0], ny - 2), [-1.0]]) / dx
    return sp.csr_matrix((d, (r, c)), shape=(ny, ny - 1))


def _pml_damp(nx: int, ny: int, npml: int, dt: float) -> np.ndarray:
    """Per-cell damping exp(-sigma*dt), sigma = quadratic PML ramp on edges."""
    sx = np.zeros(nx)
    sy = np.zeros(ny)
    for arr, n in ((sx, nx), (sy, ny)):
        idx = np.arange(n)
        dist = np.minimum(idx, n - 1 - idx)
        edge = dist < npml
        arr[edge] = 0.8 * ((npml - dist[edge]) / npml) ** 2
    sx_g, sy_g = np.meshgrid(sx, sy, indexing="ij")
    return np.exp(-(sx_g + sy_g) * dt)


def run_pylops(gpu: bool = False) -> np.ndarray:
    """Run the straight-waveguide with pylops -> Ez array (NX, NY) on CPU."""
    import pylops

    S = _scenario()
    nx, ny, dx = S.NX, S.NY, S.DX
    dt = dx * 0.5
    n = S.n_steps(dt)

    # 2D diffs matching Ez.reshape(-1) C-order layout (index = i*ny + j):
    #   diff in y (fast axis) = I_x (kron) op_y ; diff in x (slow) = op_x (kron) I_y
    Ix, Iy = sp.identity(nx), sp.identity(ny)
    Dyf = sp.kron(Ix, _fwd_y(ny, dx), format="csr")   # (nx*(ny-1), nx*ny)
    Dxf = sp.kron(_fwd_x(nx, dx), Iy, format="csr")   # ((nx-1)*ny, nx*ny)
    Dxb = sp.kron(_bwd_x(nx, dx), Iy, format="csr")   # (nx*ny, (nx-1)*ny)
    Dyb = sp.kron(Ix, _bwd_y(ny, dx), format="csr")   # (nx*ny, nx*(ny-1))

    ne, nhx, nhy = nx * ny, nx * (ny - 1), (nx - 1) * ny
    eps = S.eps_field("cpu").numpy().reshape(-1)
    inv_eps = (1.0 / eps)

    # M blocks (Ez rows, then Hx rows, then Hy rows)
    lap = (Dxb @ Dxf + Dyb @ Dyf)                      # (ne, ne), <= 0
    M_ee = sp.diags(np.ones(ne)) + sp.diags(inv_eps * dt * dt) @ lap
    M_ehx = -sp.diags(inv_eps * dt) @ Dyb
    M_ehy = sp.diags(inv_eps * dt) @ Dxb
    M_hxe = -dt * Dyf
    M_hye = dt * Dxf

    M = sp.bmat([
        [M_ee,        M_ehx,        M_ehy],
        [M_hxe,       sp.eye(nhx),  None],
        [M_hye,       None,         sp.eye(nhy)],
    ], format="csr")

    # PML row damping
    damp_e = _pml_damp(nx, ny, S.NPML, dt).reshape(-1)
    # H grids damping: average edge sigma onto the staggered positions
    damp_hx = _pml_damp(nx, ny, S.NPML, dt)[:, :-1].reshape(-1)
    damp_hy = _pml_damp(nx, ny, S.NPML, dt)[:-1, :].reshape(-1)
    damp = np.concatenate([damp_e, damp_hx, damp_hy])
    M = sp.diags(damp) @ M

    # source vector b: -(dt/eps)*amp(t) injected into Ez at the source cell
    ix = int(round((S.SRC_X - S.X_LO) / dx))
    iy = ny // 2
    src_ez = ix * ny + iy
    b = np.zeros(ne + nhx + nhy)
    b[src_ez] = -dt * inv_eps[src_ez]

    xp = _backend(gpu)
    if gpu:
        # sparse lives in cupyx, not cupy: `cupy.csr_matrix` does not exist.
        import cupyx.scipy.sparse as xsp
        M_xp = xsp.csr_matrix(M)
    else:
        M_xp = M
    u = xp.zeros(ne + nhx + nhy)
    b_xp = xp.asarray(b)
    Aop = pylops.MatrixMult(M_xp, dtype="float64")

    for t in range(n):
        amp = math.cos(2.0 * math.pi * S.FREQ * (t * dt))
        u = Aop * u + amp * b_xp

    ez = u[:ne]
    if gpu:
        ez = ez.get()
    return np.asarray(ez).reshape(nx, ny)


def _backend(gpu: bool):
    """numpy, or cupy when gpu=True.

    Raises rather than falling back: a silent CPU fallback reports a CPU
    number under a GPU label, which is worse than no number at all."""
    if not gpu:
        return np
    import cupy as xp   # ImportError here is the honest answer
    return xp
