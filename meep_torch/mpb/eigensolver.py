# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""FDFD Bloch band eigensolver for 2D lattices (MPB's omega(k) problem).

MPB asks: given a Bloch vector k on a periodic cell, what are the lowest
band frequencies omega_n(k)? That is a Hermitian eigenproblem, so the N
lowest eigenvalues come out **complete and already ordered** -- band n is
eigenvalue n, by construction.

That is the whole reason this exists. The Bloch-FDTD + harmonic-inversion
path in :mod:`meep_torch.mpb.solver` infers band index from the position of
a resonance in a sorted list of what harminv happened to detect, so any
missed or duplicated mode silently renumbers every band above it. Measured
against ``tests/examples/data/sq_tut_gamma_mpb.json``: band 1 is missed
outright at small |k| (0.1,0), (0.2,0), (0.3,0), (0.1,0.1), and elsewhere a
near-degenerate pair survives the dedup window and lands band 1's frequency
in band 2's slot. An eigensolve cannot express either failure.

Polarizations (2D, z-invariant structure, in-plane k -- see the parity note
in :mod:`meep_torch.mpb.solver`):

  TM / zodd  (Ez, Hx, Hy):  -laplacian Ez = omega^2 eps Ez
      generalized Hermitian, A z = omega^2 M z with M = diag(eps)

  TE / zeven (Hz, Ex, Ey):  -div((1/eps) grad Hz) = omega^2 Hz
      standard Hermitian, 1/eps taken at the two staggered edge positions

Both use the same Bloch derivative matrices. Frequencies are returned in
meep units (a = 1, c = 1, f = omega / 2pi).

Accuracy is 2nd-order in the grid spacing, not spectral like MPB's
planewave basis, so oracle comparisons want resolution-aware tolerances
rather than golden equality (see wiki/tasks/port_mpb-eigensolver.md).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

__all__ = ["bloch_bands_2d", "bloch_d_matrices"]


def _roll_matrix(n: int, phase: complex) -> sp.csr_matrix:
    """Shift-by-one operator with a Bloch phase on the wrap-around entry.

    Row i picks entry i+1, so the last row wraps to entry 0 carrying
    ``phase`` = exp(i k L): that factor is exactly the Bloch condition
    u(x + L) = exp(i k L) u(x) applied to the periodic envelope.
    """
    rows = np.arange(n)
    cols = (rows + 1) % n
    vals = np.ones(n, dtype=complex)
    vals[-1] = phase
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def bloch_d_matrices(nx: int, ny: int, dx: float, dy: float,
                     kx: float, ky: float):
    """Forward Bloch difference operators (Dxf, Dyf) on an nx*ny cell.

    ``kx``/``ky`` are cartesian Bloch components in radians per unit length,
    so the wrap phases are exp(i kx Lx) with Lx = nx*dx.

    Only the forward differences are returned: with periodic wrap on a
    uniform grid the backward difference is -Dxf^H exactly, so every
    operator below can be written with Dxf and its adjoint. That is what
    keeps the assembled operators Hermitian to machine precision, which the
    eigensolvers rely on.
    """
    Ix, Iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
    sx = _roll_matrix(nx, np.exp(1j * kx * nx * dx))
    sy = _roll_matrix(ny, np.exp(1j * ky * ny * dy))
    # column-major (x fastest) flattening: kron(Iy, .) acts on x
    dxf = sp.kron(Iy, (sx - sp.identity(nx, format="csr")) / dx, format="csr")
    dyf = sp.kron((sy - sp.identity(ny, format="csr")) / dy, Ix, format="csr")
    return dxf, dyf


def _avg_forward(eps: np.ndarray, axis: int) -> np.ndarray:
    """Two-point average of a cell-centred field onto the forward edge.

    Periodic in both axes, matching the Bloch cell: the Yee edge value
    between cell i and i+1 wraps to (last, first) at the boundary.
    """
    return 0.5 * (eps + np.roll(eps, -1, axis=axis))


def _assemble(eps: np.ndarray, dx: float, dy: float, kx: float, ky: float,
              polarization: str, eps_te=None):
    nx, ny = eps.shape
    dxf, dyf = bloch_d_matrices(nx, ny, dx, dy, kx, ky)
    if polarization == "tm":
        # -laplacian Ez = omega^2 eps Ez.  Dxf^H Dxf is -d2/dx2 (PSD).
        A = (dxf.conj().T @ dxf + dyf.conj().T @ dyf).tocsr()
        M = sp.diags(eps.reshape(-1, order="F"))
        return A, M
    # TE: d/dx((1/eps_yy) d/dx Hz) + d/dy((1/eps_xx) d/dy Hz) = -omega^2 Hz.
    # Note which component goes with which derivative: d/dx Hz is set by
    # E_y and d/dy Hz by E_x, so the x-derivative carries eps_yy and the
    # y-derivative eps_xx. Swapping them is invisible for scalar eps and
    # wrong for a smoothed tensor.
    # 1/eps sits on the edge the derivative lands on, hence the forward
    # average along that same axis.
    exx, eyy = (eps, eps) if eps_te is None else eps_te
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    inv_x = 1.0 / _avg_forward(eyy, axis=0)
    inv_y = 1.0 / _avg_forward(exx, axis=1)
    Wx = sp.diags(inv_x.reshape(-1, order="F"))
    Wy = sp.diags(inv_y.reshape(-1, order="F"))
    A = (dxf.conj().T @ Wx @ dxf + dyf.conj().T @ Wy @ dyf).tocsr()
    return A.tocsr(), sp.identity(nx * ny, format="csr")


def _sorted_freqs(vals: np.ndarray, num_bands: int) -> list[float]:
    """omega^2 eigenvalues -> meep frequencies f = omega / 2pi, ascending.

    Tiny negative eigenvalues are round-off around the omega = 0 mode that
    genuinely exists at k = 0 (Ez = const for TM); clamping at zero keeps
    that band at exactly 0.0, which is what meep.mpb reports.
    """
    vals = np.sort(np.real(vals))
    vals = np.clip(vals, 0.0, None)
    return [float(np.sqrt(v) / (2 * np.pi)) for v in vals[:num_bands]]


def _solve_scipy(A, M, num_bands: int) -> list[float]:
    n = A.shape[0]
    if num_bands >= n - 1:
        dense = A.toarray()
        Md = M.toarray() if sp.issparse(M) else M
        vals = np.linalg.eigvalsh(np.linalg.solve(Md, dense))
        return _sorted_freqs(vals, num_bands)
    # Shift just below zero: at k = 0 the TM operator is singular (the
    # omega = 0 mode), and shift-invert at sigma = 0 exactly would try to
    # factorize a singular matrix.
    vals = spl.eigsh(A.astype(complex), k=num_bands, M=M.astype(complex),
                     sigma=-1e-7, which="LM", return_eigenvectors=False)
    return _sorted_freqs(vals, num_bands)


def _solve_torch(eps_t: torch.Tensor, dx: float, dy: float, kx: float,
                 ky: float, num_bands: int, polarization: str,
                 eps_te=None) -> torch.Tensor:
    """Differentiable path: assemble in torch and eigendecompose.

    The operator has to be built from the eps *tensor* for autograd to
    reach it -- going through numpy would silently detach, which is the
    whole point of this path. The Bloch difference matrices themselves
    depend only on the grid and k, so they stay scipy-built constants and
    only the eps-dependent diagonals carry gradient.

    ponytail: dense eigvalsh, O(N^2) memory in the number of cells (~16 MB
    at resolution 32, ~270 MB at 64). Fine for the cell sizes MPB band
    structures use; switch to torch.lobpcg on a sparse operator if this
    ever needs a large supercell.
    """
    nx, ny = eps_t.shape
    dxf, dyf = bloch_d_matrices(nx, ny, dx, dy, kx, ky)
    Dx = torch.as_tensor(dxf.toarray(), dtype=torch.complex128)
    Dy = torch.as_tensor(dyf.toarray(), dtype=torch.complex128)
    eps_c = eps_t.to(torch.complex128)
    if polarization == "tm":
        # A is eps-independent; the gradient enters through eps^{-1/2} only
        A = Dx.conj().T @ Dx + Dy.conj().T @ Dy
        d = torch.sqrt(eps_c.T.reshape(-1))
        B = A / d[:, None] / d[None, :]
    else:
        # x-derivative carries eps_yy, y-derivative eps_xx (see _assemble)
        exx, eyy = ((eps_c, eps_c) if eps_te is None else
                    tuple(torch.as_tensor(e).to(torch.complex128) for e in eps_te))
        inv_x = (1.0 / (0.5 * (eyy + torch.roll(eyy, -1, 0)))).T.reshape(-1)
        inv_y = (1.0 / (0.5 * (exx + torch.roll(exx, -1, 1)))).T.reshape(-1)
        B = (Dx.conj().T @ (inv_x[:, None] * Dx)
             + Dy.conj().T @ (inv_y[:, None] * Dy))
    B = 0.5 * (B + B.conj().T)
    vals = torch.linalg.eigvalsh(B)[:num_bands]
    return torch.sqrt(torch.clamp(vals, min=0.0)) / (2 * np.pi)


def bloch_bands_2d(eps, dx: float, dy: float, kx: float, ky: float,
                   num_bands: int, polarization: str = "tm",
                   differentiable: bool = False, eps_te=None) -> list[float]:
    """Lowest ``num_bands`` frequencies at Bloch vector (kx, ky).

    ``eps``: (nx, ny) cell-centred relative permittivity.
    ``kx``/``ky``: cartesian, radians per unit length (2*pi*k_meep).
    ``polarization``: ``"tm"`` (zodd, Ez) or ``"te"`` (zeven, Hz).
    ``eps_te``: optional ``(eps_xx, eps_yy)`` pair for the TE operator --
    subpixel-smoothed diagonal from
    :func:`meep_torch.geometry.smoothing.blend_te_supersampled`. Without it
    TE uses the scalar ``eps`` and converges at ~1st order on a staircased
    boundary. Ignored for TM, whose single E component runs parallel to
    every interface and needs no tensor treatment.
    ``differentiable``: torch path with autograd through the eigensolve, so
    bands carry gradient w.r.t. ``eps``; returns a tensor. The default
    scipy path is faster, detached, and returns a plain list.

    Returns meep frequencies (a = 1, c = 1), ascending, one per band --
    always exactly ``num_bands`` of them, in order, with no gaps. That
    completeness is the point: band n is eigenvalue n, so there is no
    detection step that can silently renumber the bands.
    """
    if polarization not in ("tm", "te"):
        raise ValueError(f"polarization must be 'tm' or 'te', got {polarization!r}")
    if differentiable:
        eps_t = (eps if isinstance(eps, torch.Tensor)
                 else torch.as_tensor(np.asarray(eps, dtype=float)))
        return _solve_torch(eps_t, dx, dy, kx, ky, num_bands, polarization,
                            eps_te=eps_te)
    eps_np = (eps.detach().cpu().numpy() if isinstance(eps, torch.Tensor)
              else np.asarray(eps, dtype=float))
    A, M = _assemble(eps_np, dx, dy, kx, ky, polarization, eps_te=eps_te)
    return _solve_scipy(A, M, num_bands)
