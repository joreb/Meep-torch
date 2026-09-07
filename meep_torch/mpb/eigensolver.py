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
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

__all__ = ["bloch_bands_2d", "bloch_d_matrices", "te_edge_inv_eps",
           "lattice_gradients", "dual_vectors"]


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


def dual_vectors(a1, a2):
    """Reciprocal (dual) basis with ``b_i . a_j = delta_ij`` -- NO 2 pi.

    ``a1``/``a2`` are the cartesian lattice vectors (rows of
    ``ModeSolver.get_lattice()``, already scaled by the cell size). The
    duals are the rows of the inverse of the matrix whose COLUMNS are the
    lattice vectors, which is what turns a derivative in fractional
    coordinates into a cartesian one.
    """
    A = np.array([[a1[0], a2[0]],
                  [a1[1], a2[1]]], dtype=float)   # columns are a1, a2
    B = np.linalg.inv(A)                          # rows are b1, b2
    return B[0], B[1]


def lattice_gradients(a1, a2, nu: int, nv: int, k1: float, k2: float):
    """Cartesian gradient operators (Gx, Gy) on an oblique nu x nv cell.

    The grid is uniform in FRACTIONAL coordinates (u, v), so the forward
    Bloch differences Du, Dv are the same construction as the rectangular
    case with spacings 1/nu, 1/nv. The cartesian gradient is then the
    chain rule through the dual basis:

        d/dx = b1x d/du + b2x d/dv,   d/dy = b1y d/du + b2y d/dv

    which carries the metric -- including the cross term that makes an
    oblique cell different from a sheared rectangle. ``k1``/``k2`` are the
    Bloch vector in meep's RECIPROCAL basis (i.e. what the user puts in
    k_points), so the wrap phases are exp(2 pi i k_i) directly; verified
    that meep's reciprocal_to_cartesian satisfies kc . a_i = 2 pi k_i.

    For a rectangular basis b1 = (1/Lx, 0) and b2 = (0, 1/Ly), so Gx and
    Gy reduce to the plain Dxf and Dyf of :func:`bloch_d_matrices` --
    bitwise, which is what keeps every existing golden unmoved.
    """
    b1, b2 = dual_vectors(a1, a2)
    Iu = sp.identity(nu, format="csr")
    Iv = sp.identity(nv, format="csr")
    su = _roll_matrix(nu, np.exp(2j * np.pi * k1))
    sv = _roll_matrix(nv, np.exp(2j * np.pi * k2))
    # column-major (u fastest) flattening, as the rectangular path uses
    Du = sp.kron(Iv, (su - Iu) * float(nu), format="csr")
    Dv = sp.kron((sv - Iv) * float(nv), Iu, format="csr")
    Gx = (b1[0] * Du + b2[0] * Dv).tocsr()
    Gy = (b1[1] * Du + b2[1] * Dv).tocsr()
    return Gx, Gy, Du, Dv


def _avg_forward(eps: np.ndarray, axis: int) -> np.ndarray:
    """Two-point average of a cell-centred field onto the forward edge.

    Periodic in both axes, matching the Bloch cell: the Yee edge value
    between cell i and i+1 wraps to (last, first) at the boundary.
    """
    return 0.5 * (eps + np.roll(eps, -1, axis=axis))


def te_edge_inv_eps(eps: np.ndarray, eps_te=None):
    """Edge-averaged ``1/eps`` weights of the TE operator, keyed by the E
    COMPONENT each one multiplies -- deliberately NOT by the derivative axis.

    Returns ``(inv_for_ey, inv_for_ex)``.

    The crossing is real and it looks like a bug every time: ``inv_for_ey``
    carries **eps_yy** averaged along **x**. It has to, because d/dx Hz is
    set by E_y and d/dy Hz by E_x, so the x-derivative's edge weight is the
    one E_y sees. Naming these by axis (``inv_x``/``inv_y``) invites the
    next reader to "fix" the apparent swap, and for scalar eps the swap is
    numerically invisible -- it only shows up on a smoothed tensor, which is
    exactly the case nobody tests by hand. Hence component names.

    Shared by :func:`_assemble` and the field reconstruction in
    ``mpb.fields`` so the two cannot drift: E = D/eps on the operator's own
    edges, not on cell centres.
    """
    exx, eyy = (eps, eps) if eps_te is None else eps_te
    exx = np.asarray(exx, dtype=float)
    eyy = np.asarray(eyy, dtype=float)
    return 1.0 / _avg_forward(eyy, axis=0), 1.0 / _avg_forward(exx, axis=1)


def _assemble(eps: np.ndarray, a1, a2, k1: float, k2: float,
              polarization: str, eps_te=None):
    """Operator pair (A, M) for omega^2, on a possibly oblique cell.

    Everything is built in FRACTIONAL coordinates and pushed to cartesian
    through the dual basis, so an oblique cell differs from a rectangular
    one only by the metric g_ij = b_i . b_j picking up an off-diagonal
    term. For a rectangular basis g is diagonal and every expression below
    collapses to the plain Dxf/Dyf form, bitwise.
    """
    nu, nv = eps.shape
    Gx, Gy, Du, Dv = lattice_gradients(a1, a2, nu, nv, k1, k2)
    if polarization == "tm":
        # -laplacian Ez = omega^2 eps Ez.  G^H G is -d2 (PSD), and the
        # metric cross term is already inside Gx/Gy.
        A = (Gx.conj().T @ Gx + Gy.conj().T @ Gy).tocsr()
        M = sp.diags(eps.reshape(-1, order="F"))
        return A, M
    # TE: d/dx((1/eps_yy) d/dx Hz) + d/dy((1/eps_xx) d/dy Hz) = -omega^2 Hz.
    # Note which component goes with which derivative: d/dx Hz is set by
    # E_y and d/dy Hz by E_x, so the x-derivative carries eps_yy and the
    # y-derivative eps_xx. Swapping them is invisible for scalar eps and
    # wrong for a smoothed tensor.
    # 1/eps sits on the edge the derivative lands on, hence the forward
    # average along that same axis.
    b1, b2 = dual_vectors(a1, a2)
    g12 = float(b1 @ b2)
    flat = lambda w: sp.diags(np.asarray(w).reshape(-1, order="F"))
    if abs(g12) <= 1e-12:
        # ORTHOGONAL: keep the staggered edge weights. Each term is its own
        # Gram form, so this is PSD, and it is bitwise the operator every
        # existing golden was measured against.
        inv_for_ey, inv_for_ex = te_edge_inv_eps(eps, eps_te)
        A = (Du.conj().T @ flat(inv_for_ey * float(b1 @ b1)) @ Du
             + Dv.conj().T @ flat(inv_for_ex * float(b2 @ b2)) @ Dv)
    else:
        # OBLIQUE: the manifestly PSD Gram form, with 1/eps at cell
        # centres.
        #
        # The lattice-component form -- g11 Du^H W Du + g22 Dv^H W Dv +
        # g12 (Du^H W Dv + Dv^H W Du) -- is the textbook weak form and it
        # does NOT survive discretization here: the cross term pairs two
        # independent forward differences and is not a Gram form, so the
        # assembled operator loses positive-definiteness outright.
        # Measured on the triangular cell at resolution 32, with the cross
        # weight on the cell corners: min eigenvalue -5.99e+02, and every
        # band came back 0 once the negative eigenvalues were clamped.
        # G^H W G is PSD by construction (min eigenvalue +7e-13 at Gamma)
        # and lands within 2.3% of meep.mpb.
        #
        # The cost is that 1/eps sits at cell centres rather than on the
        # staggered edges, which is a discretization choice, not a
        # correctness one -- and it is why oblique TE is gated at 4% while
        # rectangular TE holds 2.6%.
        W = flat(1.0 / np.asarray(eps, dtype=float))
        A = Gx.conj().T @ W @ Gx + Gy.conj().T @ W @ Gy
    A = ((A + A.conj().T) * 0.5).tocsr()
    return A, sp.identity(nu * nv, format="csr")


def _sorted_freqs(vals: np.ndarray, num_bands: int) -> list[float]:
    """omega^2 eigenvalues -> meep frequencies f = omega / 2pi, ascending.

    The omega = 0 mode genuinely exists at k = 0 (Ez = const for TM) and
    must come back as EXACTLY 0.0, which is what meep.mpb reports and what
    every downstream ``omega == 0`` guard tests. Clamping only the negative
    side is not enough: the eigensolver puts that mode anywhere in
    +-1e-15 * |lambda|max, and +5.3e-15 survives the clamp and becomes
    f = 1.2e-8 -- nonzero, so the guards silently stop firing and a
    band-1 H field comes out as curl(const)/roundoff.

    So zero anything within 1e-12 of the spectrum's own scale first, then
    clamp. Relative to the spectrum, not absolute: these are omega^2 in
    meep units and their magnitude follows the cell size.
    """
    vals = np.sort(np.real(vals))
    if vals.size:
        vals = np.where(np.abs(vals) < 1e-12 * np.abs(vals).max(), 0.0, vals)
    vals = np.clip(vals, 0.0, None)
    return [float(np.sqrt(v) / (2 * np.pi)) for v in vals[:num_bands]]


def _sigma_for(target_freq: float) -> float:
    """Shift-invert target in omega^2.

    ``target_freq`` 0 means "the lowest bands": shift just below zero,
    because at k = 0 the TM operator is singular (the omega = 0 mode) and
    factorizing at sigma = 0 exactly would fail. A nonzero target is MPB's
    targeted eigensolver -- eigsh with that sigma and which="LM" returns
    the k eigenvalues NEAREST it, which is how a defect state inside a gap
    is found without computing every band below it.
    """
    # Threshold rather than truthiness. `if not target_freq` happens to be
    # correct today -- allow_negative_epsilon() sets target_freq = 1/inf,
    # and 1/inf is 0.0 exactly in Python, not a denormal -- but a target
    # far below any representable band frequency can only mean "lowest",
    # and shift-inverting at sigma ~ 0 would factorize the singular Gamma
    # operator. Cheap guard, and it states the intent.
    return (-1e-7 if float(target_freq) <= 1e-12
            else (2 * np.pi * float(target_freq)) ** 2)


def _is_targeted(target_freq) -> bool:
    return float(target_freq) > 1e-12


def _pick(vals, num_bands: int, target_freq: float):
    """Indices of the wanted eigenvalues, always returned in ascending
    frequency order so band n is still entry n."""
    if _is_targeted(target_freq):
        sel = np.argsort(np.abs(vals - _sigma_for(target_freq)))[:num_bands]
        return sel[np.argsort(vals[sel])]
    return np.argsort(vals)[:num_bands]


def cell_measure(a1, a2, nu: int, nv: int) -> float:
    """Area per grid point: |a1 x a2| / (nu nv).

    Reduces to dx*dy on a rectangular cell. It is the quadrature weight
    every integral over the cell uses, so it belongs in one place.
    """
    return abs(a1[0] * a2[1] - a1[1] * a2[0]) / (nu * nv)


def _normalize(vecs: np.ndarray, eps: np.ndarray, measure: float,
               polarization: str) -> np.ndarray:
    """MPB's normalisation: the energy integral over the cell is 1.

    TM carries Ez, so that is ``integral eps |E|^2 = 1``; TE carries Hz and
    mu = 1, so ``integral |H|^2 = 1``. Both make compute_field_energy's
    total come out at 1, which is the invariant every field test leans on.
    """
    w = (eps.reshape(-1, order="F")[:, None] if polarization == "tm" else 1.0)
    nrm = np.sqrt((np.abs(vecs) ** 2 * w).sum(axis=0) * measure)
    return vecs / np.where(nrm == 0, 1.0, nrm)[None, :]


def _solve_scipy(A, M, num_bands: int, return_vectors: bool = False,
                 target_freq: float = 0.0):
    n = A.shape[0]
    if num_bands >= n - 1:
        Md = (M.toarray() if sp.issparse(M) else np.asarray(M))
        # eigh, not eigvalsh: the generalized form A v = lam M v keeps the
        # vectors M-orthonormal. The old np.linalg.solve(M, A) route formed
        # a non-Hermitian product and could not return usable vectors.
        vals, vecs = sla.eigh(A.toarray(), Md)
    else:
        out = spl.eigsh(A.astype(complex), k=num_bands, M=M.astype(complex),
                        sigma=_sigma_for(target_freq), which="LM",
                        return_eigenvectors=return_vectors)
        vals, vecs = out if return_vectors else (out, None)
    sel = _pick(np.real(vals), num_bands, target_freq)
    freqs = _sorted_freqs(np.real(vals)[sel], num_bands)
    return freqs, (None if vecs is None else np.asarray(vecs)[:, sel])


def _solve_torch(eps_t: torch.Tensor, a1, a2, k1: float, k2: float,
                 num_bands: int, polarization: str,
                 eps_te=None, return_vectors: bool = False,
                 target_freq: float = 0.0):
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
    nu, nv = eps_t.shape
    Gx_s, Gy_s, Du_s, Dv_s = lattice_gradients(a1, a2, nu, nv, k1, k2)
    dense = lambda m: torch.as_tensor(m.toarray(), dtype=torch.complex128)
    eps_c = eps_t.to(torch.complex128)
    if polarization == "tm":
        # A is eps-independent; the gradient enters through eps^{-1/2} only
        Gx, Gy = dense(Gx_s), dense(Gy_s)
        A = Gx.conj().T @ Gx + Gy.conj().T @ Gy
        d = torch.sqrt(eps_c.T.reshape(-1))
        B = A / d[:, None] / d[None, :]
    else:
        # same lattice-component form as _assemble: a single weight cannot
        # sit on Gx, which mixes u- and v-edges
        b1, b2 = dual_vectors(a1, a2)
        g12 = float(b1 @ b2)
        if abs(g12) <= 1e-12:
            Du, Dv = dense(Du_s), dense(Dv_s)
            exx, eyy = ((eps_c, eps_c) if eps_te is None else
                        tuple(torch.as_tensor(e).to(torch.complex128)
                              for e in eps_te))
            w_u = (1.0 / (0.5 * (eyy + torch.roll(eyy, -1, 0)))).T.reshape(-1)
            w_v = (1.0 / (0.5 * (exx + torch.roll(exx, -1, 1)))).T.reshape(-1)
            B = (Du.conj().T @ ((float(b1 @ b1) * w_u)[:, None] * Du)
                 + Dv.conj().T @ ((float(b2 @ b2) * w_v)[:, None] * Dv))
        else:
            # Gram form; see the note in _assemble for why the
            # lattice-component form is not usable off-rectangle
            Gx, Gy = dense(Gx_s), dense(Gy_s)
            w = (1.0 / eps_c).T.reshape(-1)
            B = (Gx.conj().T @ (w[:, None] * Gx)
                 + Gy.conj().T @ (w[:, None] * Gy))
    B = 0.5 * (B + B.conj().T)
    if not return_vectors:
        vals = torch.linalg.eigvalsh(B)
        sel = _pick(vals.detach().cpu().numpy(), num_bands, target_freq)
        vals = vals[torch.as_tensor(sel.copy())]
        return torch.sqrt(torch.clamp(vals, min=0.0)) / (2 * np.pi), None
    vals, y = torch.linalg.eigh(B)
    sel = _pick(vals.detach().cpu().numpy(), num_bands, target_freq)
    idx = torch.as_tensor(sel.copy())
    vals, y = vals[idx], y[:, idx]
    if polarization == "tm":
        # B = D^-1 A D^-1 with D = sqrt(eps), so an eigenvector y of B maps
        # back to the physical Ez as v = y / sqrt(eps). Getting this wrong
        # is NOT caught by the normalisation test -- a wrongly scaled field
        # still normalises to 1 -- only by the Faraday residual, which is
        # the one check that couples the vector back to the operator.
        d = torch.sqrt(eps_c.T.reshape(-1))
        y = y / d[:, None]
    freqs = torch.sqrt(torch.clamp(vals, min=0.0)) / (2 * np.pi)
    return freqs, y.detach().cpu().numpy()


def bloch_bands_2d(eps, dx: float, dy: float, kx: float, ky: float,
                   num_bands: int, polarization: str = "tm",
                   differentiable: bool = False, eps_te=None,
                   return_vectors: bool = False, target_freq: float = 0.0,
                   lattice=None, k_frac=None):
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
    ``lattice``: ``(a1, a2)`` cartesian lattice vectors, with ``k_frac``
    the Bloch vector in the reciprocal basis. Pass these for an OBLIQUE
    cell; without them the ``dx/dy/kx/ky`` arguments describe a
    rectangular one and are converted to the same form (bitwise -- the
    lattice path reduces exactly, which is what leaves every existing
    golden unmoved).
    ``target_freq``: MPB's targeted eigensolver. 0 gives the lowest bands;
    nonzero returns the ``num_bands`` modes whose frequency is NEAREST it,
    which finds a defect state inside a gap without computing the hundreds
    of bands below it.
    ``return_vectors``: also return the mode fields as
    ``(nx*ny, num_bands)`` complex, column-major (x fastest, the same
    flattening the operator uses), normalised to MPB's convention
    (integral of eps|E|^2 = 1 for TM, of |H|^2 = 1 for TE) and ordered to
    match the frequencies. Vectors are always DETACHED, on both paths:
    the eigenvalues carry gradient, the eigenvectors do not.

    Returns meep frequencies (a = 1, c = 1), ascending, one per band --
    always exactly ``num_bands`` of them, in order, with no gaps. That
    completeness is the point: band n is eigenvalue n, so there is no
    detection step that can silently renumber the bands.
    """
    if polarization not in ("tm", "te"):
        raise ValueError(f"polarization must be 'tm' or 'te', got {polarization!r}")
    eps_np = (eps.detach().cpu().numpy() if isinstance(eps, torch.Tensor)
              else np.asarray(eps, dtype=float))
    nu, nv = eps_np.shape
    if lattice is None:
        # rectangular convenience form -> the same lattice description
        a1, a2 = (nu * dx, 0.0), (0.0, nv * dy)
        k1 = kx * (nu * dx) / (2 * np.pi)
        k2 = ky * (nv * dy) / (2 * np.pi)
    else:
        a1, a2 = lattice
        k1, k2 = (0.0, 0.0) if k_frac is None else k_frac
    if differentiable:
        eps_t = (eps if isinstance(eps, torch.Tensor)
                 else torch.as_tensor(eps_np))
        freqs, vecs = _solve_torch(eps_t, a1, a2, k1, k2, num_bands,
                                   polarization, eps_te=eps_te,
                                   return_vectors=return_vectors,
                                   target_freq=target_freq)
    else:
        A, M = _assemble(eps_np, a1, a2, k1, k2, polarization, eps_te=eps_te)
        freqs, vecs = _solve_scipy(A, M, num_bands,
                                   return_vectors=return_vectors,
                                   target_freq=target_freq)
    if not return_vectors:
        return freqs
    return freqs, _normalize(vecs, eps_np, cell_measure(a1, a2, nu, nv),
                             polarization)
