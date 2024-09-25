# SPDX-License-Identifier: MIT
# Ported from tidy3d (https://github.com/flexcompute/tidy3d), components/mode
# solver, historic MIT-licensed revision. Copyright (c) Flexcompute Inc. --
# MIT License (ceviche-derived, (c) 2019 Tyler Hughes).
# Adaptations (c) 2026 meep-torch contributors, MIT.
"""Full-tensor 2D FD mode solver for off-diagonal eps/mu (angled waveguides).

Plain-function port of the tensorial path of ``tidy3d.components.mode.solver.
EigSolver.solver_tensorial`` (MIT, ceviche-derived). Whereas the diagonal solver
(``fd2d.solve_2d``) eliminates Hz/Ez to reach a (2N, 2N) eigenproblem with
eigenvalue -(neff + 1j*keff)**2, the tensorial solver keeps all four tangential
components (Ex, Ey, Hx, Hy) and assembles a (4N, 4N) eigenproblem whose
eigenvalues are directly (neff + 1j*keff). It handles off-diagonal eps/mu
entries that arise from coordinate transforms (angled waveguides).

The eps tensor is taken at Yee positions built by the same two-point edge-
replicated averaging used in ``fd2d._yee_eps``, so a diagonal input reproduces
``fd2d.solve_2d`` to machine precision (diagonal entries: eps[0,0] averaged
along x, eps[1,1] along y, eps[2,2] at the cell center; off-diagonal entries
at the union of the two component axes' Yee positions).

Boundary-DOF elimination: the historic ``solver_tensorial`` does NOT perform
incidence-matrix DOF elimination (unlike ``solver_diagonal``); this port
mirrors that. Recorded: no elimination, ported as-is. No spurious boundary
modes were observed in the task-9 tests.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from .derivatives import C_0, create_d_matrices as d_mats, create_s_matrices as s_mats
from .fd2d import (
    ETA_0,
    Modes2D,
    TARGET_SHIFT,
    _apply_max_pmc,
    _avg_x,
    _avg_y,
    _normalize_modes,
    _orthogonalize_degenerate,
    _set_initial_vec,
    fp_eps,
)
from .transforms import angled_eps_tensor

TOL_EIGS = fp_eps / 10


# --------------------------------------------------------------------------- #
# Initial vector for the (4N, 4N) tensorial eigenproblem (len_multiplier = 4).
# --------------------------------------------------------------------------- #
def _set_initial_vec_tensorial(Nx: int, Ny: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    vec = rng.random((Nx, Ny, 4)) + 1j * rng.random((Nx, Ny, 4))
    if Nx > 1:
        vec[0, :, :] = 0
    if Ny > 1:
        vec[:, 0, :] = 0
    return np.vstack(vec).flatten("F")


# --------------------------------------------------------------------------- #
# Place an eps tensor component at its Yee position by two-point averaging.
# eps[i, j] sits at the union of the i- and j-component Yee axes: average
# along x if either index is 0 (Ex axis), along y if either is 1 (Ey axis).
# Diagonal entries reduce to _avg_x / _avg_y / raw, matching fd2d._yee_eps.
# --------------------------------------------------------------------------- #
def _yee_avg(i: int, j: int, comp: np.ndarray) -> np.ndarray:
    out = comp
    if i == 0 or j == 0:
        out = _avg_x(out)
    if i == 1 or j == 1:
        out = _avg_y(out)
    # TODO(T9 deferred / v1): the off-diagonal cross-term paths (e.g. [0,1] /
    # [1,0], the eps_xy / eps_yx Yee components) feed the tensorial solver and
    # are NOT differentiated by the diagonal eigengrad backward, which handles
    # only eps_xx / eps_yy / eps_zz. Full-tensor (anisotropic) autograd is left
    # as future work; until then gradient inputs must stay diagonal.
    return out


def _yee_eps_tensor(eps_tensor: np.ndarray) -> np.ndarray:
    """eps_tensor (3, 3, Nx, Ny) -> (3, 3, N) complex128 at Yee positions."""
    Nx, Ny = eps_tensor.shape[2], eps_tensor.shape[3]
    N = Nx * Ny
    out = np.zeros((3, 3, N), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            out[i, j, :] = _yee_avg(i, j, eps_tensor[i, j]).ravel()
    return out


# --------------------------------------------------------------------------- #
# Tensorial solver core (ported from EigSolver.solver_tensorial).
# --------------------------------------------------------------------------- #
def _solver_tensorial(eps, mu, der_mats, num_modes, neff_guess, vec_init, direction="+"):
    """Solve the (4N, 4N) full-tensor eigenproblem.

    ``eps`` / ``mu`` are (3, 3, N) arrays (Yee-position eps/mu, already averaged).
    Eigenvalues of the assembled matrix are 1j*(neff + 1j*keff); multiplying by
    -1j makes them (neff + 1j*keff), so neff = Re(eig), keff = Im(eig).
    """
    N = eps.shape[-1]
    dxf, dxb, dyf, dyb = der_mats

    inv_eps_zz = sp.spdiags(1 / eps[2, 2, :], 0, N, N)
    inv_mu_zz = sp.spdiags(1 / mu[2, 2, :], 0, N, N)
    axax = -dxf.dot(sp.spdiags(eps[2, 0, :] / eps[2, 2, :], 0, N, N)) - sp.spdiags(
        mu[1, 2, :] / mu[2, 2, :], 0, N, N
    ).dot(dyf)
    axay = -dxf.dot(sp.spdiags(eps[2, 1, :] / eps[2, 2, :], 0, N, N)) + sp.spdiags(
        mu[1, 2, :] / mu[2, 2, :], 0, N, N
    ).dot(dxf)
    axbx = -dxf.dot(inv_eps_zz).dot(dyb) + sp.spdiags(
        mu[1, 0, :] - mu[1, 2, :] * mu[2, 0, :] / mu[2, 2, :], 0, N, N
    )
    axby = dxf.dot(inv_eps_zz).dot(dxb) + sp.spdiags(
        mu[1, 1, :] - mu[1, 2, :] * mu[2, 1, :] / mu[2, 2, :], 0, N, N
    )
    ayax = -dyf.dot(sp.spdiags(eps[2, 0, :] / eps[2, 2, :], 0, N, N)) + sp.spdiags(
        mu[0, 2, :] / mu[2, 2, :], 0, N, N
    ).dot(dyf)
    ayay = -dyf.dot(sp.spdiags(eps[2, 1, :] / eps[2, 2, :], 0, N, N)) - sp.spdiags(
        mu[0, 2, :] / mu[2, 2, :], 0, N, N
    ).dot(dxf)
    aybx = -dyf.dot(inv_eps_zz).dot(dyb) + sp.spdiags(
        -mu[0, 0, :] + mu[0, 2, :] * mu[2, 0, :] / mu[2, 2, :], 0, N, N
    )
    ayby = dyf.dot(inv_eps_zz).dot(dxb) + sp.spdiags(
        -mu[0, 1, :] + mu[0, 2, :] * mu[2, 1, :] / mu[2, 2, :], 0, N, N
    )
    bxbx = -dxb.dot(sp.spdiags(mu[2, 0, :] / mu[2, 2, :], 0, N, N)) - sp.spdiags(
        eps[1, 2, :] / eps[2, 2, :], 0, N, N
    ).dot(dyb)
    bxby = -dxb.dot(sp.spdiags(mu[2, 1, :] / mu[2, 2, :], 0, N, N)) + sp.spdiags(
        eps[1, 2, :] / eps[2, 2, :], 0, N, N
    ).dot(dxb)
    bxax = -dxb.dot(inv_mu_zz).dot(dyf) + sp.spdiags(
        eps[1, 0, :] - eps[1, 2, :] * eps[2, 0, :] / eps[2, 2, :], 0, N, N
    )
    bxay = dxb.dot(inv_mu_zz).dot(dxf) + sp.spdiags(
        eps[1, 1, :] - eps[1, 2, :] * eps[2, 1, :] / eps[2, 2, :], 0, N, N
    )
    bybx = -dyb.dot(sp.spdiags(mu[2, 0, :] / mu[2, 2, :], 0, N, N)) + sp.spdiags(
        eps[0, 2, :] / eps[2, 2, :], 0, N, N
    ).dot(dyb)
    byby = -dyb.dot(sp.spdiags(mu[2, 1, :] / mu[2, 2, :], 0, N, N)) - sp.spdiags(
        eps[0, 2, :] / eps[2, 2, :], 0, N, N
    ).dot(dxb)
    byax = -dyb.dot(inv_mu_zz).dot(dyf) + sp.spdiags(
        -eps[0, 0, :] + eps[0, 2, :] * eps[2, 0, :] / eps[2, 2, :], 0, N, N
    )
    byay = dyb.dot(inv_mu_zz).dot(dxf) + sp.spdiags(
        -eps[0, 1, :] + eps[0, 2, :] * eps[2, 1, :] / eps[2, 2, :], 0, N, N
    )

    mat = sp.bmat(
        [
            [axax, axay, axbx, axby],
            [ayax, ayay, aybx, ayby],
            [bxax, bxay, bxbx, bxby],
            [byax, byay, bybx, byby],
        ]
    )

    # Eigenvalues are 1j*(neff + 1j*keff); multiply by -1j so they become
    # (neff + 1j*keff) and neff = Re(eig), keff = Im(eig).
    mat = mat * (-1j)

    # Flip sign for backward direction (phase-constant convention).
    if direction == "-":
        mat = mat * -1

    # Tensorial matrix is always complex (eigenvalue is complex even for real
    # eps); cast to complex128.
    mat = mat.astype(np.complex128)
    vec_init = vec_init.astype(np.complex128)
    eig_guess = np.array([neff_guess], dtype=np.complex128)[0]

    import scipy.sparse.linalg as spl

    vals, vecs = spl.eigs(mat, k=num_modes, sigma=eig_guess, tol=TOL_EIGS, v0=vec_init)
    neff = np.real(vals)
    keff = np.imag(vals)

    sort_inds = np.argsort(neff)[::-1]
    neff = neff[sort_inds]
    keff = keff[sort_inds]
    vecs = vecs[:, sort_inds]

    Ex = vecs[:N, :]
    Ey = vecs[N : 2 * N, :]
    Hx = vecs[2 * N : 3 * N, :]
    Hy = vecs[3 * N :, :]

    # Reconstruct Hz, Ez from the tangential components.
    hxy_term = (-mu[2, 0, :] * Hx.T - mu[2, 1, :] * Hy.T).T
    Hz = inv_mu_zz.dot(dxf.dot(Ey) - dyf.dot(Ex) + hxy_term)
    exy_term = (-eps[2, 0, :] * Ex.T - eps[2, 1, :] * Ey.T).T
    Ez = inv_eps_zz.dot(dxb.dot(Hy) - dyb.dot(Hx) + exy_term)

    E = np.stack((Ex, Ey, Ez), axis=0)
    H = np.stack((Hx, Hy, Hz), axis=0)

    # Return H to standard units (same convention as the diagonal solver).
    H *= -1j / ETA_0

    return E, H, neff, keff


def solve_2d_tensor(
    eps_tensor,
    dx: float,
    f: float,
    k: int = 2,
    npml: tuple[int, int] = (0, 0),
    dmin_pmc: tuple[bool, bool] = (False, False),
    target_neff: float | None = None,
) -> Modes2D:
    """Full-tensor 2D FD mode solver (off-diagonal eps/mu supported).

    ``eps_tensor`` is the cell-center permittivity tensor, shape ``(3, 3, Nx,
    Ny)``; components are placed at their Yee positions by two-point averaging
    before assembly. ``dx`` is the uniform grid step (um); ``f`` is the spatial
    frequency in 1/um (so ``k0 = 2*pi*f``, matching ``fd2d.solve_2d``). Returns
    the ``k`` modes sorted by descending Re(neff), with
    ``neff = neff_re + 1j*keff``. PEC on all walls by default; set ``dmin_pmc``
    to impose PMC at the low-index side of an axis (the high-index side mirrors
    it via ``_apply_max_pmc``, matching ``solve_2d``). For diagonal input this
    reproduces ``solve_2d`` to machine precision.
    """
    eps_np = np.asarray(eps_tensor).astype(np.complex128)
    Nx, Ny = eps_np.shape[2], eps_np.shape[3]
    N = Nx * Ny
    k0 = 2 * np.pi * f
    omega = k0 * C_0

    eps = _yee_eps_tensor(eps_np)  # (3, 3, N) at Yee positions
    mu = np.zeros((3, 3, N), dtype=np.complex128)
    for i in range(3):
        mu[i, i, :] = 1.0

    dl_f = (np.full(Nx, dx), np.full(Ny, dx))
    dl_b = (np.full(Nx, dx), np.full(Ny, dx))
    dls = (dl_f, dl_b)

    der_mats_tmp = d_mats((Nx, Ny), dls, dmin_pmc)
    der_mats_tmp = _apply_max_pmc(der_mats_tmp, (Nx, Ny), dmin_pmc)
    pml_mats = s_mats(omega, (Nx, Ny), npml, dls, eps, mu,
                      dmin_pml=(not dmin_pmc[0], not dmin_pmc[1]))
    der_mats = [S.dot(D) / k0 for S, D in zip(pml_mats, der_mats_tmp)]

    if target_neff is None:
        n_max = float(np.sqrt(np.max(np.abs(eps_np[2, 2, :, :]))))
        target = n_max
    else:
        target = float(target_neff)
    target_neff_p = target
    if abs(TARGET_SHIFT) > abs(target_neff_p * TARGET_SHIFT):
        target_neff_p += TARGET_SHIFT
    else:
        target_neff_p *= 1 + TARGET_SHIFT

    vec_init = _set_initial_vec_tensorial(Nx, Ny)
    E, H, neff, keff = _solver_tensorial(
        eps, mu, der_mats, k, target_neff_p, vec_init, direction="+"
    )

    # The tensorial eigenproblem does not apply the power-flow direction flip
    # (the historic dispatch skips it for tensorial modes); only normalize and
    # orthogonalize degenerate groups.
    E = E.reshape((3, Nx, Ny, k))
    H = H.reshape((3, Nx, Ny, k))
    E, H, self_orth = _normalize_modes(E, H, dl_f, dl_b, list(range(k)),
                                       1e2 * TOL_EIGS)
    E, H = _orthogonalize_degenerate(E, H, neff, keff, dl_f, dl_b, self_orth)

    neff_t = torch.from_numpy((neff + 1j * keff).astype(np.complex128))
    Ex_t = torch.from_numpy(np.ascontiguousarray(E[0]).astype(np.complex128))
    Ey_t = torch.from_numpy(np.ascontiguousarray(E[1]).astype(np.complex128))
    return Modes2D(neff=neff_t, Ex=Ex_t, Ey=Ey_t)


def solve_angled(eps_cc, dx: float, f: float, theta: float, phi: float = 0.0,
                k: int = 2, **kw):
    """Solve for the modes of a waveguide tilted by ``theta`` (radians) about
    the propagation axis (azimuth ``phi``). Builds the angled eps tensor via
    ``angled_eps_tensor`` and delegates to ``solve_2d_tensor``. Extra keyword
    arguments (``npml``, ``dmin_pmc``, ``target_neff``) are forwarded.
    """
    eps_tensor = angled_eps_tensor(eps_cc, theta, phi)
    return solve_2d_tensor(eps_tensor, dx=dx, f=f, k=k, **kw)
