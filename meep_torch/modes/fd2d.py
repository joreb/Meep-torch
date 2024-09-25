# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Flexcompute
# Copyright (c) 2019 Tyler Hughes
# Derived from the ceviche package (https://github.com/fancompute/ceviche).
# Adaptations (c) 2026 meep-torch contributors, MIT.
# Full-vector 2D finite-difference mode solver (the "diagonal" path of tidy3d's
# EigSolver). Ported from tidy3d.components.mode.solver (MIT, per-file LICENSE in
# that directory) and tidy3d.components.data.utils (MIT). Pydantic / tensorial /
# angle / bend / incidence / basis-field paths stripped; plain functions only.
# eps is taken at cell centers; the three Yee-position eps components used by the
# diagonal matrix are built by two-point averaging (same convention as Phase 1
# TE coeffs, edge-replicated to the (Nx,Ny) solver grid).

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

from .derivatives import C_0, ETA_0, create_d_matrices as d_mats, create_s_matrices as s_mats
from .transforms import bend_dwdz

# Numerical tolerances (ported from tidy3d.components.mode.solver).
fp_eps = np.finfo(np.float32).eps
TOL_COMPLEX = 1e-10
TOL_EIGS = fp_eps / 10
TOL_DEGENERATE_CANDIDATE = 1e3 * TOL_EIGS
TOL_NEEDS_ORTHOGONALIZATION = 1e2 * TOL_EIGS
TOL_POWER_FLOW_DIRECTION = 1e2 * TOL_EIGS
TARGET_SHIFT = 10 * fp_eps
pec_val = -1e8
GOOD_CONDUCTOR_THRESHOLD = 0.9
GOOD_CONDUCTOR_CUT_OFF = 1e70


class Modes2D(NamedTuple):
    neff: torch.Tensor  # (k,) complex128
    Ex: torch.Tensor  # (k, Nx, Ny) complex128
    Ey: torch.Tensor  # (k, Nx, Ny) complex128


# --------------------------------------------------------------------------- #
# Yee-position eps from cell-center eps (two-point averaging, edge-replicated).
# --------------------------------------------------------------------------- #
def _avg_x(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    out[:-1] = 0.5 * (v[:-1] + v[1:])
    out[-1] = v[-1]
    return out


def _avg_y(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    out[:, :-1] = 0.5 * (v[:, :-1] + v[:, 1:])
    out[:, -1] = v[:, -1]
    return out


def _yee_eps(eps_cc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps_zz = eps_cc
    eps_xx = _avg_x(eps_cc)
    eps_yy = _avg_y(eps_cc)
    return eps_xx, eps_yy, eps_zz


# --------------------------------------------------------------------------- #
# Overlap / power-flow helpers (ported from tidy3d.components.data.utils, MIT).
# All operate on tangential field tuples (Eu, Ev) / (Hu, Hv) with the mode axis
# leading; dS = (dS_EuHv, dS_EvHu) are (nu, nv) area elements.
# --------------------------------------------------------------------------- #
def _complex_power_flow(E, H, dS):
    dS_EuHv, dS_EvHu = dS
    Eu, Ev = E
    Hu, Hv = H
    return 0.5 * np.sum((Eu * np.conj(Hv)) * dS_EuHv - (Ev * np.conj(Hu)) * dS_EvHu, axis=(-2, -1))


def _dot(E1, H1, E2, H2, dS, conjugate=False):
    dS_EuHv, dS_EvHu = dS
    E1u, E1v = E1
    H1u, H1v = H1
    E2u, E2v = E2
    H2u, H2v = H2
    if conjugate:
        E1u, E1v, H1u, H1v = np.conj(E1u), np.conj(E1v), np.conj(H1u), np.conj(H1v)
    dS0 = np.asarray(dS_EuHv).reshape(-1)
    dS1 = np.asarray(dS_EvHu).reshape(-1)
    result = 0.25 * ((E1u.reshape(-1) * dS0) @ H2v.reshape(-1))
    result += 0.25 * ((H1v.reshape(-1) * dS0) @ E2u.reshape(-1))
    result -= 0.25 * ((E1v.reshape(-1) * dS1) @ H2u.reshape(-1))
    result -= 0.25 * ((H1u.reshape(-1) * dS1) @ E2v.reshape(-1))
    return result


def _outer_dot(E1, H1, E2, H2, dS, conjugate=False):
    # Pairwise overlap matrix. Mode axis is -3; spatial axes -2,-1.
    dS_EuHv, dS_EvHu = dS
    E1u, E1v = E1
    H1u, H1v = H1
    E2u, E2v = E2
    H2u, H2v = H2
    if conjugate:
        E1u, E1v, H1u, H1v = np.conj(E1u), np.conj(E1v), np.conj(H1u), np.conj(H1v)
    n1 = E1u.shape[-3]
    n2 = E2u.shape[-3]
    dS0 = np.asarray(dS_EuHv).reshape(-1)
    dS1 = np.asarray(dS_EvHu).reshape(-1)
    a1 = E1u.reshape(n1, -1)
    a2 = H2v.reshape(n2, -1)
    b1 = H1v.reshape(n1, -1)
    b2 = E2u.reshape(n2, -1)
    c1 = E1v.reshape(n1, -1)
    c2 = H2u.reshape(n2, -1)
    d1 = H1u.reshape(n1, -1)
    d2 = E2v.reshape(n2, -1)
    S = 0.25 * ((a1 * dS0) @ a2.T + (b1 * dS0) @ b2.T
                - (c1 * dS1) @ c2.T - (d1 * dS1) @ d2.T)
    return S


def _cauchy_schwarz_bound(E, H, dl_primal, dl_dual, mode_idx):
    Ex, Ey, Ez = E[0], E[1], E[2]
    Hx, Hy, Hz = H[0], H[1], H[2]
    Ex_Hy_dS = np.outer(dl_primal[0], dl_dual[1])
    Ey_Hx_dS = np.outer(dl_dual[0], dl_primal[1])
    Ez_dS = np.outer(dl_dual[0], dl_dual[1])
    Hz_dS = np.outer(dl_primal[0], dl_primal[1])
    E_int = 0.5 * (np.sum(np.abs(Ex[..., mode_idx]) ** 2 * Ex_Hy_dS)
                   + np.sum(np.abs(Ey[..., mode_idx]) ** 2 * Ey_Hx_dS)
                   + np.sum(np.abs(Ez[..., mode_idx]) ** 2 * Ez_dS))
    H_int = 0.5 * (np.sum(np.abs(Hx[..., mode_idx]) ** 2 * Ey_Hx_dS)
                   + np.sum(np.abs(Hy[..., mode_idx]) ** 2 * Ex_Hy_dS)
                   + np.sum(np.abs(Hz[..., mode_idx]) ** 2 * Hz_dS))
    return np.sqrt(E_int * H_int)


def _normalize_modes(E, H, dl_primal, dl_dual, mode_indices, self_orth_tol):
    self_orthogonal = set()
    dS = (np.outer(dl_primal[0], dl_dual[1]), np.outer(dl_dual[0], dl_primal[1]))
    for mi in mode_indices:
        E1 = (E[0, ..., mi], E[1, ..., mi])
        H1 = (H[0, ..., mi], H[1, ..., mi])
        self_dot = _dot(E1, H1, E1, H1, dS)
        ub = _cauchy_schwarz_bound(E, H, dl_primal, dl_dual, mi)
        if (np.abs(self_dot) < ub * self_orth_tol
                or np.any(np.isnan(E[..., mi])) or np.any(np.isnan(H[..., mi]))):
            self_orthogonal.add(mi)
            continue
        nf = 1.0 / np.sqrt(self_dot)
        E[..., mi] *= nf
        H[..., mi] *= nf
    return E, H, self_orthogonal


def _identify_degenerate(n_complex, tol):
    num = len(n_complex)
    ungrouped = set(range(num))
    sets = []
    while ungrouped:
        seed = ungrouped.pop()
        cur = {seed}
        to_remove = set()
        for mi in ungrouped:
            if np.isclose(n_complex[mi], n_complex[seed], rtol=tol, atol=tol):
                cur.add(mi)
                to_remove.add(mi)
        ungrouped -= to_remove
        if len(cur) >= 2:
            sets.append(cur)
    return sets


def _make_orthogonal_basis(deg_sets, E_vec, H_vec, dl_primal, dl_dual):
    dS = (np.outer(dl_primal[0], dl_dual[1]), np.outer(dl_dual[0], dl_primal[1]))
    for group in deg_sets:
        idx = sorted(group)
        E_sel = np.moveaxis(E_vec[:, :, :, idx], -1, 1)
        H_sel = np.moveaxis(H_vec[:, :, :, idx], -1, 1)
        E_tan = (E_sel[0], E_sel[1])
        H_tan = (H_sel[0], H_sel[1])
        S = _outer_dot(E_tan, H_tan, E_tan, H_tan, dS)
        eigvals, eigvecs = np.linalg.eig(S)
        for i in range(eigvecs.shape[1]):
            vec = eigvecs[:, i]
            sd = np.dot(vec, vec)
            if np.isclose(sd, 0, atol=np.finfo(E_vec.dtype).tiny):
                raise ValueError(f"Eigenvector {i} self-orthogonal.")
            eigvecs[:, i] = vec / np.sqrt(sd)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        E_vec[..., idx] = E_vec[..., idx] @ W
        H_vec[..., idx] = H_vec[..., idx] @ W
    return E_vec, H_vec


def _orthogonalize_degenerate(E, H, neff, keff, dl_primal, dl_dual, self_orthogonal):
    deg = _identify_degenerate(neff + keff * 1j, TOL_DEGENERATE_CANDIDATE)
    deg = [s - self_orthogonal for s in deg if len(s - self_orthogonal) > 1]
    # Filter by significant mutual overlap.
    dS = (np.outer(dl_primal[0], dl_dual[1]), np.outer(dl_dual[0], dl_primal[1]))
    filtered = []
    for group in deg:
        idx = sorted(group)
        E_sel = np.moveaxis(E[:, :, :, idx], -1, 1)
        H_sel = np.moveaxis(H[:, :, :, idx], -1, 1)
        S = _outer_dot((E_sel[0], E_sel[1]), (H_sel[0], H_sel[1]),
                       (E_sel[0], E_sel[1]), (H_sel[0], H_sel[1]), dS)
        adj = (np.abs(S) > TOL_NEEDS_ORTHOGONALIZATION).astype(int)
        np.fill_diagonal(adj, 0)
        _, labels = sp.csgraph.connected_components(sp.csr_matrix(adj), directed=False)
        groups: dict[int, list[int]] = {}
        for li, gid in enumerate(labels):
            groups.setdefault(gid, []).append(idx[li])
        for g in groups.values():
            if len(g) >= 2:
                filtered.append(set(g))
    return _make_orthogonal_basis(filtered, E, H, dl_primal, dl_dual)


# --------------------------------------------------------------------------- #
# Eigensolver wrapper (shift-invert).
# --------------------------------------------------------------------------- #
def _solver_eigs(mat, num_modes, vec_init, guess_value, M=None):
    return spl.eigs(mat, k=num_modes, sigma=guess_value, tol=TOL_EIGS, v0=vec_init, M=M)


def _eigs_to_neff(eig_list):
    sqrt_eig = np.emath.sqrt(-eig_list + 0j)
    return sqrt_eig.real, sqrt_eig.imag


def _is_complex(x, tol=TOL_COMPLEX):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x.imag) / (np.linalg.norm(x) + fp_eps) > tol
    return spl.norm(x.imag) / (spl.norm(x) + fp_eps) > tol


def _set_initial_vec(Nx, Ny):
    rng = np.random.default_rng(0)
    vec = rng.random((Nx, Ny, 2)) + 1j * rng.random((Nx, Ny, 2))
    if Nx > 1:
        vec[0, :, :] = 0
    if Ny > 1:
        vec[:, 0, :] = 0
    return np.vstack(vec).flatten("F")


# --------------------------------------------------------------------------- #
# Diagonal-solver matrix assembly (shared by the solver and the IFT autograd
# path). Returns the (2N, 2N) matrix and a context dict of the pieces needed
# by the implicit-function-theorem VJP (eps_xx/eps_yy/eps_zz Yee diagonals,
# p_mu, p_partial, q_ep, der_mats, inv_eps_zz, inv_mu_zz).
# --------------------------------------------------------------------------- #
def _assemble_diagonal(eps_tensor, mu_tensor, der_mats):
    N = eps_tensor.shape[-1]
    eps_xx = eps_tensor[0, 0, :]
    eps_yy = eps_tensor[1, 1, :]
    eps_zz = eps_tensor[2, 2, :]
    mu_xx = mu_tensor[0, 0, :]
    mu_yy = mu_tensor[1, 1, :]
    mu_zz = mu_tensor[2, 2, :]
    dxf, dxb, dyf, dyb = der_mats

    inv_eps_zz = sp.spdiags(1 / eps_zz, 0, N, N)
    inv_mu_zz = sp.spdiags(1 / mu_zz, 0, N, N)

    p_mu = sp.bmat([[None, sp.spdiags(mu_yy, 0, N, N)],
                    [sp.spdiags(-mu_xx, 0, N, N), None]])
    p_partial = sp.bmat([
        [-dxf.dot(inv_eps_zz).dot(dyb), dxf.dot(inv_eps_zz).dot(dxb)],
        [-dyf.dot(inv_eps_zz).dot(dyb), dyf.dot(inv_eps_zz).dot(dxb)],
    ])
    q_ep = sp.bmat([[None, sp.spdiags(eps_yy, 0, N, N)],
                    [sp.spdiags(-eps_xx, 0, N, N), None]])
    q_partial = sp.bmat([
        [-dxb.dot(inv_mu_zz).dot(dyf), dxb.dot(inv_mu_zz).dot(dxf)],
        [-dyb.dot(inv_mu_zz).dot(dyf), dyb.dot(inv_mu_zz).dot(dxf)],
    ])
    qmat = q_ep + q_partial
    mat = p_mu @ qmat + p_partial @ q_ep
    ctx = {
        "eps_xx": eps_xx, "eps_yy": eps_yy, "eps_zz": eps_zz,
        "mu_xx": mu_xx, "mu_yy": mu_yy, "mu_zz": mu_zz,
        "inv_eps_zz": inv_eps_zz, "inv_mu_zz": inv_mu_zz,
        "p_mu": p_mu, "p_partial": p_partial, "q_ep": q_ep, "q_partial": q_partial,
        "qmat": qmat, "der_mats": der_mats,
    }
    return mat, ctx


# --------------------------------------------------------------------------- #
# Diagonal solver core (ported from EigSolver.solver_diagonal).
# --------------------------------------------------------------------------- #
def _solver_diagonal(eps_tensor, mu_tensor, der_mats, num_modes, neff_guess, vec_init):
    N = eps_tensor.shape[-1]
    # Phase 2 scope: dielectric eps only. tidy3d's "good conductor" branch
    # (replaces |eps| > 0.9*|pec_val| cells with a PEC sentinel eps and enables
    # a preconditioner) is intentionally NOT ported -- no metals in Phase 2.
    mat, ctx = _assemble_diagonal(eps_tensor, mu_tensor, der_mats)
    dxf, dxb, dyf, dyb = der_mats
    inv_eps_zz = ctx["inv_eps_zz"]
    inv_mu_zz = ctx["inv_mu_zz"]
    qmat = ctx["qmat"]
    q_ep = ctx["q_ep"]

    if _is_complex(eps_tensor) or _is_complex(mu_tensor) or any(_is_complex(f) for f in der_mats):
        mat = mat.astype(np.complex128)
        vec_init = vec_init.astype(np.complex128)
    else:
        mat = mat.astype(np.float64)
        vec_init = vec_init.real.astype(np.float64)

    eig_guess = np.array([-(neff_guess ** 2)])
    eig_guess = eig_guess.astype(mat.dtype)[0]

    vals, vecs = _solver_eigs(mat, num_modes, vec_init, guess_value=eig_guess)
    neff, keff = _eigs_to_neff(vals)

    sort_inds = np.argsort(neff)[::-1]
    neff = neff[sort_inds]
    keff = keff[sort_inds]
    vecs = vecs[:, sort_inds]

    Ex = vecs[:N, :]
    Ey = vecs[N:, :]
    h_field = qmat.dot(vecs)
    Hx = h_field[:N, :] / (1j * neff - keff)
    Hy = h_field[N:, :] / (1j * neff - keff)
    Hz = inv_mu_zz.dot(dxf.dot(Ey) - dyf.dot(Ex))
    h_partial = q_ep.dot(vecs) / (1j * neff - keff)
    Ez = inv_eps_zz.dot(dxb.dot(h_partial[N:, :]) - dyb.dot(h_partial[:N, :]))

    E = np.stack((Ex, Ey, Ez), axis=0)
    H = np.stack((Hx, Hy, Hz), axis=0)
    H *= -1j / ETA_0
    return E, H, neff, keff


def _apply_max_pmc(der_mats, shape, dmax_pmc):
    """Impose PMC at the high-index side of selected axes.

    tidy3d's derivative layout only encodes PMC at the low-index side (the
    backward-derivative diagonal is doubled). The high-index side is the x->L-x
    mirror: the forward-derivative diagonal at the last node is doubled (sign
    kept), so dxf[Nx-1,Nx-1]: -1 -> -2 (and dyf likewise). Row-scales the
    max-face rows of the forward derivative by 2.
    """
    dxf, dxb, dyf, dyb = der_mats
    Nx, Ny = shape
    N = Nx * Ny
    if dmax_pmc[0] and Nx > 1:
        idx = np.arange((Nx - 1) * Ny, N)
        scale = np.ones(N)
        scale[idx] = 2.0
        dxf = sp.spdiags(scale, 0, N, N) @ dxf
    if dmax_pmc[1] and Ny > 1:
        idx = np.arange(Nx) * Ny + (Ny - 1)
        scale = np.ones(N)
        scale[idx] = 2.0
        dyf = sp.spdiags(scale, 0, N, N) @ dyf
    return (dxf, dxb, dyf, dyb)


def solve_2d(
    eps_cc,
    dx: float,
    f: float,
    k: int = 2,
    npml: tuple[int, int] = (0, 0),
    dmin_pmc: tuple[bool, bool] = (False, False),
    dmax_pmc: tuple[bool, bool] | None = None,
    target_neff: float | None = None,
    bend_radius: float | None = None,
    bend_center: float | None = None,
) -> Modes2D:
    """Full-vector 2D FD mode solver (diagonal eps/mu).

    ``eps_cc`` is the cell-center permittivity, shape ``(Nx, Ny)``; ``dx`` uniform
    grid step (um); ``f`` spatial frequency in 1/um (matching ``fd1d``), so the
    vacuum wavenumber is ``k0 = 2*pi*f``. ``omega = k0*C_0`` (SI rad/s) is used ONLY
    to feed the PML S-factor (so ``sigma/(omega*EPSILON_0)`` is dimensionless); it
    is not the input frequency convention. Returns the ``k`` modes sorted by
    descending Re(neff), with ``neff = neff_re + 1j*keff`` (keff carried into the
    imaginary part). PEC on all walls by default; set ``dmin_pmc``/``dmax_pmc`` to
    impose PMC at the low-/high-index side of an axis (``dmax_pmc`` defaults to
    ``dmin_pmc``, giving symmetric BCs).

    If ``bend_radius`` is set, a cylindrical (bend) coordinate transform is applied
    before matrix assembly: the diagonal Jacobian ``J = diag(1, 1, R/u)`` with
    ``u = R + (x - bend_center)`` contracts eps/mu per-component
    (``eps_xx /= dwdz``, ``eps_yy /= dwdz``, ``eps_zz *= dwdz``, mu likewise).
    ``bend_center`` defaults to the domain center ``Nx * dx / 2``.
    """
    eps_np = np.asarray(eps_cc)
    if eps_np.dtype == np.float64:
        eps_np = eps_np.astype(np.complex128)
    else:
        eps_np = eps_np.astype(np.complex128)
    Nx, Ny = eps_np.shape
    N = Nx * Ny
    # f is a spatial frequency (1/length, e.g. 1/um), matching ``fd1d``. k0 = 2*pi*f
    # is the vacuum wavenumber; omega = k0*C_0 is the SI angular frequency fed to the
    # PML S-factor (which needs rad/s so that sigma/(omega*EPSILON_0) is dimensionless).
    k0 = 2 * np.pi * f
    omega = k0 * C_0

    eps_xx, eps_yy, eps_zz = _yee_eps(eps_np)
    eps_tensor = np.zeros((3, 3, N), dtype=np.complex128)
    eps_tensor[0, 0, :] = eps_xx.ravel()
    eps_tensor[1, 1, :] = eps_yy.ravel()
    eps_tensor[2, 2, :] = eps_zz.ravel()
    mu_tensor = np.zeros((3, 3, N), dtype=np.complex128)
    for i in range(3):
        mu_tensor[i, i, :] = 1.0

    # Optional bend (cylindrical coordinate transform): apply the diagonal
    # Jacobian contraction eps' = J eps J^T / det(J), J = diag(1, 1, dwdz), to
    # eps and mu per component before matrix assembly.
    if bend_radius is not None:
        if bend_center is None:
            bend_center = Nx * dx / 2.0
        dwdz = bend_dwdz(Nx, Ny, dx, float(bend_radius), float(bend_center)).ravel()
        eps_tensor[0, 0, :] = eps_tensor[0, 0, :] / dwdz
        eps_tensor[1, 1, :] = eps_tensor[1, 1, :] / dwdz
        eps_tensor[2, 2, :] = eps_tensor[2, 2, :] * dwdz
        mu_tensor[0, 0, :] = mu_tensor[0, 0, :] / dwdz
        mu_tensor[1, 1, :] = mu_tensor[1, 1, :] / dwdz
        mu_tensor[2, 2, :] = mu_tensor[2, 2, :] * dwdz

    dl_f = (np.full(Nx, dx), np.full(Ny, dx))
    dl_b = (np.full(Nx, dx), np.full(Ny, dx))
    dls = (dl_f, dl_b)

    if dmax_pmc is None:
        dmax_pmc = dmin_pmc
    der_mats_tmp = d_mats((Nx, Ny), dls, dmin_pmc)
    der_mats_tmp = _apply_max_pmc(der_mats_tmp, (Nx, Ny), dmax_pmc)
    pml_mats = s_mats(omega, (Nx, Ny), npml, dls, eps_tensor, mu_tensor,
                      dmin_pml=(not dmin_pmc[0], not dmin_pmc[1]))
    der_mats = [S.dot(D) / k0 for S, D in zip(pml_mats, der_mats_tmp)]

    if target_neff is None:
        n_max = float(np.sqrt(np.max(np.abs(eps_np))))
        target = n_max
    else:
        target = float(target_neff)
    target_neff_p = target
    if abs(TARGET_SHIFT) > abs(target_neff_p * TARGET_SHIFT):
        target_neff_p += TARGET_SHIFT
    else:
        target_neff_p *= 1 + TARGET_SHIFT

    vec_init = _set_initial_vec(Nx, Ny)
    E, H, neff, keff = _solver_diagonal(eps_tensor, mu_tensor, der_mats, k, target_neff_p, vec_init)

    direction = "+"
    # Flip branch ambiguity (power-flow direction).
    dl_f, dl_b = dls
    E = E.reshape((3, Nx, Ny, k))
    H = H.reshape((3, Nx, Ny, k))
    dS = (np.outer(dl_f[0], dl_b[1]), np.outer(dl_b[0], dl_f[1]))
    E_tan = (np.moveaxis(E[0], -1, 0), np.moveaxis(E[1], -1, 0))
    H_tan = (np.moveaxis(H[0], -1, 0), np.moveaxis(H[1], -1, 0))
    power_flow = _complex_power_flow(E_tan, H_tan, dS)
    expected_sign = 1.0
    abs_pz = np.abs(power_flow)
    reliable = np.abs(power_flow.real) > TOL_POWER_FLOW_DIRECTION * abs_pz
    flip = (reliable & (power_flow.real * expected_sign < 0)) | ((~reliable) & (keff < 0))
    if np.any(flip):
        fi = np.where(flip)[0]
        H[0, ..., fi] *= -1
        H[1, ..., fi] *= -1
        E[2, ..., fi] *= -1
        neff[fi] *= -1
        keff[fi] *= -1

    E, H, self_orth = _normalize_modes(E, H, dl_f, dl_b, list(range(k)),
                                       TOL_NEEDS_ORTHOGONALIZATION)
    E, H = _orthogonalize_degenerate(E, H, neff, keff, dl_f, dl_b, self_orth)

    neff_t = torch.from_numpy((neff + 1j * keff).astype(np.complex128))
    Ex_t = torch.from_numpy(np.ascontiguousarray(E[0]).astype(np.complex128))
    Ey_t = torch.from_numpy(np.ascontiguousarray(E[1]).astype(np.complex128))
    return Modes2D(neff=neff_t, Ex=Ex_t, Ey=Ey_t)


def solve_2d_diff(
    eps_cc,
    dx: float,
    f: float,
    k: int = 1,
    npml: tuple[int, int] = (0, 0),
    dmin_pmc: tuple[bool, bool] = (False, False),
    dmax_pmc: tuple[bool, bool] | None = None,
    target_neff: float | None = None,
) -> torch.Tensor:
    """Differentiable entry point: real neff ``(k,)`` for the diagonal solver.

    Wraps :class:`meep_torch.modes.eigengrad.EigsNeff` so that ``neff`` carries
    an implicit-function-theorem adjoint gradient w.r.t. the cell-center
    permittivity ``eps_cc`` (requires ``eps_cc.requires_grad``). The forward
    runs the same numpy assembly + scipy shift-invert eigenproblem as
    :func:`solve_2d`, so the returned real neff matches
    ``solve_2d(...).neff.real`` to machine precision. Only the Hermitian
    (npml = 0, lossless) case is gradient-supported; PML/lossy media raise in
    backward. A degenerate-eigenvalue guard raises ``RuntimeError("degenerate
    ...")`` in forward when two returned eigenvalues coincide (gradients
    unavailable). See ``eigengrad.py`` for the IFT derivation.
    """
    from .eigengrad import EigsNeff

    return EigsNeff.apply(eps_cc, float(dx), float(f), int(k), tuple(npml),
                          tuple(dmin_pmc), dmax_pmc, target_neff)
