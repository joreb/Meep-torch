# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Implicit-function-theorem autograd for the diagonal FD mode solver neff.

Original code (not a port). Wraps the numpy assembly + scipy shift-invert
eigenproblem of ``fd2d.solve_2d`` in a ``torch.autograd.Function`` so that the
returned real neff ``(k,)`` carries an analytic gradient w.r.t. the cell-center
permittivity ``eps_cc``.

IFT derivation (lossless, npml = 0, mu = 1)
-----------------------------------------
The diagonal solver assembles the (2N, 2N) standard eigenproblem

    A(eps) v = lam v,   lam = -(neff + 1j*keff)**2 = -neff**2   (keff = 0).

For a perturbation ``d eps`` of a *Yee-position* eps diagonal entry ``p``
(``eps_xx``, ``eps_yy`` appear linearly in ``q_ep``; ``eps_zz`` enters via
``inv_eps_zz = 1/eps_zz``), the implicit-function theorem gives

    d lam_i = (w_i^T (dA/dp) v_i) / (w_i^T v_i),

where ``v_i`` is the right eigenvector and ``w_i`` the left eigenvector
(``A^T w_i = lam_i w_i``). The matrix is real but NOT symmetric (the off-diagonal
derivative blocks break symmetry), so ``w_i != v_i``; the left eigenvector is
recovered per mode by a scipy ``eigs`` on ``A^T`` (shift-invert at
``sigma = lam_i``, ``v0 = v_i``). For the chain to the returned real neff
(``neff = sqrt(-lam)`` for the forward branch),

    d neff_i / d lam_i = -1 / (2 * neff_i)   -->   d neff_i = -d lam_i / (2 neff_i).

The cell-center eps enters each Yee eps component by two-point edge-replicated
averaging (``eps_xx = avg_x(cc)``, ``eps_yy = avg_y(cc)``, ``eps_zz = cc``); the
backward chains through the transpose of that averaging (weight 0.5 on each of
the 1--2 contributing cells, weight 1.0 on the edge-replicated last cell).

Degenerate guard
----------------
If two returned eigenvalues coincide within ``1e-9 * |lam|`` the IFT denominator
``w_i^T v_i`` becomes ill-conditioned and the per-mode gradient is undefined; the
forward raises ``RuntimeError("degenerate ...")``. PML / lossy media (non-Hermitian
A, complex lam) are not gradient-supported in this v1 and raise in backward.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spl
import torch

from .derivatives import C_0, create_d_matrices as d_mats, create_s_matrices as s_mats
from .fd2d import (
    TARGET_SHIFT,
    _assemble_diagonal,
    _apply_max_pmc,
    _complex_power_flow,
    _eigs_to_neff,
    _is_complex,
    _set_initial_vec,
    _solver_eigs,
    fp_eps,
    TOL_EIGS,
)

TOL_DEGENERATE = 1e-4


def _left_eigenvector(mat, lam: float, v: np.ndarray) -> np.ndarray:
    """Left eigenvector ``w`` of ``mat`` (``mat.T w = lam w``) for the eigenpair
    ``(lam, v)``, via shift-invert on ``mat.T`` seeded with ``v``.

    The shift is nudged off the exact eigenvalue by a tiny relative amount so the
    shift-invert factorization ``(mat.T - sigma I)`` is non-singular; the nearest
    eigenvalue to the perturbed shift is still ``lam`` for any non-degenerate pair
    (the forward degenerate guard guarantees ``|lam_i - lam_j| > 1e-9 |lam_i|``).
    """
    sigma = lam + 1e-7 * (abs(lam) + 1.0)
    mat_t = mat.T.tocsr()
    try:
        vals_w, vecs_w = spl.eigs(mat_t, k=1, sigma=sigma, tol=TOL_EIGS, v0=v)
    except spl.ArpackNoConvergence:
        # Retry with the exact eigenvalue as the shift (scipy handles the
        # near-singular factorization internally in some builds).
        vals_w, vecs_w = spl.eigs(mat_t, k=1, sigma=lam, tol=TOL_EIGS, v0=v)
    # Pairing guard: the shift-invert on A.T must converge back to lam_i, not a
    # neighboring eigenvalue. A silent wrong target would feed an incorrect IFT
    # denominator (w^T v) and produce wrong gradients for future k>1 callers.
    lam_found = float(vals_w[0].real)
    if abs(lam_found - lam) > 1e-3 * (abs(lam) + 1.0):
        raise RuntimeError(
            "left-eigenvector solve converged to wrong eigenvalue "
            f"(lam_found={lam_found:.6g}, expected={lam:.6g}); "
            "shift-invert drifted to a neighbor"
        )
    w = vecs_w[:, 0]
    # Strip a global phase by aligning the dominant entry of w to that of v.
    k_dom = int(np.argmax(np.abs(w)))
    if abs(w[k_dom]) > 0:
        w = w * (np.abs(w[k_dom]) / w[k_dom])
    return w


def _avg_x_adjoint(g: np.ndarray) -> np.ndarray:
    """Transpose of ``fd2d._avg_x``: map a Yee-eps_xx sensitivity field back to
    cell-center. ``eps_xx[i] = 0.5(cc[i]+cc[i+1])`` for i<N-1, ``= cc[N-1]`` at
    the edge-replicated last row."""
    out = np.zeros_like(g)
    out[:-1] += 0.5 * g[:-1]
    out[1:] += 0.5 * g[:-1]
    out[-1] += g[-1]
    return out


def _avg_y_adjoint(g: np.ndarray) -> np.ndarray:
    """Transpose of ``fd2d._avg_y`` (along axis 1)."""
    out = np.zeros_like(g)
    out[:, :-1] += 0.5 * g[:, :-1]
    out[:, 1:] += 0.5 * g[:, :-1]
    out[:, -1] += g[:, -1]
    return out


class EigsNeff(torch.autograd.Function):
    """Differentiable real-neff wrapper around the diagonal FD mode solver."""

    @staticmethod
    def forward(ctx, eps_cc, dx, f, k, npml, dmin_pmc, dmax_pmc, target_neff):
        eps_np = np.asarray(eps_cc.detach().cpu().numpy()).astype(np.complex128)
        Nx, Ny = eps_np.shape
        N = Nx * Ny
        k0 = 2 * np.pi * f
        omega = k0 * C_0

        # Build Yee eps / mu tensors exactly as fd2d.solve_2d (no bend here).
        eps_xx = 0.5 * (eps_np[:-1] + eps_np[1:])
        eps_xx = np.vstack([eps_xx, eps_np[-1:]])
        eps_yy = 0.5 * (eps_np[:, :-1] + eps_np[:, 1:])
        eps_yy = np.hstack([eps_yy, eps_np[:, -1:]])
        eps_zz = eps_np
        eps_tensor = np.zeros((3, 3, N), dtype=np.complex128)
        eps_tensor[0, 0, :] = eps_xx.ravel()
        eps_tensor[1, 1, :] = eps_yy.ravel()
        eps_tensor[2, 2, :] = eps_zz.ravel()
        mu_tensor = np.zeros((3, 3, N), dtype=np.complex128)
        for i in range(3):
            mu_tensor[i, i, :] = 1.0

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
            target = float(np.sqrt(np.max(np.abs(eps_np))))
        else:
            target = float(target_neff)
        target_neff_p = target
        if abs(TARGET_SHIFT) > abs(target_neff_p * TARGET_SHIFT):
            target_neff_p += TARGET_SHIFT
        else:
            target_neff_p *= 1 + TARGET_SHIFT

        vec_init = _set_initial_vec(Nx, Ny)

        mat, assem = _assemble_diagonal(eps_tensor, mu_tensor, der_mats)
        if _is_complex(eps_tensor) or _is_complex(mu_tensor) or any(_is_complex(d) for d in der_mats):
            mat = mat.astype(np.complex128)
            vec_init = vec_init.astype(np.complex128)
        else:
            mat = mat.astype(np.float64)
            vec_init = vec_init.real.astype(np.float64)
        eig_guess = np.array([-(target_neff_p ** 2)]).astype(mat.dtype)[0]

        vals, vecs = _solver_eigs(mat, k, vec_init, guess_value=eig_guess)
        neff, keff = _eigs_to_neff(vals)
        sort_inds = np.argsort(neff)[::-1]
        vals = vals[sort_inds]
        vecs = vecs[:, sort_inds]
        neff = neff[sort_inds]
        keff = keff[sort_inds]

        # Degenerate-eigenvalue guard (IFT denominator ill-conditioned). The
        # gap is measured on the returned neff (the quantity differentiated);
        # a symmetry-degenerate pair is exact in theory but ARPACK reports the
        # two members with an O(1e-6) split, so the threshold must flag
        # numerical near-degeneracy, not just exact equality. k = 1 skips.
        if k > 1:
            for ii in range(k):
                for jj in range(ii + 1, k):
                    if abs(neff[ii] - neff[jj]) < TOL_DEGENERATE * abs(neff[ii]):
                        raise RuntimeError(
                            "degenerate eigenvalues -- gradients unavailable; "
                            "perturb geometry or reduce k"
                        )

        # Power-flow direction flip (replicated from fd2d.solve_2d so the
        # returned neff matches the non-differentiable path exactly).
        qmat = assem["qmat"]
        Ex = vecs[:N, :]
        Ey = vecs[N:, :]
        h_field = qmat.dot(vecs)
        Hx = h_field[:N, :] / (1j * neff - keff)
        Hy = h_field[N:, :] / (1j * neff - keff)
        Ex3 = Ex.reshape((Nx, Ny, k))
        Ey3 = Ey.reshape((Nx, Ny, k))
        Hx3 = Hx.reshape((Nx, Ny, k))
        Hy3 = Hy.reshape((Nx, Ny, k))
        dS = (np.outer(dl_f[0], dl_b[1]), np.outer(dl_b[0], dl_f[1]))
        E_tan = (np.moveaxis(Ex3, -1, 0), np.moveaxis(Ey3, -1, 0))
        H_tan = (np.moveaxis(Hx3, -1, 0), np.moveaxis(Hy3, -1, 0))
        power_flow = _complex_power_flow(E_tan, H_tan, dS)
        abs_pz = np.abs(power_flow)
        reliable = np.abs(power_flow.real) > 1e2 * TOL_EIGS * abs_pz
        flip = (reliable & (power_flow.real < 0)) | ((~reliable) & (keff < 0))
        if np.any(flip):
            fi = np.where(flip)[0]
            neff[fi] *= -1
            keff[fi] *= -1

        neff_real = neff.astype(np.float64)
        out = torch.from_numpy(neff_real).to(eps_cc.device).to(eps_cc.dtype if eps_cc.is_floating_point() else torch.float64)

        ctx.vals = vals
        ctx.vecs = vecs
        ctx.neff = neff_real
        ctx.assem = assem
        ctx.mat = mat
        ctx.Nx = Nx
        ctx.Ny = Ny
        ctx.N = N
        ctx.npml = npml
        return out

    @staticmethod
    def backward(ctx, grad_output):
        vals = ctx.vals
        vecs = ctx.vecs
        neff = ctx.neff
        assem = ctx.assem
        mat = ctx.mat
        N = ctx.N
        Nx, Ny = ctx.Nx, ctx.Ny
        npml = ctx.npml

        if npml != (0, 0):
            raise NotImplementedError(
                "PML / non-Hermitian adjoint not supported in v1 (npml must be 0)"
            )

        g = np.asarray(grad_output.detach().cpu().numpy(), dtype=np.float64)
        k = vals.shape[0]

        eps_xx = np.asarray(assem["eps_xx"]).ravel().real
        eps_yy = np.asarray(assem["eps_yy"]).ravel().real
        eps_zz = np.asarray(assem["eps_zz"]).ravel().real
        dxf, dxb, dyf, dyb = assem["der_mats"]
        p_mu = assem["p_mu"]
        p_partial = assem["p_partial"]
        P = (p_mu + p_partial).tocsr()

        cc_grad = np.zeros((Nx, Ny), dtype=np.float64)

        for i in range(k):
            gi = float(g[i])
            if gi == 0.0:
                continue
            lam_i = float(vals[i].real)
            neff_i = float(neff[i])
            if abs(neff_i) < fp_eps:
                continue  # evanescent / zero-neff: skip (no defined chain)
            # npml = 0 -> mat real -> eigenpairs real (arpack returns complex
            # with ~0 imag); take the real part for a real VJP.
            v = vecs[:, i].real
            w = _left_eigenvector(mat, lam_i, v).real
            v0 = v[:N]
            v1 = v[N:]
            w0 = w[:N]
            w1 = w[N:]
            denom = float(np.dot(w, v))
            if abs(denom) < 1e-30:
                continue

            Pt = (P.T @ w).real  # (2N,)
            Pt0 = Pt[:N]
            Pt1 = Pt[N:]

            # d lam / d eps_xx  (Yee), d lam / d eps_yy, d lam / d eps_zz.
            dlam_dex = -v0 * Pt1 / denom            # (N,)
            dlam_dey = v1 * Pt0 / denom              # (N,)
            qv0 = eps_yy * v1
            qv1 = -eps_xx * v0
            # c = lower half of (q_ep @ v): the eps-weighted tangential-E
            # combination q_ep_10 v0 + q_ep_11 v1 = dxb@(eps_xx v0) + dyb@(eps_yy v1)
            # (the minus sign on qv1 flips -eps_xx*v0 to +eps_xx*v0 under dxb).
            c = (dyb @ qv0) - (dxb @ qv1)            # (N,)
            s = (dxf.T @ w0) + (dyf.T @ w1)          # (N,)
            dlam_dez = c * s / (eps_zz ** 2) / denom  # (N,)

            # Chain d neff / d Yee = -1/(2 neff) * d lam / d Yee, weighted by gi.
            fac = -gi / (2.0 * neff_i)
            g_xx = (fac * dlam_dex).real.reshape((Nx, Ny))
            g_yy = (fac * dlam_dey).real.reshape((Nx, Ny))
            g_zz = (fac * dlam_dez).real.reshape((Nx, Ny))

            # Chain Yee -> cell-center via the averaging transpose.
            cc_grad += _avg_x_adjoint(g_xx)
            cc_grad += _avg_y_adjoint(g_yy)
            cc_grad += g_zz

        grad_eps = torch.from_numpy(cc_grad).to(grad_output.device).to(grad_output.dtype)
        return grad_eps, None, None, None, None, None, None, None