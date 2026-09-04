# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# 1D TE/TM slab mode solver. TM (Ez): symmetric eigenproblem A = D2/k0^2 + diag(eps),
# eigenvalue neff^2. TE (Hz): nonsymmetric A_te = diag(eps) d/dy(1/eps d/dy)/k0^2 + diag(eps),
# eigenvalue neff^2. Dirichlet (PEC) boundaries; cell-centered nodes y_i=(i+0.5)dx.
from typing import NamedTuple

import numpy as np
import scipy.linalg
import torch


class Modes1D(NamedTuple):
    neff: torch.Tensor        # (k,) float64, descending
    profiles: torch.Tensor    # (k, n) float64, L2-normalized, sign-fixed


def _second_diff_dirichlet(n: int, dx: float) -> np.ndarray:
    """D2 = tridiag(1, -2, 1)/dx^2 (n x n). Dirichlet ghosts are 0, so they drop out."""
    main = np.full(n, -2.0)
    off = np.ones(n - 1)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx * dx)


def _forward_diff_dirichlet(n: int, dx: float) -> np.ndarray:
    """Df (n+1, n): nodes -> half-edges (cell edges). Dirichlet ghosts = 0.
    Half-edge 0 (left boundary):  (phi_0 - 0)/dx
    Half-edge k (1..n-1, interior): (phi_k - phi_{k-1})/dx
    Half-edge n (right boundary): (0 - phi_{n-1})/dx"""
    Df = np.zeros((n + 1, n))
    Df[0, 0] = 1.0
    for k in range(1, n):
        Df[k, k - 1] = -1.0
        Df[k, k] = 1.0
    Df[n, n - 1] = -1.0
    return Df / dx


def _backward_diff_dirichlet(n: int, dx: float) -> np.ndarray:
    """Db (n, n+1): half-edges -> nodes. Node i: (g_{i+1} - g_i)/dx."""
    Db = np.zeros((n, n + 1))
    for i in range(n):
        Db[i, i] = -1.0
        Db[i, i + 1] = 1.0
    return Db / dx


def _eps_half(eps: np.ndarray) -> np.ndarray:
    """eps at the n+1 half-edges: interior = two-point average; boundary edges
    take the adjacent cell's eps (cladding extends, Dirichlet ghost)."""
    n = eps.shape[0]
    eh = np.empty(n + 1, dtype=eps.dtype)
    eh[1:n] = 0.5 * (eps[:-1] + eps[1:])
    eh[0] = eps[0]
    eh[n] = eps[-1]
    return eh


def _tm_operator(eps: np.ndarray, dx: float, k0: float) -> np.ndarray:
    n = eps.shape[0]
    D2 = _second_diff_dirichlet(n, dx)
    return D2 / (k0 * k0) + np.diag(eps)


def _te_operator(eps: np.ndarray, dx: float, k0: float) -> np.ndarray:
    n = eps.shape[0]
    Df = _forward_diff_dirichlet(n, dx)
    Db = _backward_diff_dirichlet(n, dx)
    eh = _eps_half(eps)
    deriv = Db @ np.diag(1.0 / eh) @ Df          # d/dy(1/eps d/dy), (n, n)
    return np.diag(eps) @ deriv / (k0 * k0) + np.diag(eps)


def _finalize(neff2: np.ndarray, vecs: np.ndarray, k: int):
    """Sort by eigenvalue descending, take top-k, sqrt -> neff; L2-normalize
    and sign-fix each profile (max-|value| element positive)."""
    order = np.argsort(neff2)[::-1][:k]
    top_vals = neff2[order]
    top_vecs = vecs[:, order]                      # (n, k)
    neff = np.sqrt(np.clip(top_vals, 0.0, None))
    profs = np.empty_like(top_vecs)
    for i in range(top_vecs.shape[1]):
        p = top_vecs[:, i]
        nrm = np.linalg.norm(p)
        if nrm > 0:
            p = p / nrm
        idx = int(np.argmax(np.abs(p)))
        if p[idx] < 0:
            p = -p
        profs[:, i] = p
    return Modes1D(neff=torch.from_numpy(neff.copy()).to(torch.float64),
                   profiles=torch.from_numpy(profs.T.copy()).to(torch.float64))


def solve_1d(eps_line, dx: float, f: float, pol: str, k: int = 1, bc: str = "pec"):
    """bc="pec": Dirichlet both ends. bc="pmc_lo": Neumann (magnetic wall)
    at the LOW end, Dirichlet high -- the even-mirror half-domain case
    (fdtdx symmetry +1): profiles are the y >= 0 half of even modes."""
    if bc not in ("pec", "pmc_lo"):
        raise ValueError(f"bc must be 'pec' or 'pmc_lo', got {bc!r}")
    if bc == "pmc_lo" and pol != "tm":
        raise ValueError("bc='pmc_lo' implemented for pol='tm' only")
    if pol not in ("tm", "te"):
        raise ValueError(f"pol must be 'tm' or 'te', got {pol!r}")
    k0 = 2.0 * np.pi * f
    if isinstance(eps_line, torch.Tensor):
        eps = eps_line.detach().to(torch.float64).cpu().numpy()
    else:
        eps = np.asarray(eps_line, dtype=np.float64)
    if pol == "tm":
        A = _tm_operator(eps, dx, k0)
        if bc == "pmc_lo":
            # Neumann ghost f(-1) = f(0): second-difference row 0 becomes
            # (-1, 1)/dx^2 instead of (-2, 1)/dx^2. The operator carries
            # D2/k0^2, so the adjustment is scaled the same way.
            A[0, 0] += 1.0 / (dx * dx * k0 * k0)
        vals, vecs = np.linalg.eigh(A)            # ascending, all real
    else:
        A = _te_operator(eps, dx, k0)            # nonsymmetric
        vals_c, vecs_c = scipy.linalg.eig(A)
        real = np.abs(vals_c.imag) < 1e-10
        vals, vecs = vals_c[real].real, vecs_c[:, real].real
    modes = _finalize(vals, vecs, k)
    # The eigensolve is scipy, so results come back on the CPU whatever device
    # the caller's eps was on. Return them on the input's device: every
    # consumer immediately combines the profile with that same eps (mode
    # overlaps in ports/mode_monitor, source masks in ports/mode_source), so
    # a CPU return turns into a device mismatch at each point of use instead
    # of being handled once, here.
    if isinstance(eps_line, torch.Tensor) and eps_line.device.type != "cpu":
        modes = Modes1D(neff=modes.neff, profiles=modes.profiles.to(eps_line.device))
    return modes


def solve_1d_tm_torch(eps_line, dx: float, f: float, k: int = 1) -> torch.Tensor:
    """Differentiable TM path: same tridiagonal operator as a torch tensor,
    torch.linalg.eigvalsh, returns neff (k,) descending (no profiles)."""
    k0 = 2.0 * np.pi * f
    eps = eps_line.to(torch.float64)
    n = eps.shape[0]
    D2 = torch.from_numpy(_second_diff_dirichlet(n, dx)).to(eps.dtype).to(eps.device)
    A = D2 / (k0 * k0) + torch.diag(eps)
    vals = torch.linalg.eigvalsh(A)             # ascending
    top = vals[-k:]
    neff = torch.sqrt(torch.clamp(top, min=0.0))
    return torch.flip(neff, dims=[0])           # descending
