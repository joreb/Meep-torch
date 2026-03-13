# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2D TM driven FDFD: sparse direct solve for steady-state Ez, with a
single-adjoint-solve autograd backward. Textbook stretched-coordinate PML
(cubic sigma ramp), not ported from any external solver.

Node grid, natural units (c = eps0 = mu0 = 1), omega = 2*pi*freq:
    (1/sx d/dx (1/sx' d/dx) + 1/sy d/dy (1/sy' d/dy) + omega^2 eps) Ez = -i*omega*Jz
with complex-stretch s(pos) = 1 + i*sigma(pos)/omega, sigma = sigma_max*u^3,
u the ramp fraction into the outer npml cells. Dirichlet outer boundary
(ghost nodes are dropped from the sparse rows, not appended as zeros).

S-parameters: mode-profile overlap on the Ez line (make_mode_source /
mode_amplitude), normalized against a straight-reference run -- a plain
current sheet radiates both directions, so absolute FDFD amplitudes are not
meaningful on their own (see make_mode_source docstring).
"""
import math

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu

from .modes.fd1d import solve_1d

__all__ = ["solve_fdfd_tm", "port_mode_line", "mode_amplitude", "make_mode_source"]


def _stretch(n: int, positions: np.ndarray, npml: int, dx: float, omega: float) -> np.ndarray:
    """s(pos) = 1 + i*sigma(pos)/omega at the given positions (grid units,
    same coordinate as node index i=0..n-1); sigma_max set for R0=1e-6."""
    if npml <= 0:
        return np.ones_like(positions, dtype=np.complex128)
    u = np.clip((npml - positions) / npml, 0.0, None) + np.clip(
        (positions - (n - 1 - npml)) / npml, 0.0, None
    )
    sigma_max = -(3 + 1) * math.log(1e-6) / (2.0 * npml * dx)
    sigma = sigma_max * u ** 3
    return 1.0 + 1j * sigma / omega


def _axis_operator(n: int, npml: int, dx: float, omega: float) -> sp.csr_matrix:
    """1D (1/s) D-( (1/s') D+ ) / dx^2 with Dirichlet ghosts dropped (n x n,
    complex128). s at integer positions, s' at half positions (edge i+1/2)."""
    sx = _stretch(n, np.arange(n, dtype=np.float64), npml, dx, omega)
    sxh = _stretch(n, np.arange(n - 1, dtype=np.float64) + 0.5, npml, dx, omega)
    inv_dx2 = 1.0 / (dx * dx)

    coef_plus = np.zeros(n, dtype=np.complex128)   # row i -> Ez[i+1], i=0..n-2
    coef_plus[:-1] = inv_dx2 / (sx[:-1] * sxh)
    coef_minus = np.zeros(n, dtype=np.complex128)  # row i -> Ez[i-1], i=1..n-1
    coef_minus[1:] = inv_dx2 / (sx[1:] * sxh)
    diag = -(coef_plus + coef_minus)

    return sp.diags(
        [coef_minus[1:], diag, coef_plus[:-1]], offsets=[-1, 0, 1],
        format="csr", dtype=np.complex128,
    )


def _assemble(eps: np.ndarray, dx: float, omega: float, npml: int) -> sp.csc_matrix:
    """Row-major flatten (index i*ny + j): x-operator kron'd with I_ny,
    y-operator kron'd with I_nx (standard Kronecker sum for C-order)."""
    nx, ny = eps.shape
    Lx = _axis_operator(nx, npml, dx, omega)
    Ly = _axis_operator(ny, npml, dx, omega)
    Ix = sp.identity(nx, format="csr", dtype=np.complex128)
    Iy = sp.identity(ny, format="csr", dtype=np.complex128)
    A = sp.kron(Lx, Iy, format="csr") + sp.kron(Ix, Ly, format="csr")
    A = A + sp.diags((omega ** 2) * eps.reshape(-1).astype(np.complex128), format="csr")
    return A.tocsc()


def _import_torch_sla():
    try:
        import torch_sla
    except ImportError as e:
        raise ImportError(
            "backend='torch-sla' requires the torch-sla package: uv add torch-sla"
        ) from e
    return torch_sla


class _FDFDSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, jz, dx, omega, npml, backend):
        nx, ny = eps.shape
        eps_np = eps.detach().to(torch.float64).cpu().numpy()
        b_np = (-1j * omega * jz.detach().to(torch.complex128).cpu().numpy()).reshape(-1)
        ctx.omega, ctx.shape, ctx.device, ctx.backend = omega, (nx, ny), eps.device, backend

        if backend == "scipy":
            lu = splu(_assemble(eps_np, dx, omega, npml))
            E = lu.solve(b_np)
            ctx.lu, ctx.E = lu, E
            return torch.from_numpy(E.reshape(nx, ny)).to(eps.device)
        elif backend == "torch-sla":
            torch_sla = _import_torch_sla()
            A = _assemble(eps_np, dx, omega, npml).tocoo()
            n = A.shape[0]
            row = torch.from_numpy(A.row.astype(np.int64)).to(eps.device)
            col = torch.from_numpy(A.col.astype(np.int64)).to(eps.device)
            val = torch.from_numpy(A.data).to(eps.device)
            ctx.row, ctx.col, ctx.val, ctx.n = row, col, val, n
            b = torch.from_numpy(b_np).to(eps.device)
            E = torch_sla.solve(torch_sla.SparseTensor(val, row, col, (n, n)), b)
            ctx.E = E.detach()
            return E.reshape(nx, ny)
        else:
            raise ValueError(f"backend must be 'scipy' or 'torch-sla', got {backend!r}")

    @staticmethod
    def backward(ctx, grad_output):
        # dL/deps_k = -omega^2 * Re[lam_k * E_k], lam = A^-T conj(g). Derived
        # from L(E, conj(E)) real, dE/deps_k = -omega^2 E_k A^-1 e_k, and
        # PyTorch's own complex convention grad_output = 2 * dL/dconj(E)
        # (not dL/dconj(E) -- verified against the FD gate: the textbook
        # "-2*omega^2" reads dL/dconj(E) without PyTorch's built-in factor
        # of 2, so the *2 cancels the *2 already inside grad_output).
        g = grad_output.detach()
        if ctx.backend == "scipy":
            g_np = g.cpu().numpy().reshape(-1).astype(np.complex128)
            lam = torch.from_numpy(ctx.lu.solve(np.conj(g_np), trans="T")).to(ctx.device)
            E = torch.from_numpy(ctx.E).to(ctx.device)
        else:
            torch_sla = _import_torch_sla()
            # A^T: same (values, shape), row/col swapped -- no factorization
            # to reuse, so this is a fresh solve (same cost class as forward).
            AT = torch_sla.SparseTensor(ctx.val, ctx.col, ctx.row, (ctx.n, ctx.n))
            # torch_sla's scipy backend calls .numpy() on b, which fails on a
            # lazy conj()-view tensor -- resolve_conj() materializes it first.
            lam = torch_sla.solve(AT, g.reshape(-1).conj().resolve_conj())
            E = ctx.E.reshape(-1)
        grad_eps = (-ctx.omega ** 2 * (lam * E).real).reshape(ctx.shape).to(torch.float64)
        return grad_eps, None, None, None, None, None


def solve_fdfd_tm(eps: torch.Tensor, dx: float, freq: float, jz: torch.Tensor,
                   npml: int, backend: str = "scipy") -> torch.Tensor:
    """Driven 2D TM solve. eps: real float64 (nx, ny), may require grad.
    jz: complex or real (nx, ny) current density; b = -i*omega*jz. Returns
    Ez complex128 (nx, ny). Adjoint backward w.r.t. eps only (one extra
    transposed solve, no jz/dx/freq/npml grad).

    backend: "scipy" (default, CPU, cached LU factorization) or
    "torch-sla" (opt-in GPU-capable path via the torch-sla package, not a
    hard dependency -- raises ImportError with an install hint if missing).
    Same assembled matrix either way (_assemble); torch-sla only replaces
    the linear-solve calls, not the adjoint derivation above."""
    omega = 2.0 * math.pi * freq
    jz_c = jz if torch.is_complex(jz) else jz.to(torch.complex128)
    return _FDFDSolve.apply(eps, jz_c, dx, omega, npml, backend)


def _port_eps_line(eps: torch.Tensor, port) -> torch.Tensor:
    if port.axis == "x":
        return eps[port.index, port.lo:port.hi]
    return eps[port.lo:port.hi, port.index]


def port_mode_line(eps: torch.Tensor, port, freq: float, dx: float, mode: int = 0):
    """TM mode profile + neff on a port line, extracted the same way as
    ModeSourceTM. Raises ValueError if the mode isn't guided (forward-only:
    the eigensolve runs in numpy, no grad w.r.t. eps)."""
    eps_line = _port_eps_line(eps, port).detach()
    modes = solve_1d(eps_line, dx, freq, "tm", k=mode + 1)
    neff = float(modes.neff[mode])
    clad = float(torch.sqrt(torch.max(eps_line[0], eps_line[-1])))
    if neff <= clad + 1e-9:
        raise ValueError(f"no guided mode at port (neff={neff:.4f} <= cladding {clad:.4f})")
    return modes.profiles[mode].to(torch.float64), neff


def mode_amplitude(Ez: torch.Tensor, eps: torch.Tensor, port, freq: float, dx: float,
                    mode: int = 0) -> torch.Tensor:
    """a = sum(Ez_line * profile) * dx on the port line. Differentiable
    w.r.t. Ez (plain torch indexing/sum); eps is only used to extract the
    (detached) mode profile."""
    profile, _ = port_mode_line(eps, port, freq, dx, mode)
    if port.axis == "x":
        Ez_line = Ez[port.index, port.lo:port.hi]
    else:
        Ez_line = Ez[port.lo:port.hi, port.index]
    return (Ez_line * profile.to(Ez.dtype)).sum() * dx


def make_mode_source(eps: torch.Tensor, port, freq: float, dx: float,
                      mode: int = 0) -> torch.Tensor:
    """Current sheet Jz with the port's TM mode profile placed on
    port.index, zero elsewhere. A plain current sheet radiates both
    directions (unlike the FDTD TFSF ModeSourceTM) -- S-parameters here are
    always taken as a ratio against a straight-reference run, so the
    backward radiation is harmless and this stays a single line source."""
    profile, _ = port_mode_line(eps, port, freq, dx, mode)
    jz = torch.zeros(eps.shape, dtype=torch.complex128, device=eps.device)
    prof_c = profile.to(torch.complex128).to(eps.device)
    if port.axis == "x":
        jz[port.index, port.lo:port.hi] = prof_c
    else:
        jz[port.lo:port.hi, port.index] = prof_c
    return jz
