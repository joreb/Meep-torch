# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""BiCGSTAB(L) -- port of meep's src/bicgstab.cpp.

Sleijpen and Fokkema, "BiCGSTAB(L) for linear equations involving unsymmetric
matrices with complex spectrum", Electronic Trans. on Numerical Analysis 1,
11-32 (1993).

Why not scipy's ``bicgstab``: that is BiCGSTAB(1), and Sleijpen and Fokkema
observed it stagnating exactly when the operator's eigenvalues are near-pure
imaginary. The CW timestep operator's eigenvalues ARE i*omega, and the
stagnation is not theoretical -- BiCGSTAB(1) needed 1815 iterations to reach
1e-8 on an 80x80 waveguide, and the ring tutorial asks for 1e-12 on a
resonance. L > 1 is the whole reason meep carries its own solver.

meep runs the algorithm over the REAL view of a complex vector, so every
inner product is the real one: ``dot(a, b) = Re(conj(a) . b)``. Kept that way
here (``torch.vdot(...).real``) rather than switching to a Hermitian product,
because the operator is only R-linear once a preconditioner or a complex
shift enters and the real view is what meep validates against.
"""
import torch


def bicgstabL(L, A, b, x, tol=1e-8, maxiters=10000, breaktol=1e-30):
    """Solve ``A x = b`` in place on ``x`` (a complex 1-D tensor).

    ``A`` is a callable taking and returning a vector shaped like ``b``.
    Returns ``(ierr, iters)``: ierr 0 converged, 1 hit ``maxiters``,
    -1 breakdown (rho underflowed).
    """
    if L < 1:
        raise ValueError(f"bicgstabL needs L >= 1, got {L}")

    def dot(p, q):
        return torch.vdot(p, q).real.item()

    def nrm(p):
        return float(torch.linalg.vector_norm(p))

    r = [torch.zeros_like(b) for _ in range(L + 1)]
    u = [torch.zeros_like(b) for _ in range(L + 1)]
    gamma = [0.0] * (L + 1)
    gamma_p = [0.0] * (L + 1)
    gamma_pp = [0.0] * (L + 1)
    sigma = [0.0] * (L + 1)
    tau = [[0.0] * (L + 1) for _ in range(L + 1)]

    bnrm = nrm(b) or 1.0
    r[0] = b - A(x)
    # Sleijpen normalizes rtilde in his own code; it helps slightly
    rt_nrm = nrm(r[0])
    if rt_nrm == 0.0:
        return 0, 0
    rtilde = r[0] / rt_nrm

    rho, alpha, omega = 1.0, 0.0, 1.0
    it = 0
    while nrm(r[0]) > tol * bnrm:
        it += 1
        rho = -omega * rho
        for j in range(L):
            if abs(rho) < breaktol:
                return -1, it
            rho1 = dot(r[j], rtilde)
            beta = alpha * rho1 / rho
            rho = rho1
            for i in range(j + 1):
                u[i].mul_(-beta).add_(r[i])
            u[j + 1] = A(u[j])
            alpha = rho / dot(u[j + 1], rtilde)
            for i in range(j + 1):
                r[i].add_(u[i + 1], alpha=-alpha)
            r[j + 1] = A(r[j])
            x.add_(u[0], alpha=alpha)

        for j in range(1, L + 1):
            for i in range(1, j):
                tau[j][i] = dot(r[j], r[i]) / sigma[i]
                r[j].add_(r[i], alpha=-tau[j][i])
            sigma[j] = dot(r[j], r[j])
            gamma_p[j] = dot(r[0], r[j]) / sigma[j]

        omega = gamma[L] = gamma_p[L]
        for j in range(L - 1, 0, -1):
            gamma[j] = gamma_p[j]
            for i in range(j + 1, L + 1):
                gamma[j] -= tau[i][j] * gamma[i]
        for j in range(1, L):
            gamma_pp[j] = gamma[j + 1]
            for i in range(j + 1, L):
                gamma_pp[j] += tau[i][j] * gamma[i + 1]

        x.add_(r[0], alpha=gamma[1])
        r[0].add_(r[L], alpha=-gamma_p[L])
        u[0].add_(u[L], alpha=-gamma[L])
        for j in range(1, L):
            x.add_(r[j], alpha=gamma_pp[j])
            r[0].add_(r[j], alpha=-gamma_p[j])
            u[0].add_(u[j], alpha=-gamma[j])

        if it >= maxiters:
            return 1, it
    return 0, it


def _demo():
    """Small dense check: BiCGSTAB(L) reproduces a direct solve."""
    torch.manual_seed(0)
    n = 60
    M = (torch.randn(n, n, dtype=torch.complex128) / n**0.5
         + 3.0 * torch.eye(n, dtype=torch.complex128))
    b = torch.randn(n, dtype=torch.complex128)
    exact = torch.linalg.solve(M, b)
    for L in (1, 2, 4):
        x = torch.zeros(n, dtype=torch.complex128)
        ierr, it = bicgstabL(L, lambda v: M @ v, b, x, tol=1e-12, maxiters=500)
        err = float(torch.linalg.vector_norm(x - exact)
                    / torch.linalg.vector_norm(exact))
        assert ierr == 0 and err < 1e-10, (L, ierr, it, err)
        print(f"L={L}: {it} iters, rel err {err:.2e}")


if __name__ == "__main__":
    _demo()
