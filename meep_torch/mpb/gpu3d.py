# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The 3D band solve on a GPU, end to end, through cupy.

Moving only the operator apply to a device does almost nothing here, and
the profile says why. On the hole slab at resolution 32, 9 bands:

    apply       504 calls    11.0 s   ( 7.8%)
    projector  2009 calls    67.7 s   (48.1%)
    rest                     62.2 s   (44.1%)

The "rest" is scipy's LOBPCG orthogonalising and doing Rayleigh-Ritz on a
133120 x 15 complex block, on the host, every iteration. No amount of
faster apply touches it, which is exactly what a torch-apply-only port
measured: 143 s against 154 s, 7%.

So the whole iteration runs on the device instead. cupy is the natural
vehicle -- it mirrors numpy's API closely enough that the apply is the
SAME code with a different array module, rather than a second
implementation to keep in step, and ``cupyx.scipy.sparse.linalg`` carries
its own LOBPCG so the block algebra never returns to the host.

The catch, and it is a silent one: **cupy's LOBPCG is wrong for complex
Hermitian input.** It does not raise. Measured on a 64x64 complex
Hermitian matrix against ``numpy.linalg.eigvalsh``, it returned an error
of 4.4e+04 while the same call on a real symmetric matrix was accurate to
5.0e-14 and scipy's LOBPCG on the identical complex case was accurate to
6.4e-14. A wrong answer with no exception is the worst failure mode there
is, so this module never hands cupy a complex operator.

Instead it uses the standard real embedding. Writing ``A = R + iI`` with
``R`` symmetric and ``I`` antisymmetric,

    E = [[R, -I], [I, R]]

is real symmetric of twice the size and has exactly the eigenvalues of
``A``, each TWICE. Applying it costs one complex apply and nothing else:
for a real block ``Y`` standing for ``Z = Y[:n] + i Y[n:]``,

    E Y = [Re(A Z); Im(A Z)]

so the embedding is free at the point where the work happens. Verified
against a dense complex eigensolve: pairs agree to 3.6e-14 and the values
to 5.0e-14.

cupy has no CPU backend, so the CPU path stays numpy + scipy. That is not
a fallback -- it is the reference every committed gate was measured
against, and it handles complex directly.
"""
from __future__ import annotations

import warnings

import numpy as np

__all__ = ["cupy_available", "solve_bands_cupy"]


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        import cupyx.scipy.sparse.linalg  # noqa: F401
        return True
    except Exception:
        return False


def _embed(op, parity, shift, n, cp):
    """``E`` as a cupy LinearOperator on the real 2n space, plus its M.

    Both the operator and the preconditioner are wrapped once here so the
    real/complex bookkeeping lives in exactly one place. ``parity`` is
    applied inside, as it must be: it is part of the operator whose sector
    we are solving in, not a filter on the answer.

    The projector appears ONCE on each side, not twice. ``P Theta P`` is
    the textbook form, but we force the mirrored axis to an odd grid and
    symmetrise eta precisely so that ``M`` commutes with ``Theta`` exactly
    (measured 5.5e-16), which makes ``Theta(P Z)`` already lie in the
    sector and the outer projection a no-op. The preconditioner is
    diagonal in ``|K_G|^2``, which is mirror-symmetric when ``k`` has no
    component along the mirrored axis, so it commutes too. The inner
    projection stays: it is what keeps the iterate from drifting out of
    the sector, and the shift on the complement is what stops a
    smallest-eigenvalue solver from chasing the discarded half.
    """
    from cupyx.scipy.sparse.linalg import LinearOperator

    def as_complex(Y):
        return Y[:n] + 1j * Y[n:]

    def as_real(Z):
        return cp.concatenate([Z.real, Z.imag], axis=0)

    def a_apply(Y):
        Z = as_complex(cp.atleast_2d(Y.T).T if Y.ndim == 1 else Y)
        if parity is None:
            W = op.apply_device(Z)
        else:
            PZ = parity.apply_device(Z)
            W = op.apply_device(PZ) + shift * (Z - PZ)
        out = as_real(W)
        return out[:, 0] if Y.ndim == 1 else out

    def m_apply(Y):
        Z = as_complex(cp.atleast_2d(Y.T).T if Y.ndim == 1 else Y)
        if parity is None:
            W = op.precond_device(Z)
        else:
            PZ = parity.apply_device(Z)
            W = op.precond_device(PZ) + (1.0 / shift) * (Z - PZ)
        out = as_real(W)
        return out[:, 0] if Y.ndim == 1 else out

    A = LinearOperator((2 * n, 2 * n), matvec=a_apply, matmat=a_apply,
                       dtype=np.float64)
    M = LinearOperator((2 * n, 2 * n), matvec=m_apply, matmat=m_apply,
                       dtype=np.float64)
    return A, M


def solve_bands_cupy(op, num_bands: int, parity=None, tol: float = 1e-7,
                     maxiter: int = 500, extra: int = 6, seed: int = 0):
    """``(vals, vecs)`` for the lowest ``num_bands``, computed on the device.

    ``vals`` is a real numpy array; ``vecs`` is the complex numpy block, so
    the caller sees exactly what the scipy path returns and every downstream
    check -- the residual gate, the retry, the field getters -- is unchanged.

    Each eigenvalue of the embedded operator appears twice. The pair is not
    a degeneracy of the physics: it is ``z`` and ``i z``, the same complex
    eigenvector twice over. Taking every other value after sorting recovers
    the spectrum; taking the first member of each pair recovers the vector
    up to a global phase, which no observable here depends on.
    """
    import cupy as cp

    n = op.n
    shift = 0.0
    if parity is not None:
        eta_hi = float(np.max(np.linalg.eigvalsh(op.eta_host.reshape(-1, 3, 3))))
        shift = 2.0 * float(np.max(np.sum(op.K_host ** 2, axis=1))) * max(eta_hi, 1e-300)

    A, M = _embed(op, parity, shift, n, cp)

    # Twice the block: every wanted complex mode occupies two real columns.
    k = 2 * (num_bands + max(extra, 0))
    k = min(k, 2 * n - 1)
    rng = np.random.default_rng(seed)
    Y = cp.asarray(rng.standard_normal((2 * n, k)))
    if parity is not None:
        Z = Y[:n] + 1j * Y[n:]
        Z = parity.apply_device(Z)
        Y = cp.concatenate([Z.real, Z.imag], axis=0)

    from cupyx.scipy.sparse.linalg import lobpcg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals, vecs = lobpcg(A, Y, M=M, tol=tol, maxiter=maxiter,
                            largest=False)
    vals = cp.asnumpy(vals).real
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, cp.asarray(order)]

    # De-duplicate the embedding's pairs.
    keep = order.size and np.arange(0, order.size, 2)
    vals = vals[keep][:num_bands]
    Zc = vecs[:n, cp.asarray(keep)] + 1j * vecs[n:, cp.asarray(keep)]
    Zc = cp.asnumpy(Zc)[:, :num_bands]
    nrm = np.linalg.norm(Zc, axis=0)
    Zc = Zc / np.where(nrm == 0, 1.0, nrm)[None, :]
    return vals, Zc
