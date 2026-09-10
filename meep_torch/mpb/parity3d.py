# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mirror-parity projectors and expectation values for the 3D planewave path.

MPB's ``run_zeven`` / ``run_zodd`` do not filter bands after solving; they
solve inside a symmetry sector, which is both cheaper and the only way to
get a clean answer when the two sectors interleave in frequency
(``maxwell_constraints.c``). This module builds the mirror operator in the
planewave basis so the same thing can happen here: the projector is
applied inside the operator apply and to the start block, and the
eigenvalues then come out of the chosen sector only.

The mirror. Under ``z -> -z`` a field transforms as ``V'(r) = S V(sigma r)``.
The sign matrix ``S`` is NOT the same for E and H: E is a true vector and
picks up ``diag(1, 1, -1)``, while H is a PSEUDOvector -- it is a curl of
one -- and picks up ``diag(-1, -1, 1)``. Getting that backwards swaps the
two sectors, which is invisible on a symmetric structure (both sectors
exist and both look plausible) and shows up only as the wrong band
frequencies against an oracle. The unknowns here are H, so ``diag(-1,-1,1)``
is the one used.

In the planewave basis ``sigma`` maps ``G = (m, n, p)`` to ``(m, n, -p)``,
which is an index permutation on the FFT grid, so the whole operator is a
gather plus a sign. It requires ``kc_z = 0``: with a nonzero out-of-plane
k the mirror maps the Bloch sector to a different one and there is no
symmetry to project onto. That is checked rather than assumed.

Parity as a NUMBER (MPB's ``display_zparities``) is the expectation value
``<h|M|h>`` on the normalised eigenvector: +1 for a perfectly even mode,
-1 for odd, and something in between when the structure is not symmetric
or the band is a degenerate mixture. MPB prints exactly this.
"""
from __future__ import annotations

import numpy as np

__all__ = ["MirrorParity", "zparity_operator", "yparity_operator"]

# H is a pseudovector: the sign matrix picks up an extra overall -1
# relative to a true vector, so the MIRRORED axis keeps its sign and the
# other two flip.
_H_SIGNS = {0: np.array([1.0, -1.0, -1.0]),      # x -> -x
            1: np.array([-1.0, 1.0, -1.0]),      # y -> -y
            2: np.array([-1.0, -1.0, 1.0])}      # z -> -z


class MirrorParity:
    """The mirror operator ``M`` on frame coefficients, and its projectors.

    ``M`` is Hermitian and involutive (``M^2 = I``), so ``(I +- M)/2`` are
    orthogonal projectors onto the even and odd sectors. Both facts are
    asserted by the tests rather than assumed here, because an index-map
    error breaks involutivity while still producing a plausible-looking
    operator.
    """

    def __init__(self, op, axis: int = 2, even: bool = True):
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        # Index 0 of the FFT frequency grid is G = 0, so K[0] IS kc.
        kc_perp = float(op.K[0, axis])
        if abs(kc_perp) > 1e-9:
            raise ValueError(
                f"mirror parity about axis {axis} needs k_{'xyz'[axis]} = 0; "
                f"got {kc_perp:.4g}. A nonzero component along the mirrored "
                f"axis maps the Bloch sector onto a different one, so there "
                f"is no symmetry to project onto.")
        self.op = op
        self.axis = axis
        self.even = bool(even)
        self.signs = _H_SIGNS[axis]
        self._keep = _non_nyquist_mask(op.grid, axis)
        # Frame-coefficient form: gather the mirrored G's cartesian vector,
        # apply the sign matrix, project onto THIS G's frame. The frames at
        # G and its mirror are unrelated, so this is a genuine 2x2 mixing
        # per pair and not a sign flip on the coefficients.
        self._t1, self._t2 = op.t1, op.t2

    def mirror(self, X):
        """``M X`` for (2 N_G, m) frame coefficients.

        The flip is done in REAL space, not by negating the Fourier index,
        and that is not a style choice. numpy's FFT frequency set is not
        closed under negation for an even ``n``: the Nyquist entry is
        ``-n/2`` and its negative ``+n/2`` is not in the set, it aliases
        back onto itself. So the index map ``i -> (-i) % n`` does NOT send
        ``K_G`` to its mirror on that plane -- measured, it is off by a full
        reciprocal period (8 pi on a 4-cell axis) -- and the resulting
        operator is not involutive: ``M^2`` came out at 2.8 instead of 0.

        In real space the flip IS exact, but it is ``k -> (-k) % n``, NOT
        ``k -> n - 1 - k``. ``ifftn`` puts its samples at ``r = k/n``, not
        at the half-cell-offset centres the rasterizer uses, and on that
        grid only the modular negation sends ``r`` to exactly ``-r``;
        reversing the array instead shifts by one cell.

        Neither form rescues the NYQUIST plane, and nothing can: for an
        even ``n`` the frequency ``-n/2`` has no partner ``+n/2`` in the
        basis, so a planewave there is its own alias and cannot be assigned
        a mirror image. Measured, those rows -- and only those rows -- come
        back non-transverse to their own ``K``, by O(100) against a field
        of O(1). They are dropped: see :func:`_non_nyquist_mask`.
        """
        if getattr(self.op, "_t", None) is not None:
            X = np.asarray(X, dtype=complex)
            sq = X.ndim == 1
            if sq:
                X = X[:, None]
            out = self.op._t.mirror(X, self.axis, self.signs, self._keep)
            return out[:, 0] if sq else out
        g = self.op.grid
        X = np.asarray(X, dtype=complex) * self._keep[:, None]
        v = self.op.to_cartesian(X)                     # (N_G, 3, m)
        m = v.shape[-1]
        f = np.fft.ifftn(v.reshape(g + (3, m)), axes=(0, 1, 2))
        n_ax = g[self.axis]
        f = np.take(f, (-np.arange(n_ax)) % n_ax, axis=self.axis)
        f = f * self.signs[None, None, None, :, None]
        f = np.fft.fftn(f, axes=(0, 1, 2))
        out = self.op.to_frame(f.reshape(self.op.n_g, 3, m))
        return out * self._keep[:, None]

    def __call__(self, X):
        """Project onto the chosen sector: ``(I +- M)/2`` on the kept space.

        Idempotent and Hermitian, which ``(I +- M)/2`` alone is NOT here --
        M annihilates the Nyquist plane rather than squaring to the
        identity on it, so the mask has to be applied to the leading term
        too or the projector leaks half of every Nyquist component through.
        """
        X = np.asarray(X, dtype=complex) * self._keep[:, None]
        m = self.mirror(X)
        return 0.5 * (X + m) if self.even else 0.5 * (X - m)

    def expectation(self, vecs) -> np.ndarray:
        """``<h|M|h>`` per column, MPB's printed parity number.

        The columns are assumed normalised, which is what the eigensolver
        returns; the value is divided by ``<h|h>`` anyway so an unnormalised
        column still gives the right number rather than a scaled one.
        """
        V = np.asarray(vecs, dtype=complex)
        if V.ndim == 1:
            V = V[:, None]
        num = np.einsum("gm,gm->m", V.conj(), self.mirror(V))
        den = np.einsum("gm,gm->m", V.conj(), V)
        return np.real(num / np.where(np.abs(den) == 0, 1.0, den))


def _non_nyquist_mask(grid, axis: int) -> np.ndarray:
    """1.0 on planewaves that HAVE a mirror partner, 0.0 on the Nyquist plane.

    For an even ``n`` the FFT frequency set runs ``0, 1, ..., n/2-1, -n/2,
    ..., -1``. It is closed under negation everywhere except ``-n/2``,
    whose negative ``+n/2`` is not a member -- it aliases straight back.
    A planewave there is its own mirror image up to a reciprocal lattice
    vector, which is not the same thing as being even or odd, and carrying
    it through the projector produces a component that is not transverse to
    its own K and breaks both ``M^2 = I`` and idempotency.

    Dropping it is a real approximation, not a bookkeeping trick, so it is
    worth stating what it costs: it discards the least-resolved planewave
    in the expansion. On a grid that resolves the structure at all, the
    eigenvector's weight there is at the level of the discretization error
    that is already being accepted -- the hole-slab gate measures it. An
    odd ``n`` has no Nyquist entry and loses nothing.
    """
    keep = [np.ones(n) for n in grid]
    n_ax = grid[axis]
    if n_ax % 2 == 0:
        f = np.rint(np.fft.fftfreq(n_ax) * n_ax)
        keep[axis] = (f != -(n_ax // 2)).astype(float)
    m = keep[0][:, None, None] * keep[1][None, :, None] * keep[2][None, None, :]
    return np.repeat(m.ravel(), 2)


def zparity_operator(op, even: bool = True) -> MirrorParity:
    """``run_zeven`` / ``run_zodd``'s projector."""
    return MirrorParity(op, axis=2, even=even)


def yparity_operator(op, even: bool = True) -> MirrorParity:
    """``run_yeven`` / ``run_yodd``'s projector."""
    return MirrorParity(op, axis=1, even=even)
