# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Mode fields for the 3D planewave backend.

The 2D field layer in :mod:`meep_torch.mpb.fields` reconstructs from the
FDFD eigenvectors on a Yee grid. This one reconstructs from the planewave
coefficients, which is both simpler and exact: every operator in the chain
is diagonal in one basis or the other, so there is no interpolation step
and nothing to get half a cell wrong.

    H_G   = h_G1 t1(G) + h_G2 t2(G)        (frame -> cartesian)
    D_G   = (K_G x H_G) / omega            (curl H = -i omega D, c = mu = 1)
    H(r)  = sum_G H_G exp(i K_G . r)       (inverse FFT, plus the Bloch phase)
    E(r)  = eta(r) . D(r)                  (pointwise, the TENSOR eta)

``B = H`` because mu is 1 everywhere; they stay separate entry points
because the field-type letter differs and that is what the energy
functions and the output filenames key on.

Normalisation is MPB's: the cell integral of ``|H|^2`` is 1. In the
planewave coefficients that is just ``||h|| = 1/sqrt(volume)`` by
Parseval, with no field reconstruction needed -- the same identity the 2D
module leans on, and the reason ``compute_field_energy`` can be checked
against the eigenvalue without a quadrature.

An ``omega = 0`` band (the two at Gamma) comes back with D and E zero and
H a uniform field: that band is a static curl-free H, so there is no
displacement current for the formulation to hold on to. That is the honest
answer rather than a divide-by-zero guard.
"""
from __future__ import annotations

import numpy as np

__all__ = ["planewave_fields_3d", "normalize_3d"]


def normalize_3d(vecs: np.ndarray, volume: float) -> np.ndarray:
    """Scale coefficient columns so the cell integral of |H|^2 is 1."""
    nrm = np.linalg.norm(vecs, axis=0) * np.sqrt(volume)
    return vecs / np.where(nrm == 0, 1.0, nrm)[None, :]


def planewave_fields_3d(op, coeffs, omega: float, volume: float,
                        envelope: bool = False, r=None):
    """``(E, D, H)`` on the real-space grid, each ``(n1, n2, n3, 3)`` complex.

    ``op`` is the :class:`~meep_torch.mpb.planewave3d.ThetaOperator` the
    band was solved with -- passed rather than rebuilt so the frame, the K
    vectors and the eta tensor cannot drift from the ones the eigenvalue
    belongs to.

    ``envelope=True`` returns the periodic part; the default carries the
    ``exp(i kc . r)`` Bloch factor, which needs ``r`` (the cartesian
    positions from :func:`~meep_torch.mpb.planewave3d.lattice_grid_3d`).
    """
    h = np.asarray(coeffs, dtype=complex).reshape(-1, 1)
    h = normalize_3d(h, volume)
    g = op.grid

    H_G = op.to_cartesian(h)[:, :, 0]                    # (N_G, 3)
    if omega > 0:
        D_G = np.cross(op.K, H_G) / omega
    else:
        D_G = np.zeros_like(H_G)

    def to_real(F):
        # coefficients -> sum_G F_G exp(i G . r): N * ifftn, and the N
        # cancels against nothing here, so it is written explicitly.
        return np.fft.ifftn(F.reshape(g + (3,)), axes=(0, 1, 2)) * op.n_g

    H = to_real(H_G)
    D = to_real(D_G)
    E = np.einsum("xyzij,xyzj->xyzi", op.eta, D)

    if envelope:
        return E, D, H
    if r is None:
        raise ValueError("bloch_phase=True needs the cartesian positions r")
    kc = op.K[0]                                   # G = 0 row is kc exactly
    phase = np.exp(1j * (np.asarray(r) @ kc))[..., None]
    return E * phase, D * phase, H * phase
