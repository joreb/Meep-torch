# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Harminv replacement: filter-diagonalization / matrix-pencil (Mandelshtam).

Given a real or complex time series c_n = sum_k a_k exp(-i omega_k n dt)
(omega_k = 2*pi*f_k - i*gamma_k), recover the modes (f_k, Q_k, |a_k|) in a
frequency band. Pure scipy (no meep).

Method (matrix-pencil, the FDM variant): build two shifted Hankel matrices
X0 (c[0..]) and X1 (c[1..]); SVD-truncate X0 to rank R (number of modes);
project the pencil X1 = X0 Z to an R x R generalized eigenproblem; its
eigenvalues are z_k = exp(-i omega_k dt), from which f_k = Re(omega_k)/(2*pi)
and gamma_k = -Im(omega_k), Q_k = pi*|f_k|/gamma_k. Amplitudes from a least-
squares fit of the modes to the series.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class HarminvMode:
    freq: float
    Q: float
    amp: float


def harminv(series, dt: float, fmin: float = 0.0, fmax: float | None = None,
            max_modes: int = 64, decay_threshold: float = 0.0) -> list[HarminvMode]:
    """Return the resonant modes in [fmin, fmax] sorted by |amp| (descending).

    ``series`` is a 1D real or complex array; ``dt`` the sample step. Modes with
    f outside [fmin, fmax] (default fmax = Nyquist) or with gamma <= 0 (growing)
    are dropped; ``decay_threshold`` drops modes with Q below it.
    """
    c = np.asarray(series, dtype=np.complex128)
    N = len(c)
    if fmax is None:
        fmax = 0.5 / dt
    L = min(N // 2, max_modes * 4)            # pencil window
    cols = N - L                              # lag range
    if L < 2 or cols < 2:
        return []
    X0 = np.array([[c[i + j] for j in range(cols)] for i in range(L)])
    X1 = np.array([[c[i + j + 1] for j in range(cols)] for i in range(L)])
    U, S, Vh = np.linalg.svd(X0, full_matrices=False)
    # rank: keep singular values above a relative floor (noise cutoff)
    R = int((S > 1e-7 * S[0]).sum())
    R = max(1, min(R, max_modes))
    U_R, S_R, V_R = U[:, :R], S[:R], Vh[:R].conj().T
    # pencil: A = S^{-1} U^H X1 V  (R x R); eig(A) = z_k
    A = (U_R.conj().T @ X1 @ V_R) / S_R[None, :]
    z = np.linalg.eigvals(A)
    omega = 1j * np.log(z) / dt              # omega = 2*pi*f - i*gamma
    f = np.real(omega) / (2 * np.pi)
    gamma = -np.imag(omega)
    # amplitudes: least-squares fit of exp(-i omega_k n dt) to the series
    n = np.arange(N)
    B = np.exp(-1j * np.outer(omega, n))      # (R, N)
    a, *_ = np.linalg.lstsq(B.T, c, rcond=None)
    modes = []
    for k in range(R):
        if f[k] < fmin or f[k] > fmax:
            continue
        if gamma[k] <= 0:
            continue                         # growing mode -> unstable/noise
        Q = np.pi * abs(f[k]) / gamma[k]
        if decay_threshold and Q < decay_threshold:
            continue
        modes.append(HarminvMode(freq=float(f[k]), Q=float(Q), amp=float(abs(a[k]))))
    modes.sort(key=lambda m: -m.amp)
    return modes