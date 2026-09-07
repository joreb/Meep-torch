# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Point LDOS from source-field overlap (meep dft_ldos equivalent).

ldos(w) = -Re[E_hat(w) conj(J_hat(w))] / |J_hat(w)|^2

meep's sign, and for meep's reason: a source does positive work on the
field, dD/dt = curl H - J, so -Re[E J*] is the power radiated. This read
+Re[...] and blamed the phasor convention, but conjugating both phasors
leaves Re[E J*] alone -- Re[E* J] = Re[(E J*)*] = Re[E J*]. The + was in
fact compensating for the steppers injecting +J instead of -J; with the
injection sign fixed against meep's step.cpp, the compensator has to go
or LDOS comes out negative. FDTD
timing is baked into the DFT kernels: the soft source J is applied at
(t_idx + 1/2) dt, the recorded E sits at (t_idx + 1) dt. The meep
normalization constant is dropped — every shipped gate is a ratio
(Purcell, Q/Vmode, band shape), where it cancels.
"""
import math

import numpy as np


def ldos_spectrum(e_series, j_series, dt: float, freqs) -> np.ndarray:
    e = np.asarray(e_series, dtype=float)
    j = np.asarray(j_series, dtype=float)
    n = len(e)
    te = (np.arange(n) + 1.0) * dt
    tj = (np.arange(n) + 0.5) * dt
    out = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        w = 2 * math.pi * f
        e_hat = np.trapezoid(e * np.exp(-1j * w * te), te)
        j_hat = np.trapezoid(j * np.exp(-1j * w * tj), tj)
        out[i] = -np.real(e_hat * np.conj(j_hat)) / abs(j_hat) ** 2
    return out
