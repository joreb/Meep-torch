# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Half-ring coupler extraction -> analytic ring spectra.

The method (born in the device-suite r=50 um rings, where a full weakly-
guided ring needs >= 10 round trips of FDTD before resonances form):
simulate only the coupler half of the ring -- the side guides exit
vertically into the PML -- and measure

    t(f)      bus through amplitude          (run 1, bus-driven)
    kappa(f)  amplitude coupled into the ring (run 1)
    A(f)      arc-end to arc-end amplitude    (run 2, arc-driven)

then assemble spectra on an arbitrarily dense grid. Hard-won rules
baked in (each one was a measured bug first):

- ``a_half = A / |t|``: dividing by complex t strips a bus-propagation
  phase A never traveled (it tripled the apparent FSR); A's own phase
  IS the half-loop propagation phase, so FSR comes out group-index
  correct by construction.
- |t| (real) in every resonance denominator: the measured t phase is
  bus propagation between the ports, and putting it in the denominator
  produced |S21| > 1 lobes.
- passive clamp: |loop| may exceed 1 by measurement noise (A-noise
  divided by |t| ~ 1); clamp at 1 but assert the excess is < 5% --
  a large excess means the extraction is broken, not noisy.
- unitarity guard: |t|^2 + |kappa|^2 < 0.8 at band center means the
  extraction is INVALID (T_RUN truncation or port mis-capture).

``t_run_estimate`` sizes the runs: gaussian tail + arc transit +
margin -- template T_RUN values truncate silently on new geometry.
"""
import math

import numpy as np

__all__ = ["t_run_estimate", "unitarity_check", "bare_loop",
           "assemble_allpass", "assemble_adddrop"]


def t_run_estimate(fwidth: float, radius: float, n_group: float,
                   margin: float = 80.0) -> float:
    """Gaussian tail (2*t0 = 10/fwidth) + semicircle transit + margin."""
    return 2 * (5.0 / fwidth) + math.pi * radius * n_group + margin


def unitarity_check(t, kappa, threshold: float = 0.8) -> float:
    """Returns |t|^2 + |kappa|^2 at band center; raises if below threshold."""
    n = len(t) // 2
    u = float(abs(t[n]) ** 2 + abs(kappa[n]) ** 2)
    if u < threshold:
        raise ValueError(
            f"|t|^2+|kappa|^2 = {u:.3f} < {threshold} -- extraction INVALID "
            "(T_RUN truncation or port mis-capture)")
    return u


def bare_loop(A, t, max_excess: float = 0.05):
    """Bare full-loop factor a^2 = (A/|t|)^2, passive-clamped at |.| = 1."""
    loop = (np.asarray(A) / np.abs(t)) ** 2
    ex = float(np.abs(loop).max())
    if ex >= 1.0 + max_excess:
        raise ValueError(f"|loop| max {ex:.4f} -- extraction broken, not noise")
    return np.where(np.abs(loop) > 1.0, loop / np.abs(loop), loop)


def _dense(freqs, z, fd):
    m = np.interp(fd, freqs, np.abs(z))
    p = np.interp(fd, freqs, np.unwrap(np.angle(z)))
    return m * np.exp(1j * p)


def assemble_allpass(freqs, t, loop, n_dense: int = 6000):
    """All-pass |S21|(f) on a dense grid: (|t| - a^2)/(1 - |t| a^2)."""
    fd = np.linspace(freqs[0], freqs[-1], n_dense)
    ti = np.interp(fd, freqs, np.abs(t))
    ld = _dense(freqs, loop, fd)
    S21 = (ti - ld) / (1 - ti * ld)
    T = np.abs(S21) ** 2
    assert T.max() <= 1.0 + 1e-6, f"|S21|^2 max {T.max():.4f} > 1 -- assembly bug"
    return fd, S21


def assemble_adddrop(freqs, t, kappa, loop, n_dense: int = 6000):
    """Symmetric two-coupler add-drop (through, drop) on a dense grid."""
    fd = np.linspace(freqs[0], freqs[-1], n_dense)
    ti = np.interp(fd, freqs, np.abs(t))
    ki = np.interp(fd, freqs, np.abs(kappa))
    a_half = np.sqrt(np.abs(loop)) * np.exp(
        1j * np.unwrap(np.angle(loop)) / 2)
    ai = _dense(freqs, a_half, fd)
    D = 1 - ti ** 2 * ai ** 2
    S_thru = ti - ki ** 2 * ti * ai ** 2 / D
    S_drop = -ki ** 2 * ai / D
    s = np.abs(S_thru) ** 2 + np.abs(S_drop) ** 2
    assert s.max() <= 1.0 + 5e-3, f"thru+drop max {s.max():.4f} > 1"
    return fd, S_thru, S_drop
