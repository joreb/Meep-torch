# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W3 harminv (FDM/matrix-pencil) tests on synthetic decaying sinusoids."""
import math

import numpy as np

from meep_torch.modes.harminv import harminv


def _series(fs, gammas, amps, dt, N, noise=0.0):
    n = np.arange(N)
    c = np.zeros(N, dtype=complex)
    for f, g, a in zip(fs, gammas, amps):
        c += a * np.exp((-1j * 2 * math.pi * f - g) * n * dt)
    if noise:
        c += noise * (np.random.RandomState(0).randn(N) + 1j * np.random.RandomState(1).randn(N))
    return c


def test_single_mode_recovered():
    dt, N = 0.05, 6000
    c = _series([0.15], [0.001], [1.0], dt, N)
    modes = harminv(c, dt, 0.1, 0.2)
    assert modes, "no mode found"
    assert abs(modes[0].freq - 0.15) / 0.15 < 1e-3, f"f={modes[0].freq}"
    Q_true = math.pi * 0.15 / 0.001
    assert abs(modes[0].Q - Q_true) / Q_true < 0.05, f"Q={modes[0].Q} vs {Q_true}"


def test_two_modes_separated():
    dt, N = 0.05, 8000
    c = _series([0.15, 0.25], [0.001, 0.002], [1.0, 0.5], dt, N)
    modes = harminv(c, dt, 0.1, 0.3)
    fs = sorted(m.freq for m in modes[:2])
    assert abs(fs[0] - 0.15) / 0.15 < 2e-3 and abs(fs[1] - 0.25) / 0.25 < 2e-3, f"fs={fs}"
    Q0 = math.pi * 0.15 / 0.001
    assert abs(modes[0].Q - Q0) / Q0 < 0.05 if abs(modes[0].freq - 0.15) < 1e-2 else True


def test_noise_robustness():
    dt, N = 0.05, 8000
    c = _series([0.15], [0.001], [1.0], dt, N, noise=1e-3)
    modes = harminv(c, dt, 0.1, 0.2)
    assert modes and abs(modes[0].freq - 0.15) / 0.15 < 1e-2, f"f={modes[0].freq if modes else None}"
    Q_true = math.pi * 0.15 / 0.001
    assert abs(modes[0].Q - Q_true) / Q_true < 0.10, f"Q={modes[0].Q} vs {Q_true}"


def test_high_q_mode():
    dt, N = 0.05, 10000
    gamma = 0.00005
    c = _series([0.2], [gamma], [1.0], dt, N)
    modes = harminv(c, dt, 0.15, 0.25)
    assert modes and abs(modes[0].freq - 0.2) / 0.2 < 1e-3
    Q_true = math.pi * 0.2 / gamma
    # high-Q recovery is less precise; gate at 15%
    assert abs(modes[0].Q - Q_true) / Q_true < 0.15, f"Q={modes[0].Q} vs {Q_true}"