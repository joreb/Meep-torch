# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2-deferral close: k_z 2.5D stepper — out-of-plane Fresnel (p-pol) gate."""
import math

import numpy as np
import torch

from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_kz import build_kz_coeffs, init_kz_state, step_kz

FCEN = 1.0
THETA = math.radians(40.0)
N2 = 1.5
NX, DX, NABS = 800, 0.05, 40


def _T(with_slab):
    kz = 2 * math.pi * FCEN * math.sin(THETA)
    eps = torch.ones(NX, dtype=torch.float64)
    if with_slab:
        eps[400:] = N2 ** 2                       # half-space x > 20
    c = build_kz_coeffs(NX, DX, kz, eps=eps, n_abs=NABS)
    st = init_kz_state(NX)
    pulse = GaussianPulse(FCEN, 0.2)
    n_steps = int(round(60.0 / c.dt))
    sr = np.empty(n_steps, dtype=complex)
    for t in range(n_steps):
        st = step_kz(st, c, pulse.amplitude((t + 0.5) * c.dt), 100)
        sr[t] = st.Ez[300].item()                 # upstream probe (x=15)
    return sr, c.dt


def test_kz_fresnel_p():
    s1, dt = _T(True)
    s0, _ = _T(False)
    t = np.arange(len(s1)) * dt
    kern = np.exp(-2j * math.pi * FCEN * t)
    inc = np.trapezoid(s0 * kern, t)
    refl = np.trapezoid((s1 - s0) * kern, t)
    R = abs(refl / inc) ** 2
    cos_i = math.cos(THETA)
    cos_t = math.sqrt(1.0 - (math.sin(THETA) / N2) ** 2)
    r_p = (N2 * cos_i - cos_t) / (N2 * cos_i + cos_t)
    R_an = r_p ** 2
    assert abs(R - R_an) <= 0.02, f"R={R:.4f} vs Fresnel p {R_an:.4f}"
