# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W9 wave 1: full 3D Yee stepper — vacuum dipole LDOS gate (cross-checks
the W7.5 BOR result: both must give LDOS proportional to f^2)."""
import numpy as np

from meep_torch.core.ldos import ldos_spectrum
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                        step_3d)


def test_3d_vacuum_ldos_f_squared():
    res = 10
    n = 6 * res                                   # 6 um cube
    c = build_3d_coeffs(n, n, n, 1.0 / res, n_abs=res)
    st = init_3d_state(n, n, n)
    pulse = GaussianPulse(1.0, 0.4)
    src = ("Ez", n // 2, n // 2, n // 2)
    n_steps = int(round(30.0 / c.dt))
    series = np.empty(n_steps)
    for t in range(n_steps):
        st = step_3d(st, c, pulse.amplitude((t + 0.5) * c.dt), src)
        series[t] = st.Ez[src[1], src[2], src[3]].item()
    j = np.array([pulse.amplitude((t + 0.5) * c.dt) for t in range(n_steps)])
    band = np.linspace(0.85, 1.15, 5)
    ld = ldos_spectrum(series, j, c.dt, band)
    assert (ld > 0).all()
    r = ld / band ** 2
    spread = (r.max() - r.min()) / r.mean()
    assert spread <= 0.05, f"ldos/f^2 spread {spread:.4f} ({r})"
