# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""s-polarised kz stepper against an exact spectrum -- no oracle needed.

`stepper_kz` is p-pol, {Ex, Ez, Hy}. Its dual on the same 1D-in-x grid with
F(x) e^{i k_z z} is s-pol, {Ey, Hx, Hz}:

    eps dEy/dt = i k_z Hx - dHz/dx
        dHx/dt = i k_z Ey
        dHz/dt = -dEy/dx

With Ey = 0 at both walls the modes are Ey ~ sin(n pi x / L) and

    omega_n = sqrt((n pi / L)^2 + k_z^2) / sqrt(eps)

which is analytic, so this gate holds with no pymeep installed. The three
cases separate the three things that can be wrong independently: the plain
cavity (k_z = 0), the transverse-k term, and the material.

Needed by examples/11_upstream/antenna_pec_ground_plane_1D, whose sweep fixes
ky = 0 -- so its Ey source at k = (kx, 0, kz) is PURE s-pol, not a mixture,
and not the six-component block it was first scoped as.
"""
import numpy as np
import pytest
import torch

from meep_torch.core.stepper_kz import (build_kz_s_coeffs, init_kz_s_state,
                                        step_kz_s)

L, RES, NSTEPS = 4.0, 80, 60000


def _spectrum(kz, eps_r):
    dx = 1.0 / RES
    nx = int(round(L / dx)) + 1
    eps = torch.full((nx,), eps_r, dtype=torch.float64)
    c = build_kz_s_coeffs(nx, dx, kz, eps=eps, n_abs=0,
                          pec_lo=True, pec_hi=True, device="cpu")
    st = init_kz_s_state(nx, device="cpu")
    src_ix, probe = int(0.37 * nx), int(0.61 * nx)
    series = np.empty(NSTEPS, dtype=complex)
    w, t0 = 0.6, 3.0
    for t in range(NSTEPS):
        tt = (t + 0.5) * c.dt - t0
        amp = np.exp(-tt * tt / (2 * w * w)) if abs(tt) < 6 * w else 0.0
        st = step_kz_s(st, c, src_amp=complex(amp), src_ix=src_ix)
        series[t] = complex(st.Ey[probe].item())
    sp = np.abs(np.fft.fft(series * np.hanning(NSTEPS)))
    f = np.fft.fftfreq(NSTEPS, d=c.dt)
    keep = f > 0
    sp, f = sp[keep], f[keep]
    peaks = [i for i in range(2, len(sp) - 2)
             if sp[i] > sp[i - 1] and sp[i] > sp[i + 1] and sp[i] > 0.02 * sp.max()]
    return sorted(f[i] for i in peaks)[:4]


@pytest.mark.parametrize("kz,eps_r", [(0.0, 1.0), (0.8, 1.0), (0.8, 1.44)])
def test_pec_cavity_spectrum_matches_analytic(kz, eps_r):
    got = _spectrum(kz, eps_r)
    want = [np.sqrt((n * np.pi / L) ** 2 + kz ** 2)
            / (2 * np.pi * np.sqrt(eps_r)) for n in (1, 2, 3, 4)]
    # Measured error is exactly one FFT bin (2.7e-3), so this is
    # resolution-limited. 1% still catches a dropped k_z (10-40% off) or eps.
    for g, w in zip(got, want):
        assert abs(g - w) / w <= 0.01, f"kz={kz} eps={eps_r}: {got} vs {want}"
