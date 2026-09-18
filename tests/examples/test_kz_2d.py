# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2-deferral close: k_z 2.5D stepper -- out-of-plane Fresnel (p-pol) gates.

Two boundary modes are gated:
  * absorber (n_abs): a 40 deg Fresnel point (the wave-1 matched-sigma
    absorber, regression for the W2 close).
  * kz-aware CPML (npml): the grazing sweep to 80 deg -- the matched-sigma
    absorber reflects near grazing (x wavelength outgrows the graded
    layer), the CFS-CPML is impedance-matched at every angle.
"""
import math

import numpy as np
import torch

from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_kz import build_kz_coeffs, init_kz_state, step_kz

FCEN = 1.0
N2 = 1.5
NX, DX = 800, 0.05


def _R(theta, with_slab, **kw):
    kz = 2 * math.pi * FCEN * math.sin(theta)
    eps = torch.ones(NX, dtype=torch.float64)
    if with_slab:
        eps[400:] = N2 ** 2                       # half-space x > 20
    c = build_kz_coeffs(NX, DX, kz, eps=eps, **kw)
    st = init_kz_state(NX, npml=kw.get("npml", 0))
    pulse = GaussianPulse(FCEN, 0.2)
    n_steps = int(round(60.0 / max(math.cos(theta), 0.15) / c.dt))
    sr = np.empty(n_steps, dtype=complex)
    for t in range(n_steps):
        st = step_kz(st, c, pulse.amplitude((t + 0.5) * c.dt), 100)
        sr[t] = st.Ez[300].item()                 # upstream probe (x=15)
    return sr, c.dt


def _fresnel_p(theta):
    cos_i = math.cos(theta)
    cos_t = math.sqrt(1.0 - (math.sin(theta) / N2) ** 2)
    return ((N2 * cos_i - cos_t) / (N2 * cos_i + cos_t)) ** 2


def _refl(theta, **kw):
    s1, dt = _R(theta, True, **kw)
    s0, _ = _R(theta, False, **kw)
    t = np.arange(len(s1)) * dt
    kern = np.exp(-2j * math.pi * FCEN * t)
    return abs(np.trapezoid((s1 - s0) * kern, t)
               / np.trapezoid(s0 * kern, t)) ** 2


def test_kz_fresnel_p_absorber():
    # wave-1 matched-sigma absorber at 40 deg (W2 close regression).
    R = _refl(math.radians(40.0), n_abs=40)
    R_an = _fresnel_p(math.radians(40.0))
    assert abs(R - R_an) <= 0.02, f"absorber R={R:.4f} vs Fresnel p {R_an:.4f}"


def test_kz_fresnel_p_cpml_grazing():
    # kz-aware CFS-CPML: Fresnel sweep 0..80 deg, max |R - Fresnel| <= 0.02.
    # The absorber path caps at ~65 deg (grazing reflection); the CPML fixes it.
    for deg in (0, 20, 40, 60, 70, 75, 80):
        R = _refl(math.radians(deg), npml=50, pml_m=3,
                  pml_R0=1e-10, pml_alpha=0.4)
        R_an = _fresnel_p(math.radians(deg))
        assert abs(R - R_an) <= 0.02, (
            f"CPML {deg} deg R={R:.4f} vs Fresnel p {R_an:.4f} "
            f"(d={abs(R - R_an):.4f})")


def test_the_kz_coupling_carries_the_sign_of_kz():
    """`dDx/dt = -i k_z Hy`, including the sign, which Fresnel cannot see.

    WHY THIS EXISTS. The two tests above gate |R| against analytic Fresnel,
    including near grazing -- a real gate, and blind to this. `k_z -> -k_z` is
    the opposite propagation direction and |R| is invariant under it, so
    flipping the sign of the `i k_z` coupling leaves both passing; verified by
    injecting exactly that defect.

    Seed a known Hy, take one step, read the Ex it induces, and compare with
    the equation in the module docstring rather than with a recorded number.
    """
    import torch
    from meep_torch.core.stepper_kz import (build_kz_coeffs, init_kz_state,
                                            step_kz)

    nx, dx, ix = 60, 1.0 / 20, 30

    def induced(kz):
        c = build_kz_coeffs(nx, dx, kz, eps=None, n_abs=0)
        st = init_kz_state(nx)
        hy = torch.zeros(nx - 1, dtype=torch.complex128)
        hy[ix] = 1.0
        return complex(step_kz(st._replace(Hy=hy), c, 0.0, None).Ex[ix]), c.dt

    for kz in (0.5, 2.0):
        got, dt = induced(kz)
        want = -1j * kz * dt                         # -i k_z Hy * dt, eps = 1
        assert abs(got - want) < 1e-12, f"kz={kz}: {got} vs {want}"
        neg, _ = induced(-kz)
        assert abs(neg + want) < 1e-12, f"kz={-kz} did not flip the sign"

    half, _ = induced(0.5)
    two, _ = induced(2.0)
    assert abs(two - 4 * half) < 1e-12, "not linear in k_z"
