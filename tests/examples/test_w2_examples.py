# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2 Wave 4 example gates: oblique-source (angle), refl-angular (Fresnel R(theta)),
oblique-planewave (angle, in test_w2_bloch)."""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.sources import GaussianPulse, ObliqueLineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.core.stepper_bloch import bloch_xflux, init_bloch_tm_state, step_bloch_tm


def _oblique_angle(theta_deg, fcen=0.3, dx=0.05):
    """Launch an ObliqueLineSource in a PML box; return the measured angle (deg)."""
    theta = math.radians(theta_deg)
    g = Grid(nx=300, ny=300, dx=dx)
    npml = 20
    src = ObliqueLineSource(g, ix=npml + 5, j0=npml, j1=g.ny - npml,
                            profile=GaussianPulse(fcen, 0.1), theta=theta, fcen=fcen)
    c = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device), npml=npml)
    st = init_tm_state(g)
    n_steps = int(round(60.0 / g.dt))
    for t in range(n_steps):
        st = step_tm(st, c, src.amplitude(t), src.mask)
    col = g.nx - npml - 30
    Ez_y = st.Ez[col, npml:g.ny - npml].cpu().numpy()
    spec = np.abs(np.fft.rfft(Ez_y - Ez_y.mean(), n=8192))
    kys = np.fft.rfftfreq(8192, dx) * 2 * math.pi
    k_meas = float(kys[int(spec.argmax())])
    return math.degrees(math.asin(min(1.0, k_meas / (2 * math.pi * fcen))))


def test_oblique_source_angle():
    # ObliqueLineSource is a soft source over a finite aperture (PML box), not a
    # Bloch eigenmode -- the finite aperture broadens the k-spectrum, so the
    # angle gate is looser than the Bloch oblique-planewave gate (1 deg).
    for theta in (20.0, 25.0):
        assert abs(_oblique_angle(theta) - theta) < 2.0, \
            f"oblique-source angle off at {theta}deg"


def _bloch_refl_R(theta_deg, n2=2.0, fcen=0.3, dx=0.04):
    """Vacuum->dielectric interface, Bloch y at theta. Pulse + time-integrated
    transmitted flux avoids the CW soft-source/reflected-wave contamination."""
    theta = math.radians(theta_deg)
    k0 = 2 * math.pi * fcen
    ky = k0 * math.sin(theta)
    Ly = (2 * math.pi / ky) if ky else 4.0
    ny = max(20, int(round(Ly / dx)))
    nx = 260
    g = Grid(nx=nx, ny=ny, dx=dx)
    npml = 18
    ix_if = nx // 2
    ix_src = npml + 5
    ix_mon = nx - npml - 15
    w = 1.0 / (0.5 * fcen)
    t0 = 6.0 * w
    omega = k0

    def run(diel):
        eps = torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device)
        if diel:
            eps[ix_if:, :] = n2 * n2
        c = build_tm_coeffs(g, eps_inf=eps, npml=npml, y_pmc=True)
        st = init_bloch_tm_state(g)
        mask = g.zeros(g.nx, g.ny); mask[ix_src, :] = 1.0
        n_steps = int(round(80.0 / g.dt))
        flux_sum = 0.0
        for t in range(n_steps):
            tt = (t + 0.5) * g.dt
            genv = math.exp(-((tt - t0) ** 2) / (2 * w * w))
            env_c = genv * math.cos(omega * tt)
            env_s = -genv * math.sin(omega * tt)
            st = step_bloch_tm(st, c, ky, src_c=env_c, src_s=env_s, src_mask=mask)
            flux_sum += bloch_xflux(st, ix_mon, dx)
        return flux_sum

    ref = run(False)
    slab = run(True)
    T = slab / ref if ref != 0 else 0.0
    return 1.0 - T


def _fresnel_s_R(theta_deg, n1=1.0, n2=2.0):
    th1 = math.radians(theta_deg)
    sin2 = (n1 / n2) * math.sin(th1)
    cos2 = math.sqrt(max(0.0, 1 - sin2 * sin2))
    r = (n1 * math.cos(th1) - n2 * cos2) / (n1 * math.cos(th1) + n2 * cos2)
    return r * r


def test_refl_angular_fresnel():
    errs = []
    for theta in (10.0, 20.0, 30.0, 40.0):
        R = _bloch_refl_R(theta)
        Ra = _fresnel_s_R(theta)
        errs.append((theta, R, Ra, abs(R - Ra)))
    for theta, R, Ra, e in errs:
        assert e < 0.02, f"refl-angular R({theta})={R:.4f} vs Fresnel {Ra:.4f} (off {e:.4f})"
    print("refl-angular:", [(t, f"R={R:.3f}", f"F={Ra:.3f}") for t, R, Ra, _ in errs])