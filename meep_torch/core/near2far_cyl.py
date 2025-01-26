# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Cylindrical (BOR) near-to-far transform, m = 0 (W7.75).

Surface equivalence on an (r, z) box of BOR DFT planes; axisymmetric
currents radiate through the Bessel-weighted far-zone integrals
(beta = k r' sin theta, z-phase e^{i k z' cos theta}):

  Nx = 2 pi i INT dl r' Jr J1(beta) ph      (z faces)
  Nz = 2 pi   INT dl r' Jz J0(beta) ph      (outer-r face)
  Ly = 2 pi i INT dl r' Mphi J1(beta) ph    (all faces)
  N_theta = Nx cos theta - Nz sin theta;  E_theta ~ -(N_theta + Ly)

Pattern = |E_theta|^2 normalized (m = 0, Ez-driven: E_phi = 0). The
m != 0 generalization (J_m / J_m' integrals) is W7.9.
"""
import numpy as np
from scipy.special import j0, j1


def run_bor_dft(state, c, n_steps, *, src, src_profile, freqs):
    """Step run_bor-style while accumulating e^{-i w t} DFT planes of
    (Er, Ez, Ephi, Hr, Hphi, Hz). E sampled at (t+1) dt, H at (t+1/2) dt.
    Returns {comp: (nfreq, ...) complex}."""
    from .stepper_bor import step_bor
    freqs = np.asarray(freqs, dtype=float)
    acc = {k: None for k in ("Er", "Ez", "Ephi", "Hr", "Hphi", "Hz")}
    for t in range(n_steps):
        state = step_bor(state, c, src_profile.amplitude((t + 0.5) * c.dt), src)
        for comp in acc:
            F = getattr(state, comp)
            off = 1.0 if comp.startswith("E") else 0.5
            ph = np.exp(-2j * np.pi * freqs * (t + off) * c.dt) * c.dt
            contrib = F.numpy()[None, ...] * ph.reshape((-1,) + (1,) * F.dim())
            acc[comp] = contrib if acc[comp] is None else acc[comp] + contrib
    return state, acc


def farfield_bor_m0(planes, c, *, ir, jz0, jz1, thetas, k):
    """Normalized |E_theta|^2 pattern from the (r, z) index box
    [0, ir] x [jz0, jz1] for each frequency plane (m = 0)."""
    dr = c.dr
    nfreq = planes["Ez"].shape[0]
    out = np.empty((nfreq, len(thetas)))
    z0 = 0.0                                     # phases relative to box center
    zc = 0.5 * (jz0 + jz1) * dr
    for fi in range(nfreq):
        Ez = planes["Ez"][fi]
        Er = planes["Er"][fi]
        Hphi = planes["Hphi"][fi]
        Et = np.zeros(len(thetas), dtype=complex)
        for ti, th in enumerate(thetas):
            st, ct = np.sin(th), np.cos(th)
            # outer-r face (n = +r): Jz = Hphi, Mphi = Ez
            j = np.arange(jz0, jz1)
            zp = (j + 0.5) * dr - zc
            rp = ir * dr
            hphi_f = 0.5 * (Hphi[ir - 1, j] + Hphi[ir, j]) if ir < Hphi.shape[0] \
                else Hphi[ir - 1, j]
            ez_f = Ez[ir, j]
            ph = np.exp(1j * k * zp * ct)
            beta = k * rp * st
            Nz = 2 * np.pi * dr * np.sum(rp * hphi_f * j0(beta) * ph)
            Ly = 2j * np.pi * dr * np.sum(rp * ez_f * j1(beta) * ph)
            # z faces (n = +-z): Jr = -+ Hphi, Mphi = -+ (-Er) -> top: Jr=-Hphi, Mphi=-Er...
            # structure per spec; overall sign pinned by the sin^2 gate
            Nx = 0.0 + 0.0j
            for jz, sgn in ((jz1, +1.0), (jz0, -1.0)):
                i = np.arange(0, ir)
                rp2 = (i + 0.5) * dr
                zp2 = jz * dr - zc
                hphi_z = 0.5 * (Hphi[i, jz - 1] + Hphi[i, min(jz, Hphi.shape[1] - 1)])
                er_z = Er[i, jz]
                ph2 = np.exp(1j * k * zp2 * ct)
                b2 = k * rp2 * st
                Nx += sgn * (-1.0) * 2j * np.pi * dr * np.sum(rp2 * hphi_z * j1(b2)) * ph2
                Ly += sgn * (-1.0) * 2j * np.pi * dr * np.sum(rp2 * er_z * j1(b2)) * ph2
            Nt = Nx * ct - Nz * st
            Et[ti] = -(Nt + Ly)
        p = np.abs(Et) ** 2
        out[fi] = p / p.max()
    return out
