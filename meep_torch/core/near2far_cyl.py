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
            contrib = F.cpu().numpy()[None, ...] * ph.reshape((-1,) + (1,) * F.dim())
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


def farfield_bor_m(planes, c, m, *, ir, jz0, jz1, thetas, k):
    """General-m far pattern |E_theta|^2 + |E_phi|^2, normalized (W7.9)."""
    Et, Ep = farfield_bor_m_complex(planes, c, m, ir=ir, jz0=jz0, jz1=jz1,
                                    thetas=thetas, k=k)
    p = np.abs(Et) ** 2 + np.abs(Ep) ** 2
    return p / p.max(axis=1, keepdims=True)


def farfield_bor_m_complex(planes, c, m, *, ir, jz0, jz1, thetas, k):
    """Complex (E_theta, E_phi) per frequency (un-normalized) -- the
    coherent building block for per-m source sums (off-axis dipoles).

    phi-integrals per current component (beta = k r' sin theta):
      plain:  A_m  = 2 pi i^m J_m(beta)
      cos:    -2 pi i^{m+1} J_m'(beta)
      sin:    +2 pi i^m (m / beta) J_m(beta)
    N_x picks (Cr cos - Cphi sin), N_y picks (Cr sin + Cphi cos),
    N_z picks the plain integral. Reduces to farfield_bor_m0 at m = 0.
    """
    from scipy.special import jv, jvp
    dr = c.dr
    nfreq = planes["Ez"].shape[0]
    out_t = np.empty((nfreq, len(thetas)), dtype=complex)
    out_p = np.empty((nfreq, len(thetas)), dtype=complex)
    zc = 0.5 * (jz0 + jz1) * dr
    ipm = 1j ** m

    def phi_ints(beta):
        with np.errstate(divide="ignore", invalid="ignore"):
            msin = np.where(beta > 0, m * jv(m, beta) / np.where(beta > 0, beta, 1.0),
                            0.5 if abs(m) == 1 else 0.0)
        return (2 * np.pi * ipm * jv(m, beta),
                -2 * np.pi * ipm * 1j * jvp(m, beta),
                2 * np.pi * ipm * msin)

    def add_face(N, L, rp, zp, J, M, ct, st):
        beta = k * rp * st
        ph = np.exp(1j * k * zp * ct)
        plain, ci, si = phi_ints(beta)
        for vec, C in ((N, J), (L, M)):
            Cr, Cp, Cz = C
            vec[0] += dr * np.sum(rp * (Cr * ci - Cp * si) * ph)
            vec[1] += dr * np.sum(rp * (Cr * si + Cp * ci) * ph)
            vec[2] += dr * np.sum(rp * Cz * plain * ph)

    for fi in range(nfreq):
        Er_p, Ez_p, Ep_p = planes["Er"][fi], planes["Ez"][fi], planes["Ephi"][fi]
        Hr_p, Hz_p, Hp_p = planes["Hr"][fi], planes["Hz"][fi], planes["Hphi"][fi]
        Et = np.zeros(len(thetas), dtype=complex)
        Ep_far = np.zeros(len(thetas), dtype=complex)
        for ti, th in enumerate(thetas):
            st, ct = np.sin(th), np.cos(th)
            N = np.zeros(3, dtype=complex)
            L = np.zeros(3, dtype=complex)
            # outer-r face (n = +r): J = (0, -Hz, Hphi), M = (0, Ez, -Ephi)
            j = np.arange(jz0, jz1)
            zp = (j + 0.5) * dr - zc
            rp = ir * dr
            hphi = 0.5 * (Hp_p[ir - 1, j] + Hp_p[min(ir, Hp_p.shape[0] - 1), j])
            hz = Hz_p[min(ir, Hz_p.shape[0] - 1) - 0, j] if ir < Hz_p.shape[0] \
                else Hz_p[ir - 1, j]
            add_face(N, L, rp, zp, (np.zeros_like(hphi), -hz, hphi),
                     (np.zeros_like(hphi), Ez_p[ir, j], -Ep_p[ir, j]), ct, st)
            # z faces (n = +-z): J = -+(Hphi, -Hr·?, 0) -- structure per spec,
            # overall sign pinned by the oracle gate
            for jz, sgn in ((jz1, +1.0), (jz0, -1.0)):
                i = np.arange(0, ir)
                rp2 = (i + 0.5) * dr
                zp2 = jz * dr - zc
                hphi_z = 0.5 * (Hp_p[i, jz - 1] + Hp_p[i, min(jz, Hp_p.shape[1] - 1)])
                hr_z = 0.5 * (Hr_p[i, jz - 1] + Hr_p[i, min(jz, Hr_p.shape[1] - 1)]) \
                    if Hr_p.shape[0] > ir else np.zeros_like(hphi_z)
                er_z = Er_p[i, jz]
                ep_z = Ep_p[i, jz]
                add_face(N, L, rp2, zp2,
                         (sgn * -hphi_z, sgn * hr_z, np.zeros_like(hphi_z)),
                         (sgn * ep_z, sgn * -er_z, np.zeros_like(er_z)), ct, st)
            Nt = N[0] * ct - N[2] * st
            Lt = L[0] * ct - L[2] * st
            Et[ti] = -(Nt + L[1])
            Ep_far[ti] = -(N[1] - Lt)
        out_t[fi] = Et
        out_p[fi] = Ep_far
    return out_t, out_p
