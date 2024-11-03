# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2D near-to-far-field transform (surface equivalence, exact Hankel kernels).

Phasor convention: the DFT monitors accumulate int s(t) exp(-i w t) dt of the
real FDTD signal, which is the e^{+i w t} (engineering) phasor — outgoing
waves are Hankel H^(2). All kernels below follow that convention; meep's
green2d uses e^{-i w t}/H^(1), i.e. the complex conjugate.

Far H is taken from the local plane-wave relation H = R_hat x E (TM; dual for
TE), exact to O(1/kR) — fine for radiation patterns at kR >> 1.

Vacuum far region, meep units (eps0 = mu0 = 1, k = omega = 2 pi f).
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import hankel2


@dataclass
class Face:
    """One straight contour face with its frequency-domain fields.

    E: out-of-plane phasor on the face, shape (nfreq, N) — Ez for TM, Hz
    for TE. H: the tangential in-plane phasor the currents need — Hy on
    x-normal faces, Hx on y-normal (TM); Ey / Ex for TE.
    """

    pts: np.ndarray          # (N, 2) physical coordinates
    normal: tuple            # outward (nx, ny), one of (+-1, 0) / (0, +-1)
    dl: float
    E: np.ndarray            # (nfreq, N) complex
    H: np.ndarray            # (nfreq, N) complex


def line_source_fields_tm(k, pts, x0=0.0, y0=0.0):
    """Exact outgoing wave of a unit z electric line current at (x0, y0):
    Ez = (w/4) H0^(2)(kR); H = curl E / (-i w)."""
    dx = pts[:, 0] - x0
    dy = pts[:, 1] - y0
    R = np.hypot(dx, dy)
    w = k
    Ez = (w / 4.0) * hankel2(0, k * R)
    h1 = (1j * k / 4.0) * hankel2(1, k * R) / R
    Hx = -h1 * dy
    Hy = h1 * dx
    return Ez, Hx, Hy


def line_source_fields_te(k, pts, x0=0.0, y0=0.0):
    """Dual: unit z magnetic line current — Hz = (w/4) H0^(2)(kR)."""
    Hz, mHx, mHy = line_source_fields_tm(k, pts, x0, y0)
    return Hz, -mHx, -mHy    # (Hx, Hy) -> (-Ex, -Ey) under duality


def _far_out_of_plane(faces, ks, obs_pts, pol):
    """Out-of-plane far phasor (Ez for TM, Hz for TE) at obs_pts.

    Shape (nfreq, nobs). Sums, per face point: the z current radiating via
    H0^(2), and the in-plane current radiating via H1^(2).
    """
    ks = np.asarray(ks, dtype=float)
    obs = np.asarray(obs_pts, dtype=float)
    out = np.zeros((len(ks), len(obs)), dtype=complex)
    for face in faces:
        nx, ny = face.normal
        dxo = obs[:, None, 0] - face.pts[None, :, 0]     # (nobs, N)
        dyo = obs[:, None, 1] - face.pts[None, :, 1]
        R = np.hypot(dxo, dyo)
        for fi, k in enumerate(ks):
            w = k
            Fz = face.E[fi]                              # Ez (TM) / Hz (TE)
            Ht = face.H[fi]                              # Hy or Hx (TM); Ey or Ex (TE)
            if pol == "tm":
                # Jz = (n x H)_z ; M = -n x (Ez z): Mx = -ny Ez, My = nx Ez
                Jz = (nx * Ht) if nx else (-ny * Ht)
                cx, cy = -ny * Fz, nx * Fz               # (Mx, My)
                sign_ip = +1.0
            else:
                # Mz = -(n x E)_z ; J = n x (Hz z): Jx = ny Hz, Jy = -nx Hz
                Jz = (-nx * Ht) if nx else (ny * Ht)     # Mz
                cx, cy = ny * Fz, -nx * Fz               # (Jx, Jy)
                sign_ip = -1.0
            h0 = hankel2(0, k * R)
            h1 = hankel2(1, k * R) / R
            # overall minus: pinned by the analytic line-source self-test
            # (test_w4_green2d), which fixes the A/F potential sign convention
            out[fi] -= face.dl * (
                (w / 4.0) * h0 @ Jz
                + sign_ip * (1j * k / 4.0) * (h1 * dxo) @ cy
                - sign_ip * (1j * k / 4.0) * (h1 * dyo) @ cx
            )
    return out


def farfield_tm(faces, ks, obs_pts):
    """Far (E, H) 3-vectors, shapes (nfreq, nobs, 3). H = R_hat x E."""
    obs = np.asarray(obs_pts, dtype=float)
    Ez = _far_out_of_plane(faces, ks, obs, "tm")
    r = np.linalg.norm(obs, axis=1)
    rx, ry = obs[:, 0] / r, obs[:, 1] / r
    E = np.zeros(Ez.shape + (3,), dtype=complex)
    H = np.zeros_like(E)
    E[..., 2] = Ez
    H[..., 0] = Ez * ry
    H[..., 1] = -Ez * rx
    return E, H


def farfield_te(faces, ks, obs_pts):
    """Dual: far Hz; E = -R_hat x H."""
    obs = np.asarray(obs_pts, dtype=float)
    Hz = _far_out_of_plane(faces, ks, obs, "te")
    r = np.linalg.norm(obs, axis=1)
    rx, ry = obs[:, 0] / r, obs[:, 1] / r
    E = np.zeros(Hz.shape + (3,), dtype=complex)
    H = np.zeros_like(E)
    H[..., 2] = Hz
    E[..., 0] = -Hz * ry
    E[..., 1] = Hz * rx
    return E, H
