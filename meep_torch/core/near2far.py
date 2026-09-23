# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2D near-to-far-field transform (surface equivalence, exact Hankel kernels).

Phasor convention: these kernels are the e^{+i w t} (engineering) phasor,
outgoing waves Hankel H^(2); meep's green2d uses e^{-i w t}/H^(1), i.e. the
complex conjugate.

The DFT monitors USED to accumulate int s(t) exp(-i w t) dt, which is that
phasor directly. Since the monitors moved to meep's unitary convention they
accumulate exp(+i w t), so the stored phasor is now the CONJUGATE of what
these kernels expect and `faces_from_monitors` conjugates it back on the way
in. Conjugating a propagator reverses the outgoing wave, so getting this
wrong is an ANGULAR error, not a scale one: an Ez dipole that must be
isotropic came back 40 % anisotropic.

Far H is taken from the local plane-wave relation H = R_hat x E (TM; dual for
TE), exact to O(1/kR) — fine for radiation patterns at kR >> 1.

Vacuum far region, meep units (eps0 = mu0 = 1, k = omega = 2 pi f).
"""
from dataclasses import dataclass

import math

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
    w: np.ndarray = None     # (N,) quadrature weights; None = ones


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


def _far_out_of_plane(faces, ks, obs_pts, pol, eps=1.0, mu=1.0):
    """Out-of-plane far phasor (Ez for TM, Hz for TE) at obs_pts.

    Shape (nfreq, nobs). Sums, per face point: the z current radiating via
    H0^(2), and the in-plane current radiating via H1^(2).

    ``ks`` is the VACUUM ``2 pi f``. The medium enters twice and differently,
    which is the whole point of separating them:

        Hankel argument   k = omega sqrt(eps mu)      (the wavenumber)
        electric current  omega mu / 4                (TM)
        magnetic current  omega eps / 4               (TE, the dual)

    meep does the same at ``src/near2far.cpp:217-228`` and reads eps/mu at
    each region's centre in ``add_dft_near2far``, erroring if the regions
    disagree.

    This was vacuum-only: ``k`` was used for BOTH the Hankel argument and the
    current coefficient, so every far field computed in a non-unit background
    was wrong -- not by a scale, which a normalised pattern would hide, but in
    SHAPE. A single dipole in n = 1.2, which must radiate isotropically in 2D,
    came out with 8 spurious lobes and an 82% spread; at n = 1.0 the same
    probe is flat to 0.2%. The phase error across a box of half-width L is
    ``2 pi L (n - 1) / lambda``, so the faces stop summing coherently as soon
    as that approaches a cycle.
    """
    ks = np.asarray(ks, dtype=float)
    obs = np.asarray(obs_pts, dtype=float)
    out = np.zeros((len(ks), len(obs)), dtype=complex)
    n_med = math.sqrt(eps * mu)
    for face in faces:
        nx, ny = face.normal
        dxo = obs[:, None, 0] - face.pts[None, :, 0]     # (nobs, N)
        dyo = obs[:, None, 1] - face.pts[None, :, 1]
        R = np.hypot(dxo, dyo)
        quad = face.w if face.w is not None else 1.0
        for fi, omega in enumerate(ks):
            k = omega * n_med                 # wavenumber IN THE MEDIUM
            w = omega * (mu if pol == "tm" else eps)
            Fz = face.E[fi] * quad                       # Ez (TM) / Hz (TE)
            Ht = face.H[fi] * quad                       # Hy or Hx (TM); Ey or Ex (TE)
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


def _rhat(faces, obs):
    """Radial direction from the contour centroid (frame-independent)."""
    c = np.mean([f.pts.mean(axis=0) for f in faces], axis=0)
    d = obs - c
    r = np.linalg.norm(d, axis=1)
    return d[:, 0] / r, d[:, 1] / r


def farfield_tm(faces, ks, obs_pts, eps=1.0, mu=1.0):
    """Far (E, H) 3-vectors, shapes (nfreq, nobs, 3). H = R_hat x E / Z.

    ``Z = sqrt(mu / eps)`` is the wave impedance OF THE MEDIUM, 1 only in
    vacuum. A normalised radiation pattern cannot see this -- it is one
    uniform factor on |E x H*| -- but an absolute far flux can, and did.
    """
    obs = np.asarray(obs_pts, dtype=float)
    Ez = _far_out_of_plane(faces, ks, obs, "tm", eps=eps, mu=mu)
    rx, ry = _rhat(faces, obs)
    inv_Z = math.sqrt(eps / mu)
    E = np.zeros(Ez.shape + (3,), dtype=complex)
    H = np.zeros_like(E)
    E[..., 2] = Ez
    H[..., 0] = Ez * ry * inv_Z
    H[..., 1] = -Ez * rx * inv_Z
    return E, H


def farfield_te(faces, ks, obs_pts, eps=1.0, mu=1.0):
    """Dual: far Hz; E = -Z R_hat x H."""
    obs = np.asarray(obs_pts, dtype=float)
    Hz = _far_out_of_plane(faces, ks, obs, "te", eps=eps, mu=mu)
    rx, ry = _rhat(faces, obs)
    Z = math.sqrt(mu / eps)
    E = np.zeros(Hz.shape + (3,), dtype=complex)
    H = np.zeros_like(E)
    H[..., 2] = Hz
    E[..., 0] = -Hz * ry * Z
    E[..., 1] = Hz * rx * Z
    return E, H


def near2far_monitors_tm(grid, freqs, *, ix0, ix1, iy0, iy1):
    """FluxMonitorTM box contour: [(monitor, outward_normal)] for the four
    faces of the index box [ix0, ix1] x [iy0, iy1] (faces ON those lines)."""
    from .monitors import FluxMonitorTM
    # faces include BOTH corner endpoints (hi+1) and get trapezoid weights in
    # faces_from_accs, closing the loop integral (corners shared 1/2 + 1/2)
    return [
        (FluxMonitorTM(grid, freqs, x_normal=True, idx=ix1, lo=iy0, hi=iy1 + 1), (+1, 0)),
        (FluxMonitorTM(grid, freqs, x_normal=True, idx=ix0, lo=iy0, hi=iy1 + 1), (-1, 0)),
        (FluxMonitorTM(grid, freqs, x_normal=False, idx=iy1, lo=ix0, hi=ix1 + 1), (0, +1)),
        (FluxMonitorTM(grid, freqs, x_normal=False, idx=iy0, lo=ix0, hi=ix1 + 1), (0, -1)),
    ]


def faces_from_accs(grid, mons, accs):
    """Assemble Face objects from near2far_monitors_* + run accumulators.

    mons entries are (monitor, normal) [TM: points inferred from the monitor
    indices] or (monitor, normal, pts). acc layout per monitor:
    (nfreq, 4, N) = dt-summed (ReE, ImE, ReH, ImH); Re - i Im is the phasor
    these kernels expect (the monitors store meep's conjugate).
    """
    dx = grid.dx
    faces = []
    for entry, acc in zip(mons, accs):
        if len(entry) == 3:
            mon, normal, pts = entry
        else:
            mon, normal = entry
            idxs = np.arange(mon.lo, mon.hi)
            if mon.x_normal:
                pts = np.stack([np.full(len(idxs), mon.idx * dx), idxs * dx], axis=1)
            else:
                pts = np.stack([idxs * dx, np.full(len(idxs), mon.idx * dx)], axis=1)
        a = acc.detach().cpu().numpy() if hasattr(acc, "detach") else np.asarray(acc)
        quad = np.ones(len(pts))
        quad[0] = quad[-1] = 0.5
        # conj: the monitors store meep's e^{+i w t}; these kernels are
        # e^{-i w t}/H^(2). See the module docstring.
        Fz = a[:, 0] - 1j * a[:, 1]          # monitor E slot (Ez TM / Ey|Ex TE)
        Ft = a[:, 2] - 1j * a[:, 3]          # monitor H slot (Hy|Hx TM / Hz TE)
        # TE: Face.E is the out-of-plane Hz, so the two slots swap. Keyed on
        # an EXPLICIT marker: this was `not hasattr(mon, "x_normal")`, i.e. an
        # attribute's ABSENCE carrying the polarisation. Merging the two TE
        # monitors into one class gave them an `x_normal` and silently stopped
        # the swap -- the antenna pattern went from rms 0.006 to 0.279 against
        # cos^2 with no error anywhere.
        if getattr(mon, "pol", "tm") == "te":
            Fz, Ft = Ft, Fz
        faces.append(Face(pts=pts, normal=normal, dl=dx, E=Fz, H=Ft, w=quad))
    return faces


def tile_faces(faces, n_periods: int, Ly: float, ky: float = 0.0):
    """Periodic near2far (meep nperiods): replicate contour faces n_periods
    times along y at pitch Ly, each copy phased by exp(i ky Ly n). Centered
    tiling (n in [-(N-1)/2, ...]); pass the result to farfield_*."""
    out = []
    n0 = -(n_periods - 1) / 2.0
    for n in range(n_periods):
        shift = (n0 + n) * Ly
        ph = np.exp(1j * ky * shift)
        for f in faces:
            pts = f.pts.copy()
            pts[:, 1] = pts[:, 1] + shift
            out.append(Face(pts=pts, normal=f.normal, dl=f.dl,
                            E=f.E * ph, H=f.H * ph, w=f.w))
    return out
