# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D near-to-far-field transform (surface equivalence, e^{+iwt} phasor).

3D analog of core/near2far.py: equivalent surface currents J = n x H,
M = -n x E on the six DFT'd faces of a closed box radiate through the
outgoing scalar Green's function g = e^{-ikR} / (4 pi R) (conjugate of the
physics e^{-iwt} e^{+ikR}, matching this repo's DFT monitors which
accumulate int s(t) e^{-i w t} dt). Far zone (kR >> 1):

  A = INT J g dS,  F = INT M g dS
  E = -ik [ A - rhat (rhat . A) - rhat x F ] g-factored
  H = rhat x E / Z                    (Z = sqrt(mu/eps), exact to O(1/kR))

Signs pinned by the analytic point-dipole self-test
(test_near2far_3d.py::test_green3d_selftest).

``Near2FarBox3D`` is the run-time DFT accumulator (duck-typed against
run_3d / run_3d_lean: ``.sample(state, t)``, ``reduce="sum"``); each face's
tangential E/H are co-located to the face cell centers by midpoint
averaging, so the surface integral is a pure midpoint rule.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

_Comp3D = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_TANG = {0: (1, 2), 1: (0, 2), 2: (0, 1)}      # face axis -> transverse axes


@dataclass
class Face3D:
    """One box face (axis-aligned plane) with frequency-domain fields.

    E/H: full 3-vector phasors at the face points, shape (nfreq, N, 3)
    (normal components are zero — only tangential enter the currents)."""
    pts: np.ndarray          # (N, 3) physical coordinates
    normal: tuple            # outward unit axis vector, e.g. (+1, 0, 0)
    dA: float
    E: np.ndarray            # (nfreq, N, 3) complex
    H: np.ndarray            # (nfreq, N, 3) complex


def dipole_fields_3d(k, pts, src_pt=(0.0, 0.0, 0.0), comp=2, f0=1.0,
                     eps=1.0, mu=1.0):
    """Exact fields of a unit point electric dipole (e^{+iwt} phasors):
    conjugate of the standard e^{-iwt} formulae. ``comp`` is the dipole
    direction (0/1/2 = x/y/z). Returns (E, H), each (N, 3) complex.

    ``k`` is the wavenumber IN THE MEDIUM. Relative to the vacuum formulae
    at that k, a homogeneous (eps, mu) scales E by 1/eps (Jackson 9.18 with
    1/(4 pi eps)) and H by omega/k = 1/sqrt(eps mu) (the c k^2 prefactor is
    omega k), so H/E carries the medium admittance sqrt(eps/mu)."""
    p = np.zeros(3)
    p[comp] = f0
    rv = np.asarray(pts, dtype=float) - np.asarray(src_pt, dtype=float)
    r = np.linalg.norm(rv, axis=1)
    n = rv / r[:, None]
    pdotn = n @ p                                     # (N,)
    ph = np.exp(-1j * k * r)
    nxp = np.cross(n, np.broadcast_to(p, (len(r), 3)))
    E = (ph / (4 * np.pi))[:, None] * (
        (k * k / r)[:, None] * np.cross(nxp, n)
        + (3 * n * pdotn[:, None] - p) * (1.0 / r ** 3 + 1j * k / r ** 2)[:, None]
    )
    H = (k * k * ph / (4 * np.pi))[:, None] * nxp * (1.0 / r - 1j / (k * r ** 2))[:, None]
    if eps != 1.0 or mu != 1.0:
        E = E / eps
        H = H / math.sqrt(eps * mu)
    return E, H


def farfield_3d(faces, ks, obs_pts, eps=1.0, mu=1.0):
    """Far (E, H) 3-vectors at obs_pts, shapes (nfreq, nobs, 3).
    H = rhat x E / Z (local plane wave).

    ``ks`` is the VACUUM ``2 pi f``; the homogeneous exterior (eps, mu)
    enters as in core/near2far.py and meep's green3d
    (src/near2far.cpp:141-149): the wavenumber is ``k = omega sqrt(eps mu)``
    everywhere a phase or 1/R appears, the electric-current potential A
    radiates with ``omega mu = k Z`` (Z = sqrt(mu/eps)) while the magnetic
    one F radiates with ``k``, and the far H is E / Z. Vacuum is bitwise the
    old path (Z = 1.0 multiplies and divides exactly).

    This was vacuum-only: the same scrambling the 2D kernel showed (eight
    lobes on an isotropic source at n = 1.2) applies to every 3D far field
    computed in a non-unit background, because the box-scale phase error
    ``2 pi L (n - 1) / lambda`` is what breaks the coherent sum, not the
    overall scale.
    """
    ks = np.asarray(ks, dtype=float) * math.sqrt(eps * mu)
    Z = math.sqrt(mu / eps)
    obs = np.asarray(obs_pts, dtype=float)
    cen = np.mean([f.pts.mean(axis=0) for f in faces], axis=0)
    d = obs - cen
    R = np.linalg.norm(d, axis=1)
    rhat = d / R[:, None]
    E = np.zeros((len(ks), len(obs), 3), dtype=complex)
    for face in faces:
        nvec = np.asarray(face.normal, dtype=float)
        for fi, k in enumerate(ks):
            J = np.cross(np.broadcast_to(nvec, face.H[fi].shape), face.H[fi])
            M = -np.cross(np.broadcast_to(nvec, face.E[fi].shape), face.E[fi])
            # R and the phase compensation share one origin (the box
            # center) — mixing R from cen with absolute pts double-counts
            # e^{ik rhat.cen} for off-center sources (self-test pins this).
            ph = np.exp(1j * k * (rhat @ (face.pts - cen).T))  # (nobs, N)
            A = ph @ (J * (face.dA))
            F = ph @ (M * (face.dA))
            At = A - rhat * (A * rhat).sum(axis=1, keepdims=True)
            g = np.exp(-1j * k * R) / (4 * np.pi * R)
            E[fi] += -1j * k * g[:, None] * (Z * At - np.cross(rhat, F))
    H = np.cross(np.broadcast_to(rhat, E.shape), E) / Z
    return E, H


class Near2FarBox3D:
    """DFT accumulator for the six faces of the index box
    [i0, i1] x [j0, j1] x [k0, k1] (faces ON those integer planes).

    Per face, the two tangential E comps (sampled at (t+1) dt) and the two
    tangential H comps (normal-averaged onto the plane, t+1/2) are midpoint
    co-located onto the face cell-center lattice. Acc layout per face:
    (nfreq, 8, m0, m1) = dt-summed [Re,Im] x (E_b, E_c, H_b, H_c).
    """
    reduce = "sum"

    def __init__(self, dt, freqs, box, dx, dtype=torch.float64):
        self.dt = dt
        self.dx = dx
        self.freqs = torch.tensor(list(freqs), dtype=dtype)
        self.box = tuple(int(v) for v in box)         # i0, i1, j0, j1, k0, k1
        self._faces = [(a, i, s) for a in range(3)
                       for i, s in ((self.box[2 * a], -1.0),
                                    (self.box[2 * a + 1], +1.0))]

    def _slice_face(self, state, axis, idx):
        """Tangential (E_b, E_c, H_b, H_c) of face (axis, idx), co-located
        to the face cell centers."""
        b, c = _TANG[axis]
        Eb = getattr(state, _Comp3D[b])
        Ec = getattr(state, _Comp3D[c])
        Hb = getattr(state, _Comp3D[3 + b])
        Hc = getattr(state, _Comp3D[3 + c])

        def plane_E(F):
            sl = [slice(None)] * 3
            sl[axis] = idx
            return F[tuple(sl)]

        def plane_H(F):
            sl0 = [slice(None)] * 3
            sl1 = [slice(None)] * 3
            sl0[axis] = idx - 1
            sl1[axis] = idx
            return 0.5 * (F[tuple(sl0)] + F[tuple(sl1)])

        def avg(F, ax):
            sl0 = [slice(None)] * 2
            sl1 = [slice(None)] * 2
            sl0[ax] = slice(None, -1)
            sl1[ax] = slice(1, None)
            return 0.5 * (F[tuple(sl0)] + F[tuple(sl1)])

        # order of the transverse axes on the sliced planes
        t_ax = {v: n for n, v in enumerate((b, c))}
        q0 = {v: self.box[2 * v] for v in (b, c)}
        q1 = {v: self.box[2 * v + 1] for v in (b, c)}

        def trim(F, v, hi):
            s = [slice(None)] * 2
            s[t_ax[v]] = slice(q0[v], hi)
            return F[tuple(s)]

        # average axis carries int nodes -> trim q0..q1 (incl) before avg;
        # the other tangent is already on half nodes -> trim q0..q1 (excl).
        # E sits half-node on its own axis -> average along the OTHER tangent
        Eb2 = trim(avg(trim(plane_E(Eb), c, q1[c] + 1), t_ax[c]), b, q1[b])
        Ec2 = trim(avg(trim(plane_E(Ec), b, q1[b] + 1), t_ax[b]), c, q1[c])
        # H sits int-node on its own axis -> average along its OWN tangent
        Hb2 = trim(avg(trim(plane_H(Hb), b, q1[b] + 1), t_ax[b]), c, q1[c])
        Hc2 = trim(avg(trim(plane_H(Hc), c, q1[c] + 1), t_ax[c]), b, q1[b])
        return Eb2, Ec2, Hb2, Hc2

    def sample(self, state, t_idx: int) -> torch.Tensor:
        """Returns (nfaces, nfreq, 8, m0, m1); run_* sums over steps."""
        freqs = self.freqs
        dev = None
        outs = []
        for axis, idx, _s in self._faces:
            Eb, Ec, Hb, Hc = self._slice_face(state, axis, idx)
            dev = Eb.device
            w = 2 * math.pi * freqs.to(dev)
            tE = (t_idx + 1.0) * self.dt
            tH = (t_idx + 0.5) * self.dt
            chans = []
            for F, t in ((Eb, tE), (Ec, tE), (Hb, tH), (Hc, tH)):
                ph = (w * t).reshape((-1,) + (1,) * F.ndim)
                chans += [torch.cos(ph) * F, -torch.sin(ph) * F]
            outs.append(self.dt * torch.stack(chans, dim=1))  # (nfreq, 8, m0, m1)
        return torch.stack(outs, dim=0)               # (nfaces, nfreq, 8, ...)

    def faces(self, acc):
        """Build Face3D list from the accumulated (summed) samples."""
        if hasattr(acc, "detach"):
            acc = acc.detach().cpu().numpy()
        dx = self.dx
        n0 = [np.arange(0, self.box[2 * v + 1] - self.box[2 * v]) for v in range(3)]
        out = []
        for fi_face, (axis, idx, s) in enumerate(self._faces):
            b, c = _TANG[axis]
            lb, lc = np.meshgrid(n0[b], n0[c], indexing="ij")
            pts = np.zeros((lb.size, 3))
            pts[:, axis] = idx * dx
            pts[:, b] = (self.box[2 * b] + lb.ravel() + 0.5) * dx
            pts[:, c] = (self.box[2 * c] + lc.ravel() + 0.5) * dx
            normal = [0.0, 0.0, 0.0]
            normal[axis] = s
            a = acc[fi_face]                            # (nfreq, 8, m0, m1)
            nfreq = a.shape[0]
            E = np.zeros((nfreq, pts.shape[0], 3), dtype=complex)
            H = np.zeros_like(E)
            E[:, :, b] = (a[:, 0] + 1j * a[:, 1]).reshape(nfreq, -1)
            E[:, :, c] = (a[:, 2] + 1j * a[:, 3]).reshape(nfreq, -1)
            H[:, :, b] = (a[:, 4] + 1j * a[:, 5]).reshape(nfreq, -1)
            H[:, :, c] = (a[:, 6] + 1j * a[:, 7]).reshape(nfreq, -1)
            out.append(Face3D(pts=pts, normal=tuple(normal), dA=dx * dx,
                              E=E, H=H))
        return out


def analytic_box_faces(k, dA_pts, src_pt, comp=2, f0=1.0):
    """Synthetic closed box of Face3D sampled from ``dipole_fields_3d``
    (self-tests). ``dA_pts``: (half_size, n_per_axis) — faces at
    src_pt +- half_size, midpoint lattice n_per_axis per transverse axis."""
    h, n = dA_pts
    q = (np.arange(n) + 0.5) * (2 * h / n) - h
    dA = (2 * h / n) ** 2
    faces = []
    for axis in range(3):
        b, c = _TANG[axis]
        for s in (-1.0, 1.0):
            ub, uc = np.meshgrid(q, q, indexing="ij")
            pts = np.zeros((n * n, 3))
            pts[:, axis] = s * h
            pts[:, b] = ub.ravel()
            pts[:, c] = uc.ravel()
            pts = pts + np.asarray(src_pt, dtype=float)
            E, H = dipole_fields_3d(k, pts, src_pt=src_pt, comp=comp, f0=f0)
            normal = [0.0, 0.0, 0.0]
            normal[axis] = float(s)
            faces.append(Face3D(pts=pts, normal=tuple(normal), dA=dA,
                                E=E[None, ...], H=H[None, ...]))
    return faces
