# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W4 T1-T2: 2D near2far Green kernels — analytic line-source self-test.

Near fields on a synthetic box contour are taken from the exact outgoing
cylindrical wave of a line current; farfield_* must reproduce that same wave
at far observation points (surface equivalence). No FDTD involved.
"""
import math

import numpy as np

from meep_torch.core.near2far import (Face, farfield_te, farfield_tm,
                                      line_source_fields_te,
                                      line_source_fields_tm)

FREQ = 1.0
K = 2 * math.pi * FREQ


def _box_faces(cx, cy, half, n_pts, fields_fn):
    """Four outward-normal faces of a box centered at (cx, cy)."""
    s = np.linspace(-half, half, n_pts, endpoint=False) + half / n_pts
    dl = 2 * half / n_pts
    faces = []
    for normal, pts in (
        ((+1, 0), np.stack([np.full(n_pts, cx + half), cy + s], axis=1)),
        ((-1, 0), np.stack([np.full(n_pts, cx - half), cy + s], axis=1)),
        ((0, +1), np.stack([cx + s, np.full(n_pts, cy + half)], axis=1)),
        ((0, -1), np.stack([cx + s, np.full(n_pts, cy - half)], axis=1)),
    ):
        Ez, Hx, Hy = fields_fn(pts)
        # tangential in-plane field per face: Hy on x-normal, Hx on y-normal
        Ht = Hy if normal[0] else Hx
        faces.append(Face(pts=pts, normal=normal, dl=dl,
                          E=Ez[None, :], H=Ht[None, :]))
    return faces


def _check_far(faces, farfield_fn, fields_fn, atol_rel=1e-3):
    th = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    for r in (500.0, 2000.0):
        obs = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        E, H = farfield_fn(faces, [K], obs)
        Ez_ref, _, _ = fields_fn(obs)
        got = E[0, :, 2]                      # z component, single freq
        err = np.abs(got - Ez_ref).max() / np.abs(Ez_ref).max()
        assert err < atol_rel, f"r={r}: rel err {err:.2e}"


def test_tm_line_source_reproduced():
    src = lambda pts: line_source_fields_tm(K, pts, 0.0, 0.0)
    faces = _box_faces(0.0, 0.0, 0.5, 40, src)
    _check_far(faces, farfield_tm, src)


def test_tm_contour_translation_invariance():
    # same source, off-center contour: far pattern must not change
    src = lambda pts: line_source_fields_tm(K, pts, 0.0, 0.0)
    faces = _box_faces(0.3, -0.2, 0.9, 60, src)
    _check_far(faces, farfield_tm, src)


def test_tm_farfield_H_plane_wave():
    # far-zone H = R_hat x E (eta0=1): radial Poynting ~ |Ez|^2
    src = lambda pts: line_source_fields_tm(K, pts, 0.0, 0.0)
    faces = _box_faces(0.0, 0.0, 0.5, 40, src)
    th = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    obs = np.stack([1000.0 * np.cos(th), 1000.0 * np.sin(th)], axis=1)
    E, H = farfield_tm(faces, [K], obs)
    Sx = np.real(np.conj(E[0, :, 1]) * H[0, :, 2] - np.conj(E[0, :, 2]) * H[0, :, 1])
    Sy = np.real(np.conj(E[0, :, 2]) * H[0, :, 0] - np.conj(E[0, :, 0]) * H[0, :, 2])
    Sr = Sx * np.cos(th) + Sy * np.sin(th)
    assert (Sr > 0).all()                     # outward power everywhere
    assert Sr.std() / Sr.mean() < 1e-3        # isotropic line source


def test_te_line_source_reproduced():
    src = lambda pts: line_source_fields_te(K, pts, 0.0, 0.0)

    def _faces():
        s = np.linspace(-0.5, 0.5, 40, endpoint=False) + 0.5 / 40
        dl = 1.0 / 40
        out = []
        for normal, pts in (
            ((+1, 0), np.stack([np.full(40, 0.5), s], axis=1)),
            ((-1, 0), np.stack([np.full(40, -0.5), s], axis=1)),
            ((0, +1), np.stack([s, np.full(40, 0.5)], axis=1)),
            ((0, -1), np.stack([s, np.full(40, -0.5)], axis=1)),
        ):
            Hz, Ex, Ey = src(pts)
            Et = Ey if normal[0] else Ex
            out.append(Face(pts=pts, normal=normal, dl=dl,
                            E=Hz[None, :], H=Et[None, :]))
        return out

    th = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    obs = np.stack([500.0 * np.cos(th), 500.0 * np.sin(th)], axis=1)
    E, H = farfield_te(_faces(), [K], obs)
    Hz_ref, _, _ = src(obs)
    err = np.abs(H[0, :, 2] - Hz_ref).max() / np.abs(Hz_ref).max()
    assert err < 1e-3, f"rel err {err:.2e}"


def test_periodic_tiling_grating_orders():
    # W4.5: tile a line-source contour N times at pitch Ly = 2 (lambda = 1):
    # far lobes must sit at the grating orders sin(theta) = m / 2
    from meep_torch.core.near2far import tile_faces
    src = lambda pts: line_source_fields_tm(K, pts, 0.0, 0.0)
    faces = tile_faces(_box_faces(0.0, 0.0, 0.5, 40, src), 8, 2.0, ky=0.0)
    th = np.linspace(-math.pi / 2, math.pi / 2, 721)
    obs = np.stack([2000.0 * np.cos(th), 2000.0 * np.sin(th)], axis=1)
    E, _ = farfield_tm(faces, [K], obs)
    p = np.abs(E[0, :, 2]) ** 2
    peaks = [th[i] for i in range(1, len(th) - 1)
             if p[i] > p[i - 1] and p[i] > p[i + 1] and p[i] > 0.2 * p.max()]
    want = [math.asin(m / 2.0) for m in (-1, 0, 1)]
    for w in want:
        assert min(abs(pk - w) for pk in peaks) < math.radians(1.5), \
            f"no lobe at {math.degrees(w):.1f} deg; peaks {[round(math.degrees(x),1) for x in peaks]}"
