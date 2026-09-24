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


def _medium_faces(n_index, cx, cy, half, n_pts):
    """Box faces carrying the exact TM line-source field of a MEDIUM.

    The wave number is `k = n omega`, and with `mu = 1` the impedance is
    `Z = 1/n`, so `H = n E` against the vacuum formula at the same k.
    """
    k_med = K * n_index

    def fields(pts):
        Ez, Hx, Hy = line_source_fields_tm(k_med, pts, 0.0, 0.0)
        return Ez, n_index * Hx, n_index * Hy

    return _box_faces(cx, cy, half, n_pts, fields)


def test_tm_far_field_is_independent_of_box_size():
    """The same source, enclosed by DIFFERENT boxes, must radiate the same.

    Surface equivalence says the far field depends on the source, not on where
    the equivalence contour was drawn. Translation is already gated above;
    this pins the other freedom, the box SIZE, which is what actually
    interrogates how the four faces sum: a wrong normal, quadrature weight or
    relative phase scales with the contour and so moves when it grows.

    Run in a MEDIUM (n = 1.2) because that is where it bites. Every far field
    this repo produced in a non-unit background used the vacuum wavenumber and
    was wrong in SHAPE -- see wiki/bugs/near2far-was-vacuum-only.md -- and the
    error is an interference pattern at the BOX scale, so it is precisely a
    box-size sweep that exposes it.

    Measured margin, so nobody loosens the bound without knowing what it
    costs. Boxes at half-width 0.5, 0.9 and 1.4:

        correct kernel   0.0002   0.0002
        vacuum kernel    0.8863   1.5520

    Four orders of magnitude. This gate on its own would have caught the
    vacuum-kernel defect with no meep, no tutorial and no FDTD run at all.
    """
    n_index = 1.2
    th = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    obs = np.stack([800.0 * np.cos(th), 800.0 * np.sin(th)], axis=1)

    ref = None
    for half, n_pts in ((0.5, 80), (0.9, 144), (1.4, 224)):
        faces = _medium_faces(n_index, 0.0, 0.0, half, n_pts)
        E, _ = farfield_tm(faces, [K], obs, eps=n_index ** 2)
        got = E[0, :, 2]
        if ref is None:
            ref = got
            continue
        err = np.abs(got - ref).max() / np.abs(ref).max()
        assert err < 5e-3, f"half={half}: far field moved with the box, {err:.2e}"


def test_the_box_size_gate_fails_on_the_vacuum_kernel():
    """The gate above must BITE, so pin the defect it was written for.

    Running the same medium faces through the vacuum kernel -- which is what
    shipped until c5058bd -- has to move the answer between box sizes. Without
    this, a regression that reverted the medium wavenumber would leave the
    test above passing on whatever single box it happened to use.
    """
    n_index = 1.2
    th = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    obs = np.stack([800.0 * np.cos(th), 800.0 * np.sin(th)], axis=1)

    out = []
    for half, n_pts in ((0.5, 80), (1.4, 224)):
        faces = _medium_faces(n_index, 0.0, 0.0, half, n_pts)
        E, _ = farfield_tm(faces, [K], obs)          # eps=1: the old behaviour
        out.append(E[0, :, 2])
    err = np.abs(out[1] - out[0]).max() / np.abs(out[0]).max()
    assert err > 0.05, (
        f"the vacuum kernel should NOT be box-size independent in a medium, "
        f"but moved only {err:.2e} -- this gate has stopped discriminating")
