# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Epsilon rasterization on a 2D lattice cell, oblique or rectangular.

The grid is uniform in FRACTIONAL coordinates (u, v) over
``[-1/2, 1/2)^2``, so a triangular cell is sampled on its own rhombus
rather than on a cartesian box that would clip it. World points are
``r = u a1 + v a2``; object tests happen in cartesian, because a circle is
a circle regardless of the basis; and the periodic min-image wrap happens
back in FRACTIONAL coordinates, because "the nearest image" means nearest
across the lattice, which is not the same as nearest along x and y once
the basis is oblique.

Sample positions are cell CENTRES, ``((i+0.5)/nu - 1/2) a1 + ...``. That
is this engine's convention throughout and differs from MPB's node
sampling by half a cell -- see wiki/tasks/port_mpb-grid-node-convention.md
for the measurement and why it is kept.

For a rectangular basis every expression here reduces to the per-axis
cartesian form the 1D/3D path still uses, which is what keeps the
existing goldens unmoved.
"""
from __future__ import annotations

import math

import numpy as np

from ..mp.geometry import Block, Cylinder, Sphere

__all__ = ["lattice_matrix", "frac_min_image", "cell_centres",
           "object_mask_2d", "rasterize_2d"]


def lattice_matrix(a1, a2) -> np.ndarray:
    """Columns are the lattice vectors, so ``A @ (u, v)`` is cartesian."""
    return np.array([[a1[0], a2[0]], [a1[1], a2[1]]], dtype=float)


def frac_min_image(a1, a2, dxy):
    """Wrap cartesian offsets to the nearest lattice image.

    ``dxy`` is ``(dx, dy)`` arrays of cartesian offsets. Converts to
    fractional, wraps each component to [-1/2, 1/2), converts back. On a
    rectangular cell this is exactly ``v - L*round(v/L)`` per axis; on an
    oblique one it is not, and using the per-axis form there would call
    the wrong copy of a rod the nearest one.
    """
    A = lattice_matrix(a1, a2)
    Ainv = np.linalg.inv(A)
    du = Ainv[0, 0] * dxy[0] + Ainv[0, 1] * dxy[1]
    dv = Ainv[1, 0] * dxy[0] + Ainv[1, 1] * dxy[1]
    du -= np.round(du)
    dv -= np.round(dv)
    return A[0, 0] * du + A[0, 1] * dv, A[1, 0] * du + A[1, 1] * dv


def cell_centres(a1, a2, nu: int, nv: int, supersample: int = 1):
    """Cartesian (X, Y) of the cell-centre sample points.

    With ``supersample > 1`` the same cell is sampled ``supersample`` times
    finer along each axis, for coverage averaging.
    """
    n_u, n_v = nu * supersample, nv * supersample
    u = (np.arange(n_u) + 0.5) / n_u - 0.5
    v = (np.arange(n_v) + 0.5) / n_v - 0.5
    U, V = np.meshgrid(u, v, indexing="ij")
    A = lattice_matrix(a1, a2)
    return A[0, 0] * U + A[0, 1] * V, A[1, 0] * U + A[1, 1] * V


def object_mask_2d(obj, X, Y, a1, a2, centre_shift):
    """Boolean mask of sample points inside ``obj`` (min-image periodic)."""
    c = obj.center - centre_shift
    dx, dy = frac_min_image(a1, a2, (X - float(c.x), Y - float(c.y)))
    if isinstance(obj, (Cylinder, Sphere)):
        return dx ** 2 + dy ** 2 < obj.radius ** 2
    if isinstance(obj, Block):
        from ..mp.vec import Vector3
        if (obj.e1 != Vector3(1, 0, 0) or obj.e2 != Vector3(0, 1, 0)
                or obj.e3 != Vector3(0, 0, 1)):
            raise NotImplementedError(
                "rotated Block not supported in ModeSolver")
        mask = np.ones(X.shape, dtype=bool)
        for d, half in ((dx, obj.size[0] / 2), (dy, obj.size[1] / 2)):
            if not math.isinf(half):
                mask &= np.abs(d) < half
        return mask
    raise NotImplementedError(
        f"{type(obj).__name__} rasterization not supported in ModeSolver "
        "(Cylinder/Sphere/Block only)")


def rasterize_2d(a1, a2, nu: int, nv: int, geometry, geometry_center,
                 default_material, eps_of, supersample: int = 4):
    """Supersampled coverage epsilon on the (possibly oblique) cell.

    LAST-listed object wins, which is meep's and MPB's documented rule.
    Returns ``(eps, blocks)``: the coverage average, and the raw
    supersampled block array a caller can feed to the Kottke TE blend.
    """
    X, Y = cell_centres(a1, a2, nu, nv, supersample)
    if callable(default_material) and not hasattr(default_material, "resolve"):
        from ..mp.vec import Vector3
        fine = np.empty(X.shape)
        it = np.nditer(fine, flags=["multi_index"])
        for _ in it:
            m = it.multi_index
            fine[m] = eps_of(default_material(Vector3(X[m], Y[m], 0.0)))
    else:
        fine = np.full(X.shape, eps_of(default_material))

    for obj in geometry:
        fine[object_mask_2d(obj, X, Y, a1, a2, geometry_center)] = \
            eps_of(obj.material)

    blocks = fine.reshape(nu, supersample, nv, supersample)
    return blocks.mean(axis=(1, 3)), blocks
