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

import torch

from .eigensolver import dual_vectors
from ..geometry.smoothing import blend_te_supersampled
from ..mp.geometry import Block, Cylinder, Sphere
from ..mp.material import Medium
from ..mp.vec import Vector3

__all__ = ["lattice_matrix", "frac_min_image", "cell_centres",
           "object_mask_2d", "rasterize_2d", "eps_diag_of",
           "RasterMixin"]


def eps_diag_of(material):
    """``(exx, eyy, ezz)`` of a Medium, for the 2D mpb rasterizer.

    Reads ``epsilon_diag`` directly instead of going through
    ``Medium.resolve()``. resolve() is the mp FDTD shim's entry point and
    rejects an anisotropic diagonal outright ("the core engine takes an
    in-plane tensor directly"), which is correct there and wrong here:
    MPB's own tutorial builds a complete 2D gap out of exactly such a
    material (rods epsilon_diag=(1,1,12) in a background of (12,12,1)).

    An isotropic Medium returns three equal values, so the caller needs no
    special case and the scalar path is unchanged bit for bit.
    """
    off = getattr(material, "epsilon_offdiag", None)
    if off is not None and (float(off.x) or float(off.y) or float(off.z)):
        raise NotImplementedError(
            f"epsilon_offdiag {off} is not supported: the 2D operators are "
            f"built from a DIAGONAL epsilon (ezz for TM, (exx, eyy) for "
            f"TE). An off-diagonal term couples the two polarizations and "
            f"needs the full vector operator")
    if getattr(material, "E_susceptibilities", None):
        raise NotImplementedError(
            "dispersive materials are not supported by ModeSolver: the "
            "eigenproblem is linear in omega^2 and a frequency-dependent "
            "epsilon makes it nonlinear")
    if getattr(material, "material", None) is not None:
        raise NotImplementedError(
            "material functions are not supported by ModeSolver")
    d = material.epsilon_diag
    return float(d.x), float(d.y), float(d.z)


def _eps_of(material):
    val, _ = material.resolve()
    if not isinstance(val, float):
        raise NotImplementedError(
            "dispersive/tensor materials not supported by ModeSolver")
    return val


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


def object_mask_2d(obj, X, Y, a1, a2, centre_shift, basis=None):
    """Boolean mask of sample points inside ``obj`` (min-image periodic).

    ``basis`` is the pair of UNIT lattice basis vectors. MPB reads a
    geometric object's ``center`` in LATTICE coordinates -- ``center`` is
    measured along basis1 and basis2, not along x and y -- so it has to be
    mapped before it can be compared with cartesian sample points.

    Two things make this easy to get wrong, and it was:

      - for an identity basis the two readings coincide, so every square
        and rectangular fixture in the repo agrees either way;
      - it scales with the BASIS only, not with ``Lattice.size``. Measured
        against meep.mpb: on ``size=(2,2)`` with an identity basis,
        ``center=(0.25, 0)`` lands at cartesian (0.25, 0), not (0.5, 0);
        on the triangular basis ``center=(1/6, 1/6)`` lands at
        (1/6)(b1 + b2) = (0.2887, 0).

    ``basis=None`` keeps the old cartesian reading, for callers with no
    lattice to hand.
    """
    c = obj.center - centre_shift
    cx, cy = float(c.x), float(c.y)
    if basis is not None:
        b1, b2 = basis
        cx, cy = (cx * b1[0] + cy * b2[0], cx * b1[1] + cy * b2[1])
    dx, dy = frac_min_image(a1, a2, (X - cx, Y - cy))
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
                 default_material, supersample: int = 4, basis=None):
    """Supersampled coverage epsilon on the (possibly oblique) cell.

    LAST-listed object wins, which is meep's and MPB's documented rule.
    Returns ``(eps3, blocks3)``, each carrying the three DIAGONAL
    components on axis 0: ``eps3`` is the coverage average per component,
    ``blocks3`` the raw supersampled array a caller can feed to the Kottke
    TE blend. An isotropic structure gives three identical components, so
    the scalar case is the general one with nothing special about it.
    """
    X, Y = cell_centres(a1, a2, nu, nv, supersample)
    fine = np.empty((3,) + X.shape)
    if callable(default_material) and not hasattr(default_material, "resolve"):
        it = np.nditer(X, flags=["multi_index"])
        for _ in it:
            m = it.multi_index
            fine[(slice(None),) + m] = eps_diag_of(
                default_material(Vector3(X[m], Y[m], 0.0)))
    else:
        fine[:] = np.asarray(eps_diag_of(default_material))[:, None, None]

    for obj in geometry:
        mask = object_mask_2d(obj, X, Y, a1, a2, geometry_center, basis)
        vals = eps_diag_of(obj.material)
        for c in range(3):
            fine[c][mask] = vals[c]

    blocks3 = fine.reshape(3, nu, supersample, nv, supersample)
    return blocks3.mean(axis=(2, 4)), blocks3


class RasterMixin:
    """Epsilon rasterization on the ModeSolver, split out for file size.

    Moved verbatim from ``mpb/solver.py``: same methods on the same
    object. The 1D/3D branch still rasterizes per cartesian axis and keeps
    its own orthogonality guard; 2D goes through the lattice-coordinate
    path in this module.
    """

    def _pw_eps_te(self):
        """The TE epsilon pair the planewave operator should use.

        Kottke's blended diagonal is a FINITE-DIFFERENCE smoothing: it
        belongs on the staggered edges of a Yee-like grid, not in a
        spectral operator whose basis already resolves the interface. So
        the planewave backend gets the plain per-component coverage
        averages -- None (scalar eps) for an isotropic structure, the
        (exx, eyy) pair for an anisotropic one.
        """
        if getattr(self, "_isotropic", True):
            return None
        return (self._eps_diag[0], self._eps_diag[1])

    def _rasterize_epsilon(self):
        """Supersampled coverage eps over the unit cell (min-image periodic),
        rank = number of periodic axes.

        LAST-listed object wins, which is meep's and MPB's documented rule:
        "When objects overlap, later objects in the list take precedence"
        (mpb and meep Python_User_Interface.md; MPB's fields.c says the same
        of its object list, "just like the ordinary geometry list").

        This used to iterate in reverse, giving the FIRST object precedence,
        and every single-object fixture was blind to it. The tutorial is
        not: it builds a point defect by appending an air rod over a
        lattice duplicate, and tunes it by appending a second rod on top --
        both of which the old order silently discarded, so the 5x5 defect
        never formed (band 25 came out at 0.279 instead of 0.376, energy in
        the defect cylinder 0.12 instead of ~0.7) and the tuning root-find
        got the same frequency for eps=1 and eps=12.
        """
        axes = self._active_axes()
        if not axes:
            raise ValueError("geometry_lattice.size has no nonzero component")
        gs = self._adjust_grid_size()
        if len(axes) == 2:
            return self._rasterize_lattice_2d(gs)
        # 1D/3D still rasterize on a cartesian box, which silently solves a
        # sheared structure on an oblique basis. 2D goes through the
        # lattice-coordinate path above and needs no such guard.
        self._check_basis_orthogonal()
        size = self.geometry_lattice.size
        n = [int(gs[i]) for i in axes]
        L = [float(size[i]) for i in axes]
        d = [L[a] / n[a] for a in range(len(axes))]
        if max(d) - min(d) > 1e-9 * max(d):
            raise NotImplementedError(
                "anisotropic grid cells (resolution*size mismatch across "
                "axes) not supported by the Bloch-FDTD backend")
        ss = self.supersample
        coords = [(-L[a] / 2 + (np.arange(n[a] * ss) + 0.5) * d[a] / ss)
                  for a in range(len(axes))]
        grids = np.meshgrid(*coords, indexing="ij")
        # world point per fine cell: active axes filled, collapsed axes at 0
        P = [grids[axes.index(i)] if i in axes else np.zeros(grids[0].shape)
             for i in range(3)]

        if callable(self.default_material) and not isinstance(
                self.default_material, Medium):
            fine = np.empty(grids[0].shape)
            it = np.nditer(fine, flags=["multi_index"])
            for _ in it:
                m = it.multi_index
                med = self.default_material(
                    Vector3(P[0][m], P[1][m], P[2][m]))
                fine[m] = _eps_of(med)
        else:
            fine = np.full(grids[0].shape, _eps_of(self.default_material))

        for obj in self.geometry:
            c = obj.center - self.geometry_center
            dp = []
            for a, i in enumerate(axes):
                v = grids[a] - c[i]
                dp.append(v - L[a] * np.round(v / L[a]))  # min-image wrap
            if isinstance(obj, (Cylinder, Sphere)):
                if isinstance(obj, Cylinder) and len(axes) == 3:
                    raise NotImplementedError(
                        "3D Cylinder rasterization not supported")
                mask = sum(v ** 2 for v in dp) < obj.radius ** 2
            elif isinstance(obj, Block):
                if (obj.e1 != Vector3(1, 0, 0) or obj.e2 != Vector3(0, 1, 0)
                        or obj.e3 != Vector3(0, 0, 1)):
                    raise NotImplementedError(
                        "rotated Block not supported in ModeSolver")
                mask = np.ones(grids[0].shape, dtype=bool)
                for a, i in enumerate(axes):
                    half = obj.size[i] / 2
                    if not math.isinf(half):
                        mask &= np.abs(dp[a]) < half
            else:
                raise NotImplementedError(
                    f"{type(obj).__name__} rasterization not supported "
                    "in ModeSolver (Cylinder/Sphere/Block only)")
            fine[mask] = _eps_of(obj.material)

        shape = []
        for a in range(len(axes)):
            shape += [n[a], ss]
        blocks = fine.reshape(shape)
        eps = blocks.mean(axis=tuple(range(1, 2 * len(axes), 2)))
        self._eps_te = (blend_te_supersampled(blocks)
                        if len(axes) == 2 else None)
        return torch.as_tensor(eps, dtype=torch.float64), d[0]

    def _basis_vectors(self):
        """The UNIT lattice basis vectors b1, b2 (no size scaling).

        Object centres are measured along these; the CELL vectors a_i =
        b_i * size_i are what the grid spans. Conflating the two is the
        difference between a rod at (0.25, 0) and one at (0.5, 0) on a
        size-2 cell.
        """
        lat = self.geometry_lattice
        return ((float(lat.basis1.x), float(lat.basis1.y)),
                (float(lat.basis2.x), float(lat.basis2.y)))

    def _lattice_vectors(self):
        """Cartesian a1, a2 of the 2D cell (rows of get_lattice)."""
        lat = self.get_lattice()
        return (float(lat[0][0]), float(lat[0][1])), \
               (float(lat[1][0]), float(lat[1][1]))

    def _rasterize_lattice_2d(self, gs):
        """2D epsilon on the lattice cell, oblique or not (mpb/raster.py).

        The grid is uniform in fractional coordinates, so a triangular cell
        is sampled on its own rhombus. A rectangular basis reduces to the
        cartesian per-axis form the 1D/3D path uses, bitwise, which is what
        leaves the existing goldens unmoved.
        """
        from .raster import rasterize_2d
        a1, a2 = self._lattice_vectors()
        nu, nv = int(gs[0]), int(gs[1])
        du = math.hypot(*a1) / nu
        dv = math.hypot(*a2) / nv
        if abs(du - dv) > 1e-9 * max(du, dv):
            raise NotImplementedError(
                f"anisotropic grid cells (|a1|/nu = {du:.6g} vs |a2|/nv = "
                f"{dv:.6g}): resolution*size differs across the lattice "
                f"axes, which the Bloch-FDTD backend does not support")
        eps3, blocks3 = rasterize_2d(a1, a2, nu, nv, self.geometry,
                                     self.geometry_center,
                                     self.default_material,
                                     supersample=self.supersample,
                                     basis=self._basis_vectors())
        exx, eyy, ezz = eps3
        b1, b2 = dual_vectors(a1, a2)
        orthogonal = abs(float(b1 @ b2)) < 1e-12
        isotropic = (np.allclose(exx, ezz, rtol=0, atol=1e-12)
                     and np.allclose(eyy, ezz, rtol=0, atol=1e-12))
        # TE takes the in-plane pair, TM the axial component. Three ways to
        # get the pair, in decreasing accuracy and increasing generality:
        if isotropic and orthogonal:
            # Kottke subpixel smoothing. It blends an interface normal
            # against the CARTESIAN axes and assumes a scalar contrast, so
            # it needs both conditions -- and it is what the rectangular
            # goldens were measured with, so this branch must stay first.
            self._eps_te = blend_te_supersampled(blocks3[2])
        elif isotropic:
            # oblique + isotropic: scalar eps, ~1st order on a staircase
            self._eps_te = None
        else:
            # anisotropic: per-component coverage average. Kottke has no
            # meaning here -- there is no single contrast to blend across
            # the interface when the two sides differ per component.
            self._eps_te = (exx, eyy)
        self._a1, self._a2 = a1, a2
        self._eps_diag = eps3
        self._isotropic = isotropic
        return torch.as_tensor(ezz, dtype=torch.float64), du


