# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp geometry objects -> meep-torch Polygon scene items (corner-origin).

Class structure mirrors meep/python/geom.py verbatim (names, constructor
signatures, inheritance: Wedge/Cone extend Cylinder, Ellipsoid extends Block);
scene_items() is the one private engine hook. Objects carry a Medium; resolving
to scene items splits the nondispersive/dispersive eps value from the Kerr chi3
(painted separately by rasterize_scene chi3_items). Painter's order is
preserved (later overrides). 3D-only inputs the 2D raster cannot express
(out-of-plane axes/bases, epsilon_func, sidewall_angle) raise NotImplementedError.
"""
import copy
import math
import warnings
from numbers import Number

import numpy as np

from ..geometry.polygons import Polygon
from .coords import GridSpec, rect_corners, to_corner
from .material import Medium
from .vec import Vector3


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _resolve(medium: Medium | float | None):
    if medium is None:
        return 1.0, 0.0
    if isinstance(medium, Medium):
        return medium.resolve()
    if callable(medium):
        raise NotImplementedError(
            "per-object material functions (material=callable(p) -> Medium) "
            "are not supported: the rasterizer fills polygons with uniform "
            "media (meep_torch/geometry/scene.py). Per-voxel sampling would "
            "cover an isotropic material function; the tutorial case "
            "(polarization_grating: a twisted-nematic tensor with "
            "epsilon_offdiag) additionally needs an anisotropic raster, which "
            "the scalar eps_tm/eps_te grids cannot hold")
    return float(medium), 0.0


def _ngon(radius: float, cx: float, cy: float, gs: GridSpec) -> np.ndarray:
    n = max(16, int(round(2 * math.pi * radius / gs.dx)))
    n = min(n, 256)
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)


def _inplane_basis(e1: Vector3, e2: Vector3, e3: Vector3):
    """Block e1/e2/e3 -> in-plane unit pair (u1, u2)."""
    if e1.z != 0.0 or e2.z != 0.0 or e3.x != 0.0 or e3.y != 0.0:
        raise NotImplementedError("Block basis must keep e1/e2 in the x-y plane (2D raster)")
    return e1.unit(), e2.unit()


class GeometricObject:
    """meep GeometricObject: material/center/label; shift() returns a copy.
    epsilon_func (pointwise eps callable) has no raster path -> raises."""

    def __init__(self, material=Medium(), center=Vector3(), epsilon_func=None,
                 label=None):
        if epsilon_func is not None:
            raise NotImplementedError("epsilon_func not supported (2D raster: use material)")
        self.label = label
        self.material = material
        self.center = Vector3(*center)

    def __add__(self, vec):
        return self.shift(Vector3(*vec))

    __radd__ = __add__

    def __iadd__(self, vec):
        self.center += Vector3(*vec)
        return self

    def shift(self, vec):
        c = copy.copy(self)
        c.center = self.center + Vector3(*vec)
        return c

    def info(self, indent_by=0):
        """meep GeometricObject.info: print the class and its properties,
        indented by ``indent_by`` spaces."""
        pad = " " * indent_by
        print(f"{pad}{type(self).__name__}")
        for k, v in vars(self).items():
            print(f"{pad}     {k} = {v}")


class Block(GeometricObject):
    """meep Block: brick spanned by basis e1/e2/e3 (in-plane rotation only)."""

    def __init__(self, size, e1=Vector3(1, 0, 0), e2=Vector3(0, 1, 0),
                 e3=Vector3(0, 0, 1), **kwargs):
        self.size = Vector3(*size)
        self.e1 = Vector3(*e1)
        self.e2 = Vector3(*e2)
        self.e3 = Vector3(*e3)
        super().__init__(**kwargs)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        c = self.center
        if dims == 1:
            # 1D: z is the propagation axis -> mapped to x; full y (thin grid).
            x0 = _clip(c.z - self.size.z / 2.0 + gs.sx / 2.0, 0.0, gs.sx)
            x1 = _clip(c.z + self.size.z / 2.0 + gs.sx / 2.0, 0.0, gs.sx)
            if math.isinf(self.size.z):
                x0, x1 = 0.0, gs.sx
            y0, y1 = 0.0, gs.sy
        else:
            u1, u2 = _inplane_basis(self.e1, self.e2, self.e3)
            if u1 != Vector3(1.0, 0.0, 0.0) or u2 != Vector3(0.0, 1.0, 0.0):
                if math.isinf(self.size.x) or math.isinf(self.size.y):
                    raise NotImplementedError("rotated Block with mp.inf size not supported")
                x, y = to_corner(c, gs)
                hx, hy = self.size.x / 2.0, self.size.y / 2.0
                v = np.array([[x + a * u1.x + b * u2.x, y + a * u1.y + b * u2.y]
                              for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])
                return [(Polygon(v), self.material)]
            x0, y0, x1, y1 = rect_corners(c, self.size, gs)
        v = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return [(Polygon(v), self.material)]


class Ellipsoid(Block):
    """meep Ellipsoid: inscribed in its Block (in-plane ellipse here)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        u1, u2 = _inplane_basis(self.e1, self.e2, self.e3)
        x, y = to_corner(self.center, gs)
        a, b = self.size.x / 2.0, self.size.y / 2.0
        n = min(256, max(16, int(round(2 * math.pi * max(a, b) / gs.dx))))
        ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        ca, sb = a * np.cos(ang), b * np.sin(ang)
        v = np.stack([x + ca * u1.x + sb * u2.x, y + ca * u1.y + sb * u2.y], axis=1)
        return [(Polygon(v), self.material)]


class Cylinder(GeometricObject):
    """meep Cylinder: circular cross-section; the 2D raster is the axis-normal
    slice, so the axis must stay along z."""

    def __init__(self, radius, axis=Vector3(0, 0, 1), height=1e20, n_pts=None,
                 **kwargs):
        # n_pts: meep-torch ext, N-gon resolution (default: by grid resolution)
        self.axis = Vector3(*axis)
        self.radius = float(radius)
        self.height = float(height)
        self.n_pts = n_pts
        super().__init__(**kwargs)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        if self.axis.x != 0.0 or self.axis.y != 0.0:
            raise NotImplementedError("Cylinder axis off z not supported (2D raster)")
        x, y = to_corner(self.center, gs)
        if self.n_pts is not None:
            n = max(3, int(self.n_pts))
            ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
            v = np.stack([x + self.radius * np.cos(ang), y + self.radius * np.sin(ang)], axis=1)
            return [(Polygon(v), self.material)]
        return [(Polygon(_ngon(self.radius, x, y, gs)), self.material)]


class Wedge(Cylinder):
    """meep Wedge: cylindrical wedge -> circular-sector polygon in 2D."""

    def __init__(self, radius, wedge_angle=2 * math.pi,
                 wedge_start=Vector3(1, 0, 0), **kwargs):
        self.wedge_angle = wedge_angle
        self.wedge_start = Vector3(*wedge_start)
        super().__init__(radius, **kwargs)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        x, y = to_corner(self.center, gs)
        if self.wedge_angle >= 2 * math.pi:
            return [(Polygon(_ngon(self.radius, x, y, gs)), self.material)]
        th0 = math.atan2(self.wedge_start.y, self.wedge_start.x)
        n = min(256, max(8, int(round(self.radius * self.wedge_angle / gs.dx))))
        ang = np.linspace(th0, th0 + self.wedge_angle, n)
        arc = np.stack([x + self.radius * np.cos(ang), y + self.radius * np.sin(ang)], axis=1)
        return [(Polygon(np.vstack([[x, y], arc])), self.material)]


class Cone(Cylinder):
    """meep Cone: truncated cone; radius at the base, radius2 at the tip the
    axis points to. In 2D the axis must have an x-y component -> trapezoid."""

    def __init__(self, radius, radius2=0, **kwargs):
        self.radius2 = radius2
        super().__init__(radius, **kwargs)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        ax = self.axis
        if ax.x == 0.0 and ax.y == 0.0:
            raise NotImplementedError("Cone axis needs an in-plane (x-y) component (2D raster)")
        u = Vector3(ax.x, ax.y).unit()               # in-plane axis (z part ignored)
        p = Vector3(-u.y, u.x)                       # in-plane perpendicular
        base = self.center - u * (self.height / 2.0)
        tip = self.center + u * (self.height / 2.0)
        pts = [base - p * self.radius, base + p * self.radius]
        pts += [tip] if self.radius2 == 0.0 else [tip + p * self.radius2,
                                                  tip - p * self.radius2]
        return [(Polygon(np.array([to_corner(q, gs) for q in pts])), self.material)]


class Sphere(GeometricObject):
    """meep Sphere: the x-y cross-section through the center is a circle."""

    def __init__(self, radius, **kwargs):
        self.radius = float(radius)
        super().__init__(**kwargs)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        x, y = to_corner(self.center, gs)
        return [(Polygon(_ngon(self.radius, x, y, gs)), self.material)]


class Prism(GeometricObject):
    """meep Prism: polygonal prism from world-coord vertices (z axis only).

    height keeps a default (required in meep) so 2D callers can omit it;
    a given center recenters the vertices at it, like meep."""

    def __init__(self, vertices, height=1e20, axis=Vector3(z=1), center=None,
                 sidewall_angle=0, **kwargs):
        if sidewall_angle != 0:
            raise NotImplementedError("Prism sidewall_angle not supported (2D raster)")
        ax = Vector3(*axis)
        if ax.x != 0.0 or ax.y != 0.0:
            raise NotImplementedError("Prism axis must be along z (2D raster)")
        vs = [Vector3(*v) for v in vertices]
        centroid = sum(vs, Vector3()) * (1.0 / len(vs))
        if center is not None:
            delta = Vector3(*center) - centroid
            vs = [v + delta for v in vs]
            centroid = Vector3(*center)
        self.vertices = vs
        self.height = float(height)
        self.axis = ax
        self.sidewall_angle = sidewall_angle
        super().__init__(center=centroid, **kwargs)

    def shift(self, vec):
        # the raster reads self.vertices, not center: shift both (meep does too)
        c = copy.copy(self)
        d = Vector3(*vec)
        c.vertices = [v + d for v in self.vertices]
        c.center = self.center + d
        return c

    def scene_items(self, gs: GridSpec, dims: int = 2):
        v = [to_corner(p, gs) for p in self.vertices]
        return [(Polygon(np.array(v)), self.material)]


def build_scene_items(geometry, gs: GridSpec, dims: int = 2, with_cond: bool = False):
    """Flatten geometry objects -> (eps_items, chi3_items) for rasterize_scene.

    with_cond additionally returns [(poly, (sigma_d, sigma_b, mu))] for
    media that carry W5 conductivity/permeability."""
    eps_items, chi3_items, cond_items = [], [], []
    for obj in geometry:
        for poly, medium in obj.scene_items(gs, dims):
            val, chi3 = _resolve(medium)
            eps_items.append((poly, val))
            if chi3 != 0.0:
                chi3_items.append((poly, chi3))
            sd = float(getattr(medium, "D_conductivity", 0.0) or 0.0)
            sb = float(getattr(medium, "B_conductivity", 0.0) or 0.0)
            mu = float(getattr(medium, "mu", 1.0) or 1.0)
            # EVERY item is painted, defaults included: a later air block must
            # overwrite an earlier conductor in painter order
            cond_items.append((poly, (sd, sb, mu)))
    if with_cond:
        has_cond = any(v != (0.0, 0.0, 1.0) for _, v in cond_items)
        return eps_items, chi3_items, cond_items if has_cond else []
    return eps_items, chi3_items


# ------------------------------------------------------------------ lattice math
# Matrix/Lattice/Mesh and the lattice<->cartesian<->reciprocal conversions,
# ported verbatim from meep/python/geom.py (same signatures and arithmetic).


class Mesh(GeometricObject):
    """meep Mesh: triangulated 3D object (closed manifold) built from vertex
    positions and 0-based triangle index tuples; center recenters the centroid.
    Mesh.from_file(filename, ...) loads via the optional trimesh package."""

    def __init__(self, vertices, triangles, center=None, **kwargs):
        self.vertices = [Vector3(*v) for v in vertices]
        self.triangles = [tuple(t) for t in triangles]

        centroid = sum(self.vertices, Vector3(0)) * (1.0 / len(self.vertices))
        if center is not None:
            center = Vector3(*center)
            shift = center - centroid
            self.vertices = [v + shift for v in self.vertices]
        else:
            center = centroid

        super().__init__(center=center, **kwargs)

    @classmethod
    def from_file(cls, filename, material=None, center=None, scale=1.0):
        try:
            import trimesh
        except ImportError as e:
            raise ImportError(
                "Mesh.from_file requires the 'trimesh' package. "
                "Install it with: pip install trimesh"
            ) from e

        loaded = trimesh.load(filename, process=True)
        if isinstance(loaded, trimesh.Scene):
            nparts = len(loaded.geometry)
            loaded = loaded.dump(concatenate=True)
            if nparts > 1:
                print(f"{filename}: file had {nparts} parts, merged into one")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"{filename}: file does not contain a triangular mesh")

        if scale != 1.0:
            loaded.apply_scale(scale)

        if not loaded.is_watertight or not loaded.is_winding_consistent:
            warnings.warn(
                f"{filename}: mesh is not a watertight, consistently-oriented "
                "manifold; subpixel smoothing may be inaccurate"
            )

        kwargs = {}
        if material is not None:
            kwargs["material"] = material
        return cls(
            vertices=loaded.vertices,
            triangles=loaded.faces,
            center=center,
            **kwargs,
        )


class Matrix:
    """meep Matrix: 3x3 matrix with c1/c2/c3 columns (hermitian construction
    from diag+offdiag when no columns given); supports m1*m2, m*v, s*m, +, -,
    /, transpose, getH/H, conj, determinant, inverse."""

    def __init__(
        self,
        c1=Vector3(),
        c2=Vector3(),
        c3=Vector3(),
        diag=Vector3(),
        offdiag=Vector3(),
    ):
        self.c1 = Vector3(*c1)
        self.c2 = Vector3(*c2)
        self.c3 = Vector3(*c3)
        if np.all(c1 == c2) and np.all(c2 == c3) and np.all(c3 == Vector3()):
            self.c1 = Vector3(diag[0], offdiag[0], offdiag[1])
            self.c2 = Vector3(np.conj(offdiag[0]), diag[1], offdiag[2])
            self.c3 = Vector3(np.conj(offdiag[1]), np.conj(offdiag[2]), diag[2])

    def __getitem__(self, i):
        return self.row(i)

    def __mul__(self, m):
        if type(m) is Matrix:
            return self.mm_mult(m)
        elif type(m) is Vector3:
            return self.mv_mult(m)
        elif isinstance(m, Number):
            return self.scale(m)
        else:
            raise TypeError(f"No operation known for 'Matrix * {type(m)}'")

    def __rmul__(self, left_arg):
        if isinstance(left_arg, Number):
            return self.scale(left_arg)
        else:
            raise TypeError(f"No operation known for 'Matrix * {type(left_arg)}'")

    def __truediv__(self, scalar):
        return Matrix(self.c1 / scalar, self.c2 / scalar, self.c3 / scalar)

    def __add__(self, m):
        return Matrix(self.c1 + m.c1, self.c2 + m.c2, self.c3 + m.c3)

    def __sub__(self, m):
        return Matrix(self.c1 - m.c1, self.c2 - m.c2, self.c3 - m.c3)

    def __repr__(self):
        r0 = self.row(0)
        r1 = self.row(1)
        r2 = self.row(2)
        return (f"<<{r0[0]} {r0[1]} {r0[2]}>\n"
                f" <{r1[0]} {r1[1]} {r1[2]}>\n"
                f" <{r2[0]} {r2[1]} {r2[2]}>>")

    def __array__(self, dtype=None, copy=None):
        a = np.array([[v.x, v.y, v.z]
                      for v in (self.row(0), self.row(1), self.row(2))])
        return a if dtype is None else a.astype(dtype)

    def row(self, i):
        return Vector3(self.c1[i], self.c2[i], self.c3[i])

    def mm_mult(self, m):
        c1 = Vector3(
            self.row(0).dot(m.c1), self.row(1).dot(m.c1), self.row(2).dot(m.c1)
        )
        c2 = Vector3(
            self.row(0).dot(m.c2), self.row(1).dot(m.c2), self.row(2).dot(m.c2)
        )
        c3 = Vector3(
            self.row(0).dot(m.c3), self.row(1).dot(m.c3), self.row(2).dot(m.c3)
        )
        return Matrix(c1, c2, c3)

    def mv_mult(self, v):
        return Vector3(*[self.row(i).dot(Vector3(*v)) for i in range(3)])

    def scale(self, s):
        return Matrix(self.c1.scale(s), self.c2.scale(s), self.c3.scale(s))

    def determinant(self):
        sum1 = (self[0][0] * self[1][1] * self[2][2]
                + self[0][1] * self[1][2] * self[2][0]
                + self[1][0] * self[2][1] * self[0][2])
        sum2 = (self[0][2] * self[1][1] * self[2][0]
                + self[0][1] * self[1][0] * self[2][2]
                + self[1][2] * self[2][1] * self[0][0])
        return sum1 - sum2

    def conj(self):
        return Matrix(self.c1.conj(), self.c2.conj(), self.c3.conj())

    def transpose(self):
        return Matrix(self.row(0), self.row(1), self.row(2))

    def getH(self):
        return self.transpose().conj()

    def inverse(self):
        v1 = Vector3(self[1][1] * self[2][2] - self[1][2] * self[2][1],
                     self[1][2] * self[2][0] - self[1][0] * self[2][2],
                     self[1][0] * self[2][1] - self[1][1] * self[2][0])
        v2 = Vector3(self[2][1] * self[0][2] - self[0][1] * self[2][2],
                     self[0][0] * self[2][2] - self[0][2] * self[2][0],
                     self[0][1] * self[2][0] - self[0][0] * self[2][1])
        v3 = Vector3(self[0][1] * self[1][2] - self[1][1] * self[0][2],
                     self[1][0] * self[0][2] - self[0][0] * self[1][2],
                     self[1][1] * self[0][0] - self[1][0] * self[0][1])
        m = Matrix(v1, v2, v3)
        return m.scale(1 / self.determinant())

    H = property(getH, None)


class Lattice:
    """meep Lattice: cell size plus basis1/2/3 (stored normalized) scaled by
    basis_size; basis is the column matrix of b1/b2/b3, metric is B^T B."""

    def __init__(
        self,
        size=Vector3(1, 1, 1),
        basis_size=Vector3(1, 1, 1),
        basis1=Vector3(1, 0, 0),
        basis2=Vector3(0, 1, 0),
        basis3=Vector3(0, 0, 1),
    ):
        self.size = Vector3(*size)
        self.basis_size = Vector3(*basis_size)
        self.basis1 = Vector3(*basis1)
        self.basis2 = Vector3(*basis2)
        self.basis3 = Vector3(*basis3)

    @property
    def basis1(self):
        return self._basis1

    @basis1.setter
    def basis1(self, val):
        self._basis1 = val.unit()

    @property
    def basis2(self):
        return self._basis2

    @basis2.setter
    def basis2(self, val):
        self._basis2 = val.unit()

    @property
    def basis3(self):
        return self._basis3

    @basis3.setter
    def basis3(self, val):
        self._basis3 = val.unit()

    @property
    def b1(self):
        return self.basis1.scale(self.basis_size.x)

    @property
    def b2(self):
        return self.basis2.scale(self.basis_size.y)

    @property
    def b3(self):
        return self.basis3.scale(self.basis_size.z)

    @property
    def basis(self):
        B = Matrix(self.b1, self.b2, self.b3)
        if B.determinant() == 0:
            raise ValueError("Lattice basis vectors must be linearly independent.")
        return B

    @property
    def metric(self):
        B = self.basis
        return B.transpose() * B


def lattice_to_cartesian(x, lat):
    if isinstance(x, Vector3):
        return lat.basis * x
    return (lat.basis * x) * lat.basis.inverse()


def cartesian_to_lattice(x, lat):
    if isinstance(x, Vector3):
        return lat.basis.inverse() * x
    return (lat.basis.inverse() * x) * lat.basis


def reciprocal_to_cartesian(x, lat):
    s = Vector3(*[1 if v == 0 else v for v in lat.size])
    m = Matrix(Vector3(s.x), Vector3(y=s.y), Vector3(z=s.z))
    Rst = (lat.basis * m).transpose()
    return Rst.inverse() * x if isinstance(x, Vector3) else (Rst.inverse() * x) * Rst


def cartesian_to_reciprocal(x, lat):
    s = Vector3(*[1 if v == 0 else v for v in lat.size])
    m = Matrix(Vector3(s.x), Vector3(y=s.y), Vector3(z=s.z))
    Rst = (lat.basis * m).transpose()
    return Rst * x if isinstance(x, Vector3) else (Rst * x) * Rst.inverse()


def lattice_to_reciprocal(x, lat):
    return cartesian_to_reciprocal(lattice_to_cartesian(x, lat), lat)


def reciprocal_to_lattice(x, lat):
    return cartesian_to_lattice(reciprocal_to_cartesian(x, lat), lat)


def get_rotation_matrix(axis, theta):
    """Unitary rotation matrix: get_rotation_matrix(axis, theta) * v equals
    v.rotate(axis, theta) (theta in radians, right-hand around axis)."""
    return Matrix(
        Vector3(x=1).rotate(axis, theta),
        Vector3(y=1).rotate(axis, theta),
        Vector3(z=1).rotate(axis, theta),
    )
