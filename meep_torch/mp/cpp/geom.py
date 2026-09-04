# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""C++ SWIG-surface parity (math/geometry group): mp.vec, mp.ivec, mp.volume,
mp.grid_volume, mp.symmetry, mp.material_function, ...

Genuine ports of the coordinate/box/symmetry math in meep/src/meep/vec.hpp and
vec.cpp. MPI/chunk-decomposition machinery (splits, ghost interpolation) and
the libctl geometry-tree material lookup raise NotImplementedError with a
pointer to the meep-torch replacement (whole-domain tensors / rasterizer).
"""
import cmath
import math

import numpy as np

from ..vec import Vector3

# enums from meep/src/meep/vec.hpp
D1, D2, D3, Dcyl = 0, 1, 2, 3
X, Y, Z, R, P, NO_DIRECTION = 0, 1, 2, 3, 4, 5
High, Low = 0, 1
NO_COMPONENT = 22
_SX, _ENERGY = 100, 105  # first derived component / first energy density
_E_STUFF, _H_STUFF = 0, 1

_DIRS = ((Z,), (X, Y), (X, Y, Z), (Z, R))  # LOOP_OVER_DIRECTIONS per ndim

_CHUNK_MSG = ("chunk/MPI machinery is not ported: meep-torch uses whole-domain "
              "GPU tensors (see meep_torch.core.grid)")


def _dirs(dim):
    return _DIRS[dim]


def _my_round(x):
    return int(math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1))


def _yucky_dir(dim, n):
    if dim == Dcyl:
        return (P, R, Z)[n]
    if dim == D2:
        return (n + 2) % 3
    return n


def _is_derived(c):
    return c >= _SX


def _component_direction(c):
    if _is_derived(c):
        i = c - _SX
        return (X, Y, R, P, Z)[i] if i < 5 else NO_DIRECTION
    if c >= 20:  # Dielectric, Permeability, NO_COMPONENT
        return NO_DIRECTION
    return (X, Y, R, P, Z)[c % 5]


def _direction_component(c, d):
    off = {X: 0, Y: 1, R: 2, P: 3, Z: 4}
    if _is_derived(c):
        if c >= _ENERGY:
            return c
        return _SX + off[d]
    if d == NO_DIRECTION and _component_direction(c) == d:
        return c
    return (c // 5) * 5 + off[d]


def _component_index(c):
    return (0, 1, 0, 1, 2)[c % 5] if c < 20 else -1


class vec:
    """meep::vec — dim-aware coordinate: vec(z) is 1D, vec(x,y) 2D, vec(x,y,z) 3D."""
    __slots__ = ("dim", "t")

    def __init__(self, *args):
        self.t = [0.0] * 5
        if len(args) == 1:
            self.dim = D1
            self.t[Z] = float(args[0])
        elif len(args) == 2:
            self.dim = D2
            self.t[X], self.t[Y] = float(args[0]), float(args[1])
        elif len(args) == 3:
            self.dim = D3
            self.t[X], self.t[Y], self.t[Z] = (float(a) for a in args)
        elif not args:
            self.dim = D3  # C++ leaves dim unset; D3 zeros is the useful default
        else:
            raise TypeError("vec takes 0-3 coordinates")

    @classmethod
    def _make(cls, dim, t=None):
        v = cls.__new__(cls)
        v.dim = dim
        v.t = list(t) if t is not None else [0.0] * 5
        return v

    def r(self):
        return self.t[R]

    def x(self):
        return self.t[X]

    def y(self):
        return self.t[Y]

    def z(self):
        return self.t[Z]

    def in_direction(self, d):
        return self.t[d]

    def set_direction(self, d, val):
        self.t[d] = val

    def __add__(self, a):
        r = vec._make(a.dim, a.t)
        for d in _dirs(self.dim):
            r.t[d] += self.t[d]
        return r

    def __sub__(self, a):
        r = vec._make(a.dim, a.t)
        for d in _dirs(self.dim):
            r.t[d] = self.t[d] - r.t[d]
        return r

    def __neg__(self):
        r = vec._make(self.dim)
        for d in _dirs(self.dim):
            r.t[d] = -self.t[d]
        return r

    def __mul__(self, s):
        r = vec._make(self.dim, self.t)
        for d in _dirs(self.dim):
            r.t[d] *= s
        return r

    __rmul__ = __mul__

    def __truediv__(self, s):
        return self * (1.0 / s)

    def __eq__(self, a):
        if not isinstance(a, vec):
            return NotImplemented
        return all(self.t[d] == a.t[d] for d in _dirs(self.dim))

    def __ne__(self, a):
        eq = self.__eq__(a)
        return NotImplemented if eq is NotImplemented else not eq

    def round_float(self):
        return vec._make(self.dim, [float(np.float32(v)) for v in self.t])

    def project_to_boundary(self, d, boundary_loc):
        return abs(boundary_loc - self.t[d])

    def to_vector3(self):
        if self.dim == Dcyl:
            return Vector3(self.t[R], 0.0, self.t[Z])
        return Vector3(self.t[X], self.t[Y], self.t[Z])

    @classmethod
    def from_vector3(cls, v3, dim=D3):
        v = cls._make(dim)
        if dim == Dcyl:
            v.t[R], v.t[Z] = v3.x, v3.z
        else:
            v.t[X], v.t[Y], v.t[Z] = v3.x, v3.y, v3.z
        return v

    def __repr__(self):
        return f"vec<dim={self.dim}>({', '.join(str(self.t[d]) for d in _dirs(self.dim))})"


class ivec:
    """meep::ivec — integer (Yee half-step) coordinate; same overloads as vec."""
    __slots__ = ("dim", "t")

    def __init__(self, *args):
        self.t = [0] * 5
        if len(args) == 1:
            self.dim = D1
            self.t[Z] = int(args[0])
        elif len(args) == 2:
            self.dim = D2
            self.t[X], self.t[Y] = int(args[0]), int(args[1])
        elif len(args) == 3:
            self.dim = D3
            self.t[X], self.t[Y], self.t[Z] = (int(a) for a in args)
        elif not args:
            self.dim = D2  # matches the C++ default ctor
        else:
            raise TypeError("ivec takes 0-3 coordinates")

    @classmethod
    def _make(cls, dim, t=None):
        v = cls.__new__(cls)
        v.dim = dim
        v.t = list(t) if t is not None else [0] * 5
        return v

    def r(self):
        return self.t[R]

    def x(self):
        return self.t[X]

    def y(self):
        return self.t[Y]

    def z(self):
        return self.t[Z]

    def in_direction(self, d):
        return self.t[d]

    def set_direction(self, d, val):
        self.t[d] = val

    def yucky_val(self, n):
        d = _yucky_dir(self.dim, n)
        return self.t[d] if d in _dirs(self.dim) else 0

    def __add__(self, a):
        if isinstance(a, int):
            r = ivec._make(self.dim, self.t)
            for d in _dirs(self.dim):
                r.t[d] += a
            return r
        r = ivec._make(a.dim, a.t)
        for d in _dirs(self.dim):
            r.t[d] += self.t[d]
        return r

    def __sub__(self, a):
        r = ivec._make(a.dim, a.t)
        for d in _dirs(self.dim):
            r.t[d] = self.t[d] - r.t[d]
        return r

    def __neg__(self):
        r = ivec._make(self.dim)
        for d in _dirs(self.dim):
            r.t[d] = -self.t[d]
        return r

    def __mul__(self, s):
        if isinstance(s, ivec):  # element-wise
            r = ivec._make(self.dim, self.t)
            for d in _dirs(self.dim):
                r.t[d] *= s.t[d]
            return r
        if isinstance(s, float):  # ivec * double -> vec (grid coords)
            r = vec._make(self.dim)
            for d in _dirs(self.dim):
                r.t[d] = self.t[d] * s
            return r
        r = ivec._make(self.dim, self.t)
        for d in _dirs(self.dim):
            r.t[d] *= s
        return r

    def __truediv__(self, s):
        r = ivec._make(self.dim, self.t)
        for d in _dirs(self.dim):
            r.t[d] = int(r.t[d] / s)  # C++ integer division truncates
        return r

    def _cmp(self, a, op):
        return all(op(self.t[d], a.t[d]) for d in _dirs(self.dim))

    def __eq__(self, a):
        if not isinstance(a, ivec):
            return NotImplemented
        return self._cmp(a, lambda p, q: p == q)

    def __ne__(self, a):
        eq = self.__eq__(a)
        return NotImplemented if eq is NotImplemented else not eq

    def __le__(self, a):
        return self._cmp(a, lambda p, q: p <= q)

    def __ge__(self, a):
        return self._cmp(a, lambda p, q: p >= q)

    def __lt__(self, a):
        return self._cmp(a, lambda p, q: p < q)

    def __gt__(self, a):
        return self._cmp(a, lambda p, q: p > q)

    def round_down_to_even(self):
        r = ivec._make(self.dim)
        for d in _dirs(self.dim):
            r.t[d] = self.t[d] - abs(self.t[d]) % 2
        return r

    def to_vector3(self):
        if self.dim == Dcyl:
            return Vector3(self.t[R], 0, self.t[Z])
        return Vector3(self.t[X], self.t[Y], self.t[Z])

    def __repr__(self):
        return f"ivec<dim={self.dim}>({', '.join(str(self.t[d]) for d in _dirs(self.dim))})"


def _vec_minmax(a, b, pick):
    r = vec._make(a.dim)
    r.t = [pick(p, q) for p, q in zip(a.t, b.t)]
    return r


def zero_vec(di):
    return vec._make(di)


def one_vec(di):
    v = vec._make(di)
    for d in _dirs(di):
        v.t[d] = 1.0
    return v


def unit_vec(di, d):
    v = zero_vec(di)
    v.t[d] = 1.0
    return v


def clean_vec(pt, val_unused=0.0):
    v = vec._make(pt.dim, [val_unused] * 5)
    for d in _dirs(pt.dim):
        v.t[d] = pt.t[d]
    return v


def veccyl(rr, zz):
    v = vec._make(Dcyl)
    v.t[R], v.t[Z] = rr, zz
    return v


def zero_ivec(di):
    return ivec._make(di)


def one_ivec(di):
    v = ivec._make(di)
    for d in _dirs(di):
        v.t[d] = 1
    return v


def unit_ivec(di, d):
    v = zero_ivec(di)
    v.t[d] = 1
    return v


def iveccyl(rr, zz):
    v = ivec._make(Dcyl)
    v.t[R], v.t[Z] = rr, zz
    return v


class volume:
    """meep::volume — axis-aligned box between two vecs (corners auto-sorted)."""

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], int):
            self.dim = args[0]
            self.min_corner = vec._make(self.dim)
            self.max_corner = vec._make(self.dim)
        elif len(args) == 1:
            pt = args[0]
            self.dim = pt.dim
            self.min_corner = vec._make(pt.dim, pt.t)
            self.max_corner = vec._make(pt.dim, pt.t)
        elif len(args) == 2:
            v1, v2 = args
            self.dim = v1.dim
            self.min_corner = _vec_minmax(v1, v2, min)
            self.max_corner = _vec_minmax(v1, v2, max)
        else:
            raise TypeError("volume(dim) | volume(pt) | volume(vec1, vec2)")

    def set_direction_min(self, d, val):
        self.min_corner.t[d] = val

    def set_direction_max(self, d, val):
        self.max_corner.t[d] = val

    def in_direction_min(self, d):
        return self.min_corner.t[d]

    def in_direction_max(self, d):
        return self.max_corner.t[d]

    def in_direction(self, d):
        return self.max_corner.t[d] - self.min_corner.t[d]

    def get_min_corner(self):
        return vec._make(self.dim, self.min_corner.t)

    def get_max_corner(self):
        return vec._make(self.dim, self.max_corner.t)

    def center(self):
        return (self.min_corner + self.max_corner) * 0.5

    def computational_volume(self):
        v = 1.0
        for d in _dirs(self.dim):
            v *= self.in_direction(d)
        return v

    def integral_volume(self):
        v = 1.0
        for d in _dirs(self.dim):
            if self.in_direction(d) != 0.0:
                v *= self.in_direction(d)
        if self.dim == Dcyl:
            v *= math.pi * (self.in_direction_max(R) + self.in_direction_min(R))
        return v

    def full_volume(self):
        v = self.computational_volume()
        if self.dim == Dcyl:
            v *= math.pi * (self.in_direction_max(R) + self.in_direction_min(R))
        return v

    def diameter(self):
        return max((self.in_direction(d) for d in _dirs(self.dim)), default=0.0)

    def contains(self, a):
        if isinstance(a, volume):
            return self.contains(a.min_corner) and self.contains(a.max_corner)
        return all(self.min_corner.t[d] <= a.t[d] <= self.max_corner.t[d]
                   for d in _dirs(self.dim))

    def intersect_with(self, a):
        if a.dim != self.dim:
            raise RuntimeError("Can't intersect volumes of dissimilar dimensions.")
        result = volume(self.dim)
        for d in _dirs(self.dim):
            lo = max(self.in_direction_min(d), a.in_direction_min(d))
            hi = min(self.in_direction_max(d), a.in_direction_max(d))
            if lo > hi:
                return volume(zero_vec(self.dim), zero_vec(self.dim))
            result.set_direction_min(d, lo)
            result.set_direction_max(d, hi)
        return result

    def intersects(self, a):
        if a.dim != self.dim:
            raise RuntimeError("Can't intersect volumes of dissimilar dimensions.")
        return all(max(self.in_direction_min(d), a.in_direction_min(d))
                   <= min(self.in_direction_max(d), a.in_direction_max(d))
                   for d in _dirs(self.dim))

    def normal_direction(self):
        ext = {d: self.in_direction(d) for d in _dirs(self.dim)}
        if self.dim == D1:
            return Z
        zero = [d for d, e in ext.items() if e == 0]
        nonzero = [d for d, e in ext.items() if e > 0]
        if len(zero) == 1 and len(nonzero) == len(ext) - 1:
            return zero[0]
        return NO_DIRECTION

    def round_float(self):
        return volume(self.min_corner.round_float(), self.max_corner.round_float())

    def __eq__(self, a):
        if not isinstance(a, volume):
            return NotImplemented
        return self.min_corner == a.min_corner and self.max_corner == a.max_corner

    def __ne__(self, a):
        eq = self.__eq__(a)
        return NotImplemented if eq is NotImplemented else not eq

    def __repr__(self):
        return f"volume({self.min_corner!r}, {self.max_corner!r})"


class grid_volume:
    """meep::grid_volume — a uniform Yee grid over a box (resolution a).

    Construct via vol1d/vol2d/vol3d/volcyl; chunk-split machinery raises."""

    def __init__(self):
        self.dim = D3
        self.a = 1.0
        self.inva = 1.0
        self._num = [0, 0, 0]
        self._stride = [0] * 5
        self._ntot = 1
        self.io = zero_ivec(self.dim)
        self.origin = zero_vec(self.dim)
        self._num_changed()

    @classmethod
    def _create(cls, dim, a, na, nb, nc):
        gv = cls()
        gv.dim = dim
        gv.a = a
        gv.inva = 1.0 / a
        gv._num = [na, nb, nc]
        gv.io = zero_ivec(dim)
        gv.origin = zero_vec(dim)
        gv._num_changed()
        gv.set_origin(zero_vec(dim))
        return gv

    def _clone(self):
        gv = grid_volume()
        gv.dim, gv.a, gv.inva = self.dim, self.a, self.inva
        gv._num = list(self._num)
        gv.io = ivec._make(self.io.dim, self.io.t)
        gv.origin = vec._make(self.origin.dim, self.origin.t)
        gv._num_changed()
        return gv

    def _num_changed(self):
        self._ntot = 1
        for d in _dirs(self.dim):
            self._ntot *= self._num[d % 3] + 1
        self._stride = [0] * 5
        for d in _dirs(self.dim):
            if d == Z:
                self._stride[d] = 1
            elif d == R:
                self._stride[d] = self.nz() + 1
            elif d == X:
                self._stride[d] = (self.nz() + 1) * (self.ny() + 1)
            elif d == Y:
                self._stride[d] = self.nz() + 1

    # --- sizes / strides
    def stride(self, d):
        return self._stride[d]

    def num_direction(self, d):
        return self._num[d % 3]

    def set_num_direction(self, d, value):
        self._num[d % 3] = value
        self._num_changed()

    def yucky_num(self, n):
        d = _yucky_dir(self.dim, n)
        return self.num_direction(d) if d in _dirs(self.dim) else 1

    def yucky_direction(self, n):
        return _yucky_dir(self.dim, n)

    def nr(self):
        return self.num_direction(R)

    def nx(self):
        return self.num_direction(X)

    def ny(self):
        return self.num_direction(Y)

    def nz(self):
        return self.num_direction(Z)

    def ntot(self):
        return self._ntot

    def nowned_min(self):
        n = 1
        for d in _dirs(self.dim):
            n *= self.num_direction(d)
        return n

    def nowned(self, c):
        pt = self.big_corner() - self.little_owned_corner(c)
        n = 1
        for d in _dirs(self.dim):
            n *= pt.t[d] // 2 + 1
        return n

    # --- coordinates
    def __getitem__(self, p):
        return p * (0.5 * self.inva)

    def round_vec(self, p):
        r = ivec._make(self.dim)
        for d in _dirs(self.dim):
            r.t[d] = _my_round(p.t[d] * 2 * self.a)
        return r

    def set_origin(self, *args):
        if len(args) == 2:  # (direction, int)
            d, o = args
            self.io.set_direction(d, o)
        elif isinstance(args[0], ivec):
            self.io = ivec._make(args[0].dim, args[0].t)
        else:
            self.io = self.round_vec(args[0])
        self.origin = self[self.io]

    def shift_origin(self, *args):
        if len(args) == 2:  # (direction, int)
            d, s = args
            self.set_origin(self.io + unit_ivec(self.dim, d) * s)
        elif isinstance(args[0], ivec):
            self.set_origin(self.io + args[0])
        else:
            self.set_origin(self.origin + args[0])

    def center_origin(self):
        self.shift_origin(-self.icenter())

    def get_origin(self):
        return vec._make(self.origin.dim, self.origin.t)

    def origin_in_direction(self, d):
        return self.origin.t[d]

    def iorigin_in_direction(self, d):
        return self.io.t[d]

    def origin_r(self):
        return self.origin.r()

    def origin_x(self):
        return self.origin.x()

    def origin_y(self):
        return self.origin.y()

    def origin_z(self):
        return self.origin.z()

    def little_corner(self):
        return ivec._make(self.io.dim, self.io.t)

    def big_corner(self):
        n = zero_ivec(self.dim)
        for d in _dirs(self.dim):
            n.t[d] = self.num_direction(d)
        return self.io + n * 2

    def corner(self, b):
        if b == Low:
            return self.get_origin()
        tmp = self.get_origin()
        for d in _dirs(self.dim):
            tmp.t[d] += self.num_direction(d) * self.inva
        return tmp

    def center(self):
        return self[self.icenter()]

    def icenter(self):
        n = zero_ivec(self.dim)
        for d in _dirs(self.dim):
            n.t[d] = self.num_direction(d)
        if self.dim == Dcyl:
            n.t[R] = 0
        return self.io + n.round_down_to_even()

    def contains(self, p):
        if isinstance(p, ivec):
            o = p - self.io
            return all(0 <= o.t[d] < (self.num_direction(d) + 1) * 2
                       for d in _dirs(self.dim))
        o = p - self.origin
        return all(-self.inva <= o.t[d] <= self.num_direction(d) * self.inva + self.inva
                   for d in _dirs(self.dim))

    def owns(self, p):
        o = p - self.io
        if self.dim == Dcyl:
            if self.origin.r() == 0.0 and 0 < o.z() <= self.nz() * 2 and o.r() == 0:
                return True
            return (0 < o.r() <= self.nr() * 2 and 0 < o.z() <= self.nz() * 2)
        return all(0 < o.t[d] <= self.num_direction(d) * 2 for d in _dirs(self.dim))

    # --- Yee lattice
    def iyee_shift(self, c):
        out = zero_ivec(self.dim)
        cd = _component_direction(c)
        for d in _dirs(self.dim):
            if (c in (20, 21)  # Dielectric, Permeability
                    or (c < 5 or 10 <= c < 15) and d == cd        # E or D
                    or (5 <= c < 10 or 15 <= c < 20) and d != cd):  # H or B
                out.t[d] = 1
        return out

    def yee_shift(self, c):
        return self[self.iyee_shift(c)]

    def yee_index(self, c):
        sh = self.iyee_shift(c)
        return sum((1 - sh.t[d]) * self.stride(d) for d in _dirs(self.dim))

    def eps_component(self):
        return {D1: 6, D2: 9, D3: 20, Dcyl: 8}[self.dim]  # Hy/Hz/Dielectric/Hp

    def yee2cent_offsets(self, c):
        offset1 = offset2 = 0
        sh = self.iyee_shift(c)
        for d in _dirs(self.dim):
            if not sh.t[d]:
                if offset2:
                    raise RuntimeError(f"weird yee shift for component {c}")
                if offset1:
                    offset2 = self.stride(d)
                else:
                    offset1 = self.stride(d)
        return offset1, offset2

    def cent2yee_offsets(self, c):
        o1, o2 = self.yee2cent_offsets(c)
        return -o1, -o2

    def has_field(self, c):
        if self.dim == D1:
            return c in (0, 6, 10, 16)  # Ex, Hy, Dx, By
        cd = _component_direction(c)
        return cd > Y if self.dim == Dcyl else cd < R

    def has_boundary(self, b, d):
        if self.dim == Dcyl:
            return d == Z or (d == R and (b == High or self.get_origin().r() > 0))
        return d in _dirs(self.dim)

    def index(self, c, p):
        offset = p - self.io - self.iyee_shift(c)
        return sum(offset.t[d] // 2 * self.stride(d) for d in _dirs(self.dim))

    def iloc(self, c, ind):
        out = zero_ivec(self.dim)
        for d in _dirs(self.dim):
            ios = int(ind / self.stride(d))
            while ios < 0:
                ios += self.num_direction(d) + 1
            out.t[d] = 2 * (ios % (self.num_direction(d) + 1))
        return out + self.iyee_shift(c) + self.io

    def loc(self, c, ind):
        return self[self.iloc(c, ind)]

    def loc_at_resolution(self, index, res):
        where = self.get_origin()
        for d in (X, Y, Z, R):
            if self.has_boundary(High, d):
                dist = self.boundary_location(High, d) - self.boundary_location(Low, d)
                nhere = max(1, int(math.floor(dist * res + 0.5)))
                where.t[d] = self.origin.t[d] + ((index % nhere) + 0.5) / res
                index //= nhere
        return where

    def ntot_at_resolution(self, res):
        mytot = 1
        for d in (X, Y, Z, R):
            if self.has_boundary(High, d):
                dist = self.boundary_location(High, d) - self.boundary_location(Low, d)
                mytot *= max(1, int(dist * res + 0.5))
        return mytot

    def boundary_location(self, b, d):
        Ex_, Ez_, Ep_ = 0, 4, 8
        ind = self.ntot() - 1 if b == High else 0
        if d in (X, Y):
            return self.loc(Ez_, ind).t[d]
        if d == R:
            return self.loc(Ep_, ind).r()
        if d == Z:
            return self.loc(Ep_ if self.dim == Dcyl else Ex_, ind).z()
        raise RuntimeError("no boundary in this direction")

    def little_owned_corner0(self, c):
        return self.little_corner() + one_ivec(self.dim) * 2 - self.iyee_shift(c)

    def little_owned_corner(self, c):
        il = self.little_owned_corner0(c)
        if self.dim == Dcyl and self.origin.r() == 0.0 and il.r() == 2:
            il.set_direction(R, 0)
        return il

    def big_owned_corner(self, c):
        return self.big_corner() - self.iyee_shift(c)

    # --- extents and measures
    def _min(self, d):
        return self.origin.t[d] + 0.25 * self.inva

    def _max(self, d):
        return self.origin.t[d] + self.num_direction(d) * self.inva + 0.25 * self.inva

    def xmin(self):
        return self._min(X)

    def xmax(self):
        return self._max(X)

    def ymin(self):
        return self._min(Y)

    def ymax(self):
        return self._max(Y)

    def zmin(self):
        return self._min(Z)

    def zmax(self):
        return self._max(Z)

    def rmin(self):
        if self.dim != Dcyl:
            raise RuntimeError("No rmin in these dimensions.")
        return 0.0 if self.origin.r() == 0.0 else self._min(R)

    def rmax(self):
        if self.dim != Dcyl:
            raise RuntimeError("No rmax in these dimensions.")
        return self._max(R)

    def dr(self):
        if self.dim != Dcyl:
            raise RuntimeError("Error in dr")
        return veccyl(self.inva, 0.0)

    def dx(self):
        if self.dim == D3:
            return vec(self.inva, 0, 0)
        if self.dim == D2:
            return vec(self.inva, 0)
        raise RuntimeError("Error in dx.")

    def dy(self):
        if self.dim == D3:
            return vec(0, self.inva, 0)
        if self.dim == D2:
            return vec(0, self.inva)
        raise RuntimeError("Error in dy.")

    def dz(self):
        if self.dim == Dcyl:
            return veccyl(0.0, self.inva)
        if self.dim == D3:
            return vec(0, 0, self.inva)
        if self.dim == D1:
            return vec(self.inva)
        raise RuntimeError("dz doesn't exist in 2D")

    def dV(self, *args):
        if isinstance(args[0], ivec):
            here = args[0]
            diameter = args[1] if len(args) > 1 else 1.0
            hinva = 0.5 * self.inva * diameter
            h = self[here]
            out = volume(self.dim)
            for d in _dirs(self.dim):
                out.set_direction_min(d, h.t[d] - hinva)
                out.set_direction_max(d, h.t[d] + hinva)
            if self.dim == Dcyl and here.r() == 0:
                out.set_direction_min(R, 0.0)
            return out
        c, ind = args
        if not self.owns(self.iloc(c, ind)):
            return volume(self.dim)
        return self.dV(self.iloc(c, ind))

    def surroundings(self):
        return volume(self[self.little_corner()], self[self.big_corner()])

    def interior(self):
        return volume(self[self.little_corner()],
                      self[self.big_corner() - one_ivec(self.dim) * 2])

    def surface_area(self):
        if self.dim == Dcyl:
            return 2 * (self.nr() + self.nz())
        if self.dim == D1:
            return 2
        if self.dim == D2:
            return 2 * (self.nx() + self.ny())
        return 2 * (self.nx() * self.ny() + self.nx() * self.nz()
                    + self.ny() * self.nz())

    def pad_self(self, d):
        self._num[d % 3] += 2  # one grid point on each side
        self._num_changed()
        self.shift_origin(d, -2)

    def pad(self, d=None):
        gv = self._clone()
        if d is None:
            for dd in _dirs(self.dim):
                gv.pad_self(dd)
        else:
            gv.pad_self(d)
        return gv

    def halve(self, d):
        gv = self._clone()
        gv.set_num_direction(
            d, 1 + (self.big_corner().t[d] - self.icenter().t[d]) // 2)
        gv.set_origin(d, self.icenter().t[d] - 2)
        return gv

    def print_grid_volume(self):
        for d in _dirs(self.dim):
            o = self.origin.t[d]
            print(f"dir {d} = {o:5g} - {o + self.num_direction(d) / self.a:5g} "
                  f"({self.num_direction(d) / self.a:5g})", end="\t")
        print()

    # --- chunk/MPI machinery: replaced by whole-domain tensors
    def subvolume(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def init_subvolume(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def intersect_with(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def interpolate(self, *a, **k):
        raise NotImplementedError(
            "Yee interpolation is not ported; sample fields via meep_torch.core.grid")

    def get_boundary_icorners(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def get_cost(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def get_split_costs(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def split_at_fraction(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def tile_split(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)

    def find_best_split(self, *a, **k):
        raise NotImplementedError(_CHUNK_MSG)


def volone(zsize, a):
    return grid_volume._create(D1, a, 0, 0, int(zsize * a + 0.5))


def vol1d(zsize, a):
    return volone(zsize, a)


def voltwo(xsize, ysize, a):
    return grid_volume._create(D2, a, 1 if xsize == 0 else int(xsize * a + 0.5),
                               1 if ysize == 0 else int(ysize * a + 0.5), 0)


def vol2d(xsize, ysize, a):
    return voltwo(xsize, ysize, a)


def vol3d(xsize, ysize, zsize, a):
    return grid_volume._create(D3, a, 1 if xsize == 0 else int(xsize * a + 0.5),
                               1 if ysize == 0 else int(ysize * a + 0.5),
                               1 if zsize == 0 else int(zsize * a + 0.5))


def volcyl(rsize, zsize, a):
    return grid_volume._create(Dcyl, a, int(rsize * a + 0.5), 0,
                               1 if zsize == 0.0 else int(zsize * a + 0.5))


class signed_direction:
    """meep::signed_direction — direction with mirror flip and complex phase."""

    def __init__(self, *args):
        self.d = args[0] if len(args) > 0 else X
        self.flipped = args[1] if len(args) > 1 else False
        self.phase = args[2] if len(args) > 2 else 1.0

    def __mul__(self, ph):
        return signed_direction(self.d, self.flipped, self.phase * ph)

    def __eq__(self, sd):
        if not isinstance(sd, signed_direction):
            return NotImplemented
        return (self.d == sd.d and self.flipped == sd.flipped
                and self.phase == sd.phase)

    def __ne__(self, sd):
        eq = self.__eq__(sd)
        return NotImplemented if eq is NotImplemented else not eq

    def __repr__(self):
        return f"signed_direction(d={self.d}, flipped={self.flipped}, phase={self.phase})"


def _flip(sd):
    return signed_direction(sd.d, not sd.flipped, sd.phase)


class symmetry:
    """meep::symmetry — mirror/rotation transforms of the Yee lattice.

    Build with identity()/mirror()/rotate2()/rotate4()/r_to_minus_r_symmetry()."""

    def __init__(self, *args):
        self.g = 1
        self.ph = 1.0
        self.S = [signed_direction(d) for d in range(5)]
        self.symmetry_point = zero_vec(D3)
        self.i_symmetry_point = zero_ivec(D3)
        self.next = None
        if args:
            self.symmetry_assign(args[0])

    def symmetry_assign(self, s):
        """SWIG name for operator=: copy s into self."""
        self.g = s.g
        self.ph = s.ph
        self.S = [signed_direction(sd.d, sd.flipped, sd.phase) for sd in s.S]
        self.symmetry_point = vec._make(s.symmetry_point.dim, s.symmetry_point.t)
        self.i_symmetry_point = ivec._make(s.i_symmetry_point.dim,
                                           s.i_symmetry_point.t)
        self.next = symmetry(s.next) if s.next is not None else None
        return self

    def multiplicity(self, x=None):
        m = self.g * self.next.multiplicity() if self.next else self.g
        if x is None:
            return m
        for n in range(1, m):
            if self.transform(x, n) == x:
                return n
        return m

    def __add__(self, b):
        if self.multiplicity() == 1:
            return b
        if b.multiplicity() == 1:
            return self
        s = symmetry(self)
        sn = s
        while sn.next:
            sn = sn.next
        sn.next = symmetry(b)
        return s

    def __mul__(self, p):
        s = symmetry(self)
        s.ph *= p
        return s

    def _transform_direction(self, d, n):
        if n == 0 or d == NO_DIRECTION:
            return signed_direction(d)
        if n < 0:
            nme = (self.g - (-n) % self.g) % self.g
            nrest = -((-n) // self.g)
        else:
            nme = n % self.g
            nrest = n // self.g
        if nme == 0:
            if nrest == 0 or self.next is None:
                return signed_direction(d)
            return self.next._transform_direction(d, nrest)
        sd = self._transform_direction(self.S[d].d, nme - 1)
        if self.S[d].flipped:
            sd = _flip(sd)
        if self.next and nrest:
            sd2 = self.next._transform_direction(sd.d, nrest)
            if sd.flipped:
                sd2 = _flip(sd2)
            return sd2 * self.ph
        return sd * self.ph

    def _transform_component(self, c, n):
        return _direction_component(c, self._transform_direction(
            _component_direction(c), n).d)

    def transform(self, obj, n):
        if isinstance(obj, ivec):
            out = ivec._make(obj.dim, obj.t)
            if n == 0:
                return out
            for d in _dirs(obj.dim):
                s = self._transform_direction(d, n)
                delta = obj.t[d] - self.i_symmetry_point.t[d]
                sp_sd = self.i_symmetry_point.t[s.d]
                out.t[s.d] = sp_sd - delta if s.flipped else sp_sd + delta
            return out
        if isinstance(obj, vec):
            if n == 0:
                return vec._make(obj.dim, obj.t)
            delta = vec._make(obj.dim, obj.t)
            for d in _dirs(obj.dim):
                s = self._transform_direction(d, n)
                deltad = obj.t[d] - self.symmetry_point.t[d]
                delta.t[s.d] = -deltad if s.flipped else deltad
            return self.symmetry_point + delta
        if isinstance(obj, volume):
            return volume(self.transform(obj.get_min_corner(), n),
                          self.transform(obj.get_max_corner(), n))
        return self._transform_direction(obj, n)  # direction int

    def transform_unshifted(self, ov, n):
        if n == 0:
            return ivec._make(ov.dim, ov.t)
        out = zero_ivec(ov.dim)
        for d in _dirs(ov.dim):
            s = self._transform_direction(d, n)
            out.t[s.d] = -ov.t[d] if s.flipped else ov.t[d]
        return out

    def phase_shift(self, c, n):
        if _is_derived(c):
            if c < _ENERGY:  # Poynting flux: E x H gets |phase|^2
                ds = self._transform_direction(_component_direction(c), n)
                ph = ds.phase.conjugate() * ds.phase if isinstance(ds.phase, complex) \
                    else ds.phase * ds.phase
                return -ph if ds.flipped else ph
            return 1.0
        if c in (20, 21, 22):  # Dielectric, Permeability, NO_COMPONENT
            return 1.0
        sd = self._transform_direction(_component_direction(c), n)
        phase, flip = sd.phase, sd.flipped
        if 5 <= c < 10 or 15 <= c < 20:  # H or B: pseudovector handedness
            have_one = have_two = False
            for d in range(5):
                td = self._transform_direction(d, n)
                if td.flipped:
                    flip = not flip
                shift = (td.d - d + 6) % 3
                if shift == 1:
                    have_one = True
                if shift == 2:
                    have_two = True
            if have_one and have_two:
                flip = not flip
        return -phase if flip else phase

    def is_primitive(self, p):
        m = self.multiplicity()
        if m == 1:
            return True
        for i in range(1, m):
            pp = self.transform(p, i)
            if p.dim == D2:
                if pp.x() + pp.y() < p.x() + p.y():
                    return False
                if (pp.x() + pp.y() == p.x() + p.y() and p.y() > p.x()
                        and pp.y() <= pp.x()):
                    return False
            elif p.dim == D3:
                s0, s1 = pp.x() + pp.y() + pp.z(), p.x() + p.y() + p.z()
                if s0 < s1:
                    return False
                if s0 == s1 and pp.x() + pp.y() - pp.z() < p.x() + p.y() - p.z():
                    return False
                if (s0 == s1 and pp.x() + pp.y() - pp.z() == p.x() + p.y() - p.z()
                        and pp.x() - pp.y() - pp.z() < p.x() - p.y() - p.z()):
                    return False
            else:  # D1, Dcyl
                if pp.z() < p.z():
                    return False
        return True

    def symmetry_reduce(self, gl):
        """SWIG name for reduce(): weight-minimize a volume_list by symmetry."""
        m = self.multiplicity()
        glnew = None
        g = gl
        while g is not None:
            sn = 0
            while sn < m:
                gS = self.transform(g.v, sn)
                cS = self._transform_component(g.c, sn)
                gn = glnew
                while gn is not None:
                    if gn.c == cS and gn.v.round_float() == gS.round_float():
                        break
                    gn = gn.next
                if gn is not None:
                    cc_ph = self.phase_shift(g.cc, sn)
                    gn.weight += (g.weight * self.phase_shift(g.c, sn)
                                  * (cc_ph.conjugate() if isinstance(cc_ph, complex)
                                     else cc_ph))
                    break
                sn += 1
            if sn == m and g.weight != 0.0:
                glnew = volume_list(g.v, g.c, g.weight, glnew, g.cc)
            g = g.next
        # reduce volumes redundant with themselves; drop zero weights
        gprev, g = None, glnew
        while g is not None:
            halve = [False] * 5
            weight = g.weight
            for sn in range(1, m):
                if (g.c == self._transform_component(g.c, sn)
                        and g.v.round_float() == self.transform(g.v, sn).round_float()):
                    for d in _dirs(g.v.dim):
                        if self._transform_direction(d, sn).flipped:
                            halve[d] = True
                            break
                    g.weight += weight * self.phase_shift(g.c, sn)
            for d in _dirs(g.v.dim):
                if halve[d]:
                    g.v.set_direction_max(
                        d, g.v.in_direction_min(d) + 0.5 * g.v.in_direction(d))
            if g.weight == 0.0:
                if gprev is None:
                    glnew = g.next
                else:
                    gprev.next = g.next
            else:
                gprev = g
            g = g.next
        return glnew

    def __eq__(self, sym):
        if not isinstance(sym, symmetry):
            return NotImplemented
        m = self.multiplicity()
        if m != sym.multiplicity():
            return False
        return all(self._transform_direction(d, sn) == sym._transform_direction(d, sn)
                   for sn in range(1, m) for d in range(5))

    def __ne__(self, sym):
        eq = self.__eq__(sym)
        return NotImplemented if eq is NotImplemented else not eq


def identity():
    return symmetry()


def mirror(axis, gv):
    s = identity()
    s.g = 2
    s.S[axis] = signed_direction(axis, True)
    s.symmetry_point = gv.center()
    s.i_symmetry_point = gv.icenter()
    return s


def rotate2(axis, gv):
    if axis > 2:
        raise RuntimeError("Can only rotate2 in 2D or 3D.")
    s = identity()
    s.g = 2
    s.S[(axis + 1) % 3] = signed_direction((axis + 1) % 3, True)
    s.S[(axis + 2) % 3] = signed_direction((axis + 2) % 3, True)
    s.symmetry_point = gv.center()
    s.i_symmetry_point = gv.icenter()
    return s


def rotate4(axis, gv):
    if axis > 2:
        raise RuntimeError("Can only rotate4 in 2D or 3D.")
    s = identity()
    s.g = 4
    s.S[(axis + 1) % 3] = signed_direction((axis + 2) % 3, True)
    s.S[(axis + 2) % 3] = signed_direction((axis + 1) % 3, False)
    s.symmetry_point = gv.center()
    s.i_symmetry_point = gv.icenter()
    return s


def r_to_minus_r_symmetry(m):
    s = identity()
    s.g = 2
    s.S[R] = signed_direction(R, True)
    s.S[P] = signed_direction(P, True)
    s.symmetry_point = zero_vec(Dcyl)
    s.i_symmetry_point = zero_ivec(Dcyl)
    if m == int(m):
        s.ph = -1.0 if int(m) & 1 else 1.0
    else:
        s.ph = cmath.exp(1j * m * math.pi)
    return s


class symm_matrix:
    """meep_geom symm_matrix — symmetric 3x3 (chi1inv) tensor entries."""

    def __init__(self, m00=0.0, m01=0.0, m02=0.0, m11=0.0, m12=0.0, m22=0.0):
        self.m00, self.m01, self.m02 = m00, m01, m02
        self.m11, self.m12, self.m22 = m11, m12, m22


class vector3:
    """libctl vector3 — plain {x, y, z} C struct (mp.Vector3 is the scripting type)."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class geom_box:
    """libctlgeom geom_box — low/high Vector3 corners (plain data)."""

    def __init__(self, low=None, high=None):
        self.low = low if low is not None else Vector3()
        self.high = high if high is not None else Vector3()


class volume_list:
    """meep::volume_list — linked list of (volume, component, weight)."""

    def __init__(self, *args):
        v, c = args[0], args[1]
        self.v = volume(v.get_min_corner(), v.get_max_corner())  # by value, like C++
        self.c = c
        self.weight = args[2] if len(args) > 2 else 1.0
        self.next = args[3] if len(args) > 3 else None
        self.cc = args[4] if len(args) > 4 else NO_COMPONENT


class material_function:
    """meep::material_function — pointwise eps/mu/chi callback protocol."""

    def set_volume(self, v):
        pass  # default ignores the restriction volume, like the C++ base

    def unset_volume(self):
        pass

    def chi1p1(self, ft, r):
        return 1.0

    def eps(self, r):
        return self.chi1p1(_E_STUFF, r)

    def has_mu(self):
        return False

    def mu(self, r):
        return self.chi1p1(_H_STUFF, r)

    def has_conductivity(self, c):
        return False

    def conductivity(self, c, r):
        return 0.0

    def normal_vector(self, ft, v):
        raise NotImplementedError(
            "spherical-quadrature normals are not ported; interfaces are handled "
            "by the raster blend in meep_torch.geometry.scene")

    def eff_chi1inv_row(self, c, v, tol=1e-4, maxeval=100000):
        raise NotImplementedError(
            "subpixel averaging is not ported; meep_torch.geometry.scene "
            "rasterizes with area-fraction blending instead")

    def is_thread_safe(self):
        return False

    def sigma_row(self, c, r):
        return [0.0, 0.0, 0.0]

    def has_chi3(self, c):
        return False

    def chi3(self, c, r):
        return 0.0

    def has_chi2(self, c):
        return False

    def chi2(self, c, r):
        return 0.0


class simple_material_function(material_function):
    """meep::simple_material_function — one scalar f(vec) drives everything."""

    def __init__(self, func):
        self.func = func

    def chi1p1(self, ft, r):
        return self.func(r)

    def eps(self, r):
        return self.func(r)

    def mu(self, r):
        return self.func(r)

    def conductivity(self, c, r):
        return self.func(r)

    def sigma_row(self, c, r):
        row = [0.0, 0.0, 0.0]
        row[_component_index(c)] = self.func(r)
        return row

    def chi3(self, c, r):
        return self.func(r)

    def chi2(self, c, r):
        return self.func(r)


_GEOM_MSG = ("libctl geometry-tree material lookup is not ported; materials "
             "rasterize via meep_torch.geometry.scene.rasterize_scene")


class geom_epsilon(material_function):
    """meepgeom geom_epsilon — structure present for parity; the libctl
    geometry-tree lookups raise (the shim rasterizes eps arrays instead)."""

    def __init__(self, *args):
        self.geometry = args[0] if len(args) > 0 else None
        self.extra_materials = args[1] if len(args) > 1 else None
        self.restricted_volume = args[2] if len(args) > 2 else None

    def set_volume(self, v):
        self.restricted_volume = v

    def unset_volume(self):
        self.restricted_volume = None

    def is_thread_safe(self):
        return False

    def _raise(self, *a, **k):
        raise NotImplementedError(_GEOM_MSG)

    add_susceptibilities = _raise
    chi = _raise
    chi1p1 = _raise
    chi2 = _raise
    chi3 = _raise
    conductivity = _raise
    eff_chi1inv_matrix = _raise
    eff_chi1inv_row = _raise
    fallback_chi1inv_row = _raise
    get_material_pt = _raise
    has_chi = _raise
    has_chi2 = _raise
    has_chi3 = _raise
    has_conductivity = _raise
    has_mu = _raise
    set_cond_profile = _raise
    sigma_row = _raise
