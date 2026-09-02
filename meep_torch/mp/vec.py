# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp.Vector3 -- meep-compatible 3D vector (2D uses x,y; z carried for W9)."""
import math
from dataclasses import dataclass
from typing import Union

import numpy as np

Number = Union[int, float]


@dataclass(frozen=True)
class Vector3:
    x: Number = 0.0
    y: Number = 0.0
    z: Number = 0.0

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s):
        return Vector3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def dot(self, v) -> float:
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cdot(self, v):
        """conj(self) dot v (meep semantics; equals dot for real components)."""
        return (self.x.conjugate() * v.x + self.y.conjugate() * v.y
                + self.z.conjugate() * v.z)

    def cross(self, v):
        return Vector3(self.y * v.z - self.z * v.y,
                       self.z * v.x - self.x * v.z,
                       self.x * v.y - self.y * v.x)

    def scale(self, s):
        return Vector3(self.x * s, self.y * s, self.z * s)

    def conj(self):
        return Vector3(self.x.conjugate(), self.y.conjugate(), self.z.conjugate())

    def norm(self) -> float:
        return float(np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2))

    def unit(self):
        return self * (1.0 / self.norm())

    def close(self, v, tol: float = 1.0e-7) -> bool:
        return (abs(self.x - v.x) <= tol and abs(self.y - v.y) <= tol
                and abs(self.z - v.z) <= tol)

    def rotate(self, axis, theta):
        """Rotate by theta (radians, right-hand) around axis (length ignored)."""
        u = axis.unit()
        vpar = u * u.dot(self)
        vperp = self - vpar
        return vpar + vperp * math.cos(theta) + u.cross(self) * math.sin(theta)

    # rotate vectors in lattice/reciprocal coords (note that the axis
    # is also given in the corresponding basis):

    def rotate_lattice(self, axis, theta, lat):
        from .geometry import cartesian_to_lattice, lattice_to_cartesian
        a = lattice_to_cartesian(axis, lat)
        v = lattice_to_cartesian(self, lat)
        return cartesian_to_lattice(v.rotate(a, theta), lat)

    def rotate_reciprocal(self, axis, theta, lat):
        from .geometry import cartesian_to_reciprocal, reciprocal_to_cartesian
        a = reciprocal_to_cartesian(axis, lat)
        v = reciprocal_to_cartesian(self, lat)
        return cartesian_to_reciprocal(v.rotate(a, theta), lat)


def interpolate(n: int, kpoints: list) -> list:
    """meep mp.interpolate(n, [k0, k1]) -> n+1 k-points along the segment."""
    if len(kpoints) != 2:
        raise ValueError("interpolate supports exactly 2 endpoints here")
    a, b = kpoints[0], kpoints[1]
    return [Vector3(a.x + (b.x - a.x) * i / (n + 1),
                    a.y + (b.y - a.y) * i / (n + 1),
                    a.z + (b.z - a.z) * i / (n + 1)) for i in range(n + 2)]