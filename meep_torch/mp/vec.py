# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp.Vector3 -- meep-compatible 3D vector (2D uses x,y; z carried for W9)."""
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

    def norm(self) -> float:
        return float(np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2))


def interpolate(n: int, kpoints: list) -> list:
    """meep mp.interpolate(n, [k0, k1]) -> n+1 k-points along the segment."""
    if len(kpoints) != 2:
        raise ValueError("interpolate supports exactly 2 endpoints here")
    a, b = kpoints[0], kpoints[1]
    return [Vector3(a.x + (b.x - a.x) * i / (n + 1),
                    a.y + (b.y - a.y) * i / (n + 1),
                    a.z + (b.z - a.z) * i / (n + 1)) for i in range(n + 2)]