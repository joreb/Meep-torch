# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from dataclasses import dataclass

import numpy as np


def _shoelace(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass(frozen=True)
class Polygon:
    vertices: np.ndarray  # (N, 2) float64, implicitly closed, any winding

    def __post_init__(self):
        v = np.asarray(self.vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
            raise ValueError(f"need >=3 vertices of shape (N,2), got {v.shape}")
        object.__setattr__(self, "vertices", v)
        n = len(v)
        for i in range(n):
            a1, a2 = v[i], v[(i + 1) % n]
            for j in range(i + 1, n):
                if j == i or (j + 1) % n == i or (i + 1) % n == j:
                    continue
                b1, b2 = v[j], v[(j + 1) % n]
                if _segments_intersect(a1, a2, b1, b2):
                    raise ValueError("polygon self-intersects")
        if abs(_shoelace(v)) < 1e-300:
            raise ValueError("polygon has zero area")

    @property
    def area(self) -> float:
        return abs(_shoelace(self.vertices))
