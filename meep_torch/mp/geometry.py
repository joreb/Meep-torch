# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp geometry objects -> meep-torch Polygon scene items (corner-origin).

Block/Cylinder/Prism carry a Medium; resolving to scene items splits the
nondispersive/dispersive eps value from the Kerr chi3 (painted separately by
rasterize_scene chi3_items). Painter's order is preserved (later overrides).
"""
import math
from dataclasses import dataclass

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
    return float(medium), 0.0


def _ngon(radius: float, cx: float, cy: float, gs: GridSpec) -> np.ndarray:
    n = max(16, int(round(2 * math.pi * radius / gs.dx)))
    n = min(n, 256)
    ang = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)


@dataclass
class Block:
    size: Vector3
    center: Vector3 = None
    material: Medium | float = None

    def scene_items(self, gs: GridSpec, dims: int = 2):
        c = self.center or Vector3()
        if dims == 1:
            # 1D: z is the propagation axis -> mapped to x; full y (thin grid).
            x0 = _clip(c.z - self.size.z / 2.0 + gs.sx / 2.0, 0.0, gs.sx)
            x1 = _clip(c.z + self.size.z / 2.0 + gs.sx / 2.0, 0.0, gs.sx)
            if math.isinf(self.size.z):
                x0, x1 = 0.0, gs.sx
            y0, y1 = 0.0, gs.sy
        else:
            x0, y0, x1, y1 = rect_corners(c, self.size, gs)
        v = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        return [(Polygon(v), self.material)]


@dataclass
class Cylinder:
    radius: float
    center: Vector3 = None
    material: Medium | float = None
    height: float = math.inf
    n_pts: int | None = None     # meep-torch ext: N-gon resolution (default: by resolution)

    def scene_items(self, gs: GridSpec, dims: int = 2):
        c = self.center or Vector3()
        x, y = to_corner(c, gs)
        if self.n_pts is not None:
            n = max(3, int(self.n_pts))
            ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
            v = np.stack([x + self.radius * np.cos(ang), y + self.radius * np.sin(ang)], axis=1)
            return [(Polygon(v), self.material)]
        return [(Polygon(_ngon(self.radius, x, y, gs)), self.material)]


@dataclass
class Prism:
    vertices: list  # list[Vector3] in world (center-origin) coords
    material: Medium | float = None
    center: Vector3 = None

    def scene_items(self, gs: GridSpec, dims: int = 2):
        v = [to_corner(p, gs) for p in self.vertices]
        return [(Polygon(np.array(v)), self.material)]


def build_scene_items(geometry, gs: GridSpec, dims: int = 2):
    """Flatten geometry objects -> (eps_items, chi3_items) for rasterize_scene."""
    eps_items, chi3_items = [], []
    for obj in geometry:
        for poly, medium in obj.scene_items(gs, dims):
            val, chi3 = _resolve(medium)
            eps_items.append((poly, val))
            if chi3 != 0.0:
                chi3_items.append((poly, chi3))
    return eps_items, chi3_items