# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""gdsfactory -> meep-torch interface (the gplugins role, GPL-free).

gplugins' meep plugin drives conda pymeep; this thin adapter plays the
same role for meep-torch: take a gdsfactory Component, rasterize its
layer polygons into an effective-index 2D eps map (supersampled
coverage), and report the port positions in grid indices.
"""
import gdsfactory as gf
import numpy as np
import torch

from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.core.grid import Grid


def component_to_eps(component, *, res: int, pad: float, n_core: float,
                     n_clad: float, layer=1):
    """Rasterize a Component's polygons -> (grid, eps, port_index_map).

    Effective-index 2D: core polygons get n_core^2, background n_clad^2.
    Coordinates shift so the padded bbox starts at (0, 0).
    """
    comp = component.copy() if hasattr(component, "copy") else component
    # merge=False: merged annuli come back as self-intersecting keyhole
    # outlines that the rasterizer rejects; the raw pieces paint fine
    polys = comp.get_polygons_points(merge=False)[layer]
    xs = np.concatenate([p[:, 0] for p in polys])
    ys = np.concatenate([p[:, 1] for p in polys])
    x0, y0 = xs.min() - pad, ys.min() - pad
    sx = (xs.max() + pad) - x0
    sy = (ys.max() + pad) - y0
    dx = 1.0 / res
    grid = Grid(nx=int(round(sx * res)), ny=int(round(sy * res)), dx=dx)
    items = [(Polygon(np.asarray(p) - [x0, y0]), n_core ** 2) for p in polys]
    rm = rasterize_scene(items, grid, background=n_clad ** 2)
    ports = {p.name: (int(round((p.center[0] - x0) * res)),
                      int(round((p.center[1] - y0) * res)),
                      p.orientation) for p in comp.ports}
    return grid, rm.eps_tm, ports
