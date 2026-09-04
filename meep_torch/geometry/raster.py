# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import numpy as np
import torch

from ..core.grid import Grid
from .polygons import Polygon


def _clip_halfplane(pts, axis, bound, keep_leq):
    """Sutherland-Hodgman: keep points with pts[:,axis] <= bound (or >=)."""
    out = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ain = (a[axis] <= bound) if keep_leq else (a[axis] >= bound)
        bin_ = (b[axis] <= bound) if keep_leq else (b[axis] >= bound)
        if ain:
            out.append(a)
        if ain != bin_:
            t = (bound - a[axis]) / (b[axis] - a[axis])
            out.append(a + t * (b - a))
    return out


def _area_left_of(pts, x):
    """Area of polygon `pts` (list of 2D points, closed) lying left of x."""
    clipped = _clip_halfplane(pts, axis=0, bound=x, keep_leq=True)
    if len(clipped) < 3:
        return 0.0
    v = np.array(clipped)
    return abs(
        0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
    )


def coverage(poly: Polygon, grid: Grid, origin=(0.0, 0.0)) -> torch.Tensor:
    dx = grid.dx
    cov = np.zeros((grid.nx, grid.ny), dtype=np.float64)
    verts = poly.vertices - np.array(origin)
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    j0 = max(0, int(np.floor(ymin / dx)))
    j1 = min(grid.ny, int(np.ceil(ymax / dx)))
    for j in range(j0, j1):
        strip = _clip_halfplane(list(verts), 1, (j + 1) * dx, True)
        if len(strip) >= 3:
            strip = _clip_halfplane(strip, 1, j * dx, False)
        if len(strip) < 3:
            continue
        sx = np.array(strip)[:, 0]
        i0 = max(0, int(np.floor(sx.min() / dx)))
        i1 = min(grid.nx, int(np.ceil(sx.max() / dx)))
        if i1 <= i0:
            continue
        # prefix areas at cell boundaries i0..i1 (vector of i1-i0+1 evaluations)
        prefix = [_area_left_of(strip, i * dx) for i in range(i0, i1 + 1)]
        cov[i0:i1, j] = np.diff(prefix) / (dx * dx)
    cov = np.clip(cov, 0.0, 1.0)
    # Snap FP noise: fully-covered/empty cells must be exactly 1.0/0.0 so that
    # boundary masks (0 < cov < 1) and interface-normal gradients treat them as
    # interior/exterior. Strip-shoelace noise is ~1e-15; 1e-9 threshold is far
    # below the smallest genuine fractional coverage (~dx-scaled, O(0.01)).
    cov[np.isclose(cov, 1.0, atol=1e-9)] = 1.0
    cov[np.isclose(cov, 0.0, atol=1e-9)] = 0.0
    # coverage feeds the smoothing blends, which combine it with grid-sized
    # engine tensors, so it has to land where the grid lives.
    return torch.tensor(cov, dtype=torch.float64, device=grid.device)


def normals(cov: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(cov[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    gx = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5
    gy = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
    g = torch.stack([-gx, -gy], dim=-1)          # inside -> outside
    length = torch.linalg.norm(g, dim=-1, keepdim=True)
    boundary = ((cov > 0) & (cov < 1)).unsqueeze(-1)
    return torch.where(boundary & (length > 0), g / length.clamp(min=1e-300), torch.zeros_like(g))
