# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Scene -> rasterized engine coefficient fields.

Painter's-algorithm compositing: later items overwrite earlier where their coverage > 0,
blending at their own boundary. TM uses arithmetic subpixel blending; TE uses the in-plane
Kottke triple (blend_te: xx, yy and the off-diagonal xy) composited against the prior
arithmetic field.
Dispersive materials contribute their Lorentz poles to a union list with coverage-weighted
masks; chi3 is painted coverage-weighted from a separate chi3_items list.
"""
from typing import NamedTuple, Sequence, Union

import torch

from ..core.coeffs import _avg_to_ey, build_te_coeffs
from ..core.grid import Grid
from ..core.materials import LorentzPole, Material
from .polygons import Polygon
from .raster import coverage, normals
from .smoothing import blend_te, blend_tm

SceneItem = tuple[Polygon, Union[Material, float]]


class RasterizedMaterials(NamedTuple):
    eps_tm: torch.Tensor       # (nx, ny) arithmetic blend
    eps_te_xx: torch.Tensor    # (nx, ny) Kottke diagonal xx -- the TE steppers
    eps_te_yy: torch.Tensor    # (nx, ny) Kottke diagonal yy -- read these two
    eps_te_xy: torch.Tensor    # (nx, ny) Kottke off-diagonal; no stepper reads
                               # it yet -- see tasks/port_full-tensor-eps-smoothing
    poles: list                # list[LorentzPole], union over all materials
    pole_masks: list           # list[Tensor (nx, ny)], coverage-weighted
    chi3: torch.Tensor         # (nx, ny) Kerr coefficient


def _material_value(m):
    """float -> (eps_value, ()); Material -> (eps_inf, poles tuple)."""
    if isinstance(m, Material):
        return float(m.eps_inf), tuple(m.poles)
    return float(m), ()


def _find_pole(poles: list, p: LorentzPole):
    for i, q in enumerate(poles):
        if p == q:
            return i
    return None


def rasterize_scene(
    items: Sequence[SceneItem],
    grid: Grid,
    background: Union[Material, float] = 1.0,
    chi3_items: Sequence[tuple[Polygon, float]] = (),
) -> RasterizedMaterials:
    bg_val, bg_poles = _material_value(background)
    eps_tm = torch.full((grid.nx, grid.ny), bg_val, dtype=torch.float64, device=grid.device)
    eps_te_xx = torch.full((grid.nx, grid.ny), bg_val, dtype=torch.float64, device=grid.device)
    eps_te_yy = torch.full((grid.nx, grid.ny), bg_val, dtype=torch.float64, device=grid.device)
    eps_te_xy = torch.zeros((grid.nx, grid.ny), dtype=torch.float64, device=grid.device)
    chi3 = torch.zeros((grid.nx, grid.ny), dtype=torch.float64, device=grid.device)

    poles: list = list(bg_poles)
    masks: list = [torch.ones((grid.nx, grid.ny), dtype=torch.float64, device=grid.device) for _ in bg_poles]

    for poly, mat in items:
        cov = coverage(poly, grid)
        nrm = normals(cov)
        val, item_poles = _material_value(mat)

        # TM: arithmetic composite against the prior field.
        eps_tm_old = eps_tm
        eps_tm = blend_tm(val, eps_tm_old, cov)
        # TE triple: Kottke blend of this item against the prior arithmetic field, composited
        # the same way (boundary of the later item blends against the earlier field where
        # they overlap, against background elsewhere).
        eps_te_xx, eps_te_yy, eps_te_xy = blend_te(val, eps_tm_old, cov, nrm)

        # Pole masks: generic compositing field = cov*1 + (1-cov)*old for the item's own
        # poles (item_val=1), (1-cov)*old for others (item_val=0). Scale existing first,
        # then add cov for own poles.
        for k in range(len(masks)):
            masks[k] = (1.0 - cov) * masks[k]
        for p in item_poles:
            idx = _find_pole(poles, p)
            if idx is None:
                poles.append(p)
                masks.append(cov.clone())
            else:
                masks[idx] = cov + masks[idx]

    for poly, val in chi3_items:
        cov = coverage(poly, grid)
        chi3 = cov * val + (1.0 - cov) * chi3

    return RasterizedMaterials(
        eps_tm=eps_tm, eps_te_xx=eps_te_xx, eps_te_yy=eps_te_yy,
        eps_te_xy=eps_te_xy,
        poles=poles, pole_masks=masks, chi3=chi3,
    )


def te_coeffs_from_scene(grid: Grid, rm: RasterizedMaterials, npml: int = 0):
    """Build Phase 1 TECoeffs from a rasterized scene without modifying coeffs.py.

    Phase 1's build_te_coeffs averages a single cell-center eps field to both Ex and Ey
    positions. Here we pass eps_te_xx as that cell-center field (so eps_x = avg_to_ex of
    the xx component) and then replace eps_y with avg_to_ey of the yy component via
    NamedTuple._replace, leaving every other field intact.
    """
    tc = build_te_coeffs(
        grid,
        eps_inf=rm.eps_te_xx,
        poles=rm.poles,
        pole_masks=rm.pole_masks,
        chi3=rm.chi3,
        npml=npml,
    )
    return tc._replace(eps_y=_avg_to_ey(rm.eps_te_yy))
