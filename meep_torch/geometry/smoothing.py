# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# Diagonal-projected Kottke averaging, implemented from the published scheme
# (Kottke, Farjadpour & Johnson, Phys. Rev. E 77, 036611 (2008)).
import numpy as np
import torch

from .raster import normals


def blend_tm(eps_in, eps_out, cov: torch.Tensor) -> torch.Tensor:
    return cov * eps_in + (1.0 - cov) * eps_out


def blend_te(eps_in, eps_out, cov: torch.Tensor, nrm: torch.Tensor):
    a = cov * eps_in + (1.0 - cov) * eps_out
    h = 1.0 / (cov / eps_in + (1.0 - cov) / eps_out)
    nx2 = nrm[..., 0] ** 2
    ny2 = nrm[..., 1] ** 2
    boundary = (nx2 + ny2) > 0.5           # unit normal where set, zero elsewhere
    eps_xx = torch.where(boundary, nx2 * h + (1 - nx2) * a, a)
    eps_yy = torch.where(boundary, ny2 * h + (1 - ny2) * a, a)
    return eps_xx, eps_yy


def blend_te_supersampled(blocks):
    """Kottke (eps_xx, eps_yy) straight from a supersampled rasterization.

    ``blocks`` is the fine grid reshaped to (nx, ss, ny, ss) -- the coarse
    cell layout with its ss x ss subsamples still separate, which is what a
    supersampling rasterizer already has in hand before it averages.

    Callers that own polygons should use :func:`blend_te` with
    :func:`~meep_torch.geometry.raster.coverage` instead; this is for the
    rasterizers that only ever materialize sampled epsilon (the mpb
    ModeSolver builds its cell from mp.Cylinder/Block masks, never
    polygons). The two phases and the fill fraction are read off the block,
    and the interface normal comes from the coverage gradient.

    Why it matters: TE carries E in-plane across dielectric boundaries and
    the component normal to an interface is discontinuous there, so a plain
    volume average smears the discontinuity. Averaging each component the
    way its own continuity condition demands roughly halves the band error
    -- measured on the mpb_tutorial rod lattice against meep.mpb, TE went
    5.43 -> 2.45% at resolution 16, 2.60 -> 1.22% at 32, 1.16 -> 0.67% at 64.

    It does not reach 2nd order, because :func:`blend_te` is the
    *diagonal-projected* Kottke variant and drops the off-diagonal eps_xy
    an interface oblique to the axes carries -- which is most of a circular
    rod's boundary. Raising ``supersample`` does not help (16 measures the
    same as 4), confirming the limit is the projection, not the coverage
    estimate.

    One interface per cell is Kottke's own assumption; the grid is treated
    as two phases (its global extreme eps values), the same simplification
    meep makes.
    """
    nx, ss, ny, _ = blocks.shape
    flat = blocks.transpose(0, 2, 1, 3).reshape(nx, ny, ss * ss)
    # The coverage field must reference ONE phase across the whole grid --
    # the normal comes from its gradient, so a per-cell reference makes a
    # uniform "inside" cell and a uniform "outside" cell both read 1.0, the
    # gradient vanish, and every interface go unsmoothed (or worse, pick up
    # a spurious two-sided gradient at a boundary ring).
    hi, lo = float(flat.max()), float(flat.min())
    if hi <= lo:                                  # uniform cell grid
        eps = torch.as_tensor(flat.mean(axis=2), dtype=torch.float64)
        return eps, eps
    cov_t = torch.as_tensor((flat >= hi - 1e-12).mean(axis=2),
                            dtype=torch.float64)
    return blend_te(hi, lo, cov_t, normals(cov_t))
