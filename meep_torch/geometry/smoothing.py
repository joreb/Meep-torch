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


# --------------------------------------------------------------- Kottke tau --
# The general (anisotropic) Kottke blend. `blend_te` above is the scalar
# closed form with only its diagonal kept; this is the tau-map route, which
# takes tensor constituents and keeps the off-diagonal. Derived and proved
# equal to the closed form in `symbolic_test/kottke.py`; the equality is
# gated in `symbolic_test/test_kottke.py`.
#
# The normal is the BLEND AXIS. Averaging two adjacent cells onto the node
# between them is a two-phase cell whose interface is perpendicular to that
# axis, so no normal estimation is needed -- the thing that usually makes
# Kottke awkward to apply to a sampled (rather than polygonal) epsilon.

_PERM = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}


def _as_matrix(eps6):
    """(exx, exy, exz, eyy, eyz, ezz) -> (..., 3, 3) symmetric."""
    exx, exy, exz, eyy, eyz, ezz = eps6
    rows = [torch.stack([exx, exy, exz], dim=-1),
            torch.stack([exy, eyy, eyz], dim=-1),
            torch.stack([exz, eyz, ezz], dim=-1)]
    return torch.stack(rows, dim=-2)


def _as_six(m):
    return (m[..., 0, 0], m[..., 0, 1], m[..., 0, 2],
            m[..., 1, 1], m[..., 1, 2], m[..., 2, 2])


def _tau(e, sign):
    """Kottke's map (``sign=+1``) and its inverse (``sign=-1``).

    The two differ only in the sign of the first row and column's
    off-diagonals, which is why one function covers both.
    """
    e11 = e[..., 0, 0]
    a01, a02 = e[..., 0, 1] / e11, e[..., 0, 2] / e11
    a10, a20 = e[..., 1, 0] / e11, e[..., 2, 0] / e11
    out = torch.empty_like(e)
    out[..., 0, 0] = -1.0 / e11
    out[..., 0, 1], out[..., 0, 2] = sign * a01, sign * a02
    out[..., 1, 0], out[..., 2, 0] = sign * a10, sign * a20
    out[..., 1, 1] = e[..., 1, 1] - a10 * e[..., 0, 1]
    out[..., 1, 2] = e[..., 1, 2] - a10 * e[..., 0, 2]
    out[..., 2, 1] = e[..., 2, 1] - a20 * e[..., 0, 1]
    out[..., 2, 2] = e[..., 2, 2] - a20 * e[..., 0, 2]
    return out


def blend_tensor_tau(eps6_a, eps6_b, axis: int, cov=0.5):
    """Kottke-blend two tensor media whose interface is normal to ``axis``.

    ``cov`` is the fill fraction of ``eps6_a``. Returns the six independent
    components of the blended tensor. Reduces to harmonic-normal /
    arithmetic-parallel for scalar constituents, which is the whole point:
    a plain arithmetic average of eps across an interface is FIRST order and
    measurably worse than not averaging at all.
    """
    p = _PERM[axis]
    inv_p = [p.index(i) for i in range(3)]
    A = _as_matrix(eps6_a)[..., p, :][..., :, p]
    B = _as_matrix(eps6_b)[..., p, :][..., :, p]
    t = cov * _tau(A, +1) + (1.0 - cov) * _tau(B, +1)
    e = _tau(t, -1)
    e = e[..., inv_p, :][..., :, inv_p]
    return _as_six(e)
