# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Kottke subpixel smoothing: the effective tensor at a dielectric interface.

Kottke, Farjadpour and Johnson (Phys. Rev. E 77, 036611, 2008) smooth an
interface by averaging each field component the way its own continuity
condition demands: components PARALLEL to the interface are continuous, so eps
averages arithmetically; the component NORMAL to it has continuous D, so eps
averages harmonically. Both statements are about a frame aligned with the
interface normal, which makes the whole construction frame algebra -- the same
kind `core.symbolic` does for the symmetry operators, where a mirror is the
Householder `I - 2 n n^T`.

TWO ROUTES, and they must agree.

- The CLOSED FORM, for scalar constituents. With the rank-1 projector
  `P = n n^T` onto the normal,

      eps_eff = h P + a (I - P)

  `a` arithmetic, `h` harmonic. Manifestly rotation-covariant: nothing picks a
  coordinate axis. This is :func:`blend`, of which :func:`blend_tm` and
  :func:`blend_te` are SLICES -- the TM/TE naming describes the engine's 2D
  polarisation split, not the physics. With an in-plane normal `n_z = 0`, so
  `P_zz = 0`, so `zz` is the plain arithmetic mean, which is all `blend_tm`
  ever was.

- The GENERAL ROUTE, for ANISOTROPIC constituents, which the polarization
  grating needs. An invertible map `tau` sends eps to a matrix whose entries
  average linearly; average, then map back:

      eps_eff = tau_inv( <tau(eps)> )

  This is :func:`blend_tensor_tau`. For scalar eps it collapses to the closed
  form.

Both are DERIVED and proved equal to each other symbolically in
`symbolic_test/kottke.py`; `symbolic_test/test_kottke.py` gates this module
against that derivation, and the two implementations are independent -- torch
over batched matrices against sympy over symbols -- so agreement is evidence
rather than a restatement. meep assembles the same objects in
`src/anisotropic_averaging.cpp` (closed form, on the inverse:
`P minveps + (I - P)/meps`) and `src/meepgeom.cpp` (the tau map, component for
component).

THE OFF-DIAGONAL. `blend_te` used to return only the diagonal,
`eps_xx = nx^2 h + (1 - nx^2) a` and its yy partner, dropping
`eps_xy = (h - a) nx ny`. That was exact for axis-aligned interfaces
(`nx ny = 0`) and worst at 45 degrees, where the dropped term reaches 0.67x the
diagonal at 12:1 contrast -- a first-order term of the same magnitude, absent
because the engine had nowhere to put it, not because it was small. It is
returned now. NOTHING CONSUMES IT YET: the TE steppers multiply by two scalar
fields, so `RasterizedMaterials.eps_te_xy` is carried and unread until
`tasks/port_full-tensor-eps-smoothing.md` changes that, and no band or flux has
yet been shown to move.

A CAUTION, measured. Moving eps onto a field's own Yee node with a plain
ARITHMETIC average is first order across an interface and made `stepper_fv2d`
WORSE on every row it was tried on. Use these blends, not a mean.
"""
import numpy as np
import torch

from .raster import normals


def blend(eps_in, eps_out, cov: torch.Tensor, nrm: torch.Tensor):
    """Kottke's effective tensor for scalar constituents: `h P + a (I - P)`.

    ONE rule. `blend_tm` and `blend_te` below are slices of this, not separate
    conventions -- the TM/TE naming is an artefact of the engine's 2D
    polarisation split, not of the physics. With an in-plane normal (`n_z = 0`):

        zz  ->  n_z^2 h + (1 - n_z^2) a  =  a        the arithmetic mean
        xx, yy, xy                       ->  the projector block

    so TM's "arithmetic is already second order because Ez is parallel to every
    in-plane interface" is the same statement as `P_zz = 0`. meep does not split
    them either: `eff_chi1inv_row(component c, ...)` takes the component and
    returns its row of a single tensor.

    Returns `(xx, yy, zz, xy)`; `xz` and `yz` vanish for an in-plane normal.
    Derived in `symbolic_test/kottke.py`.
    """
    a = cov * eps_in + (1.0 - cov) * eps_out
    h = 1.0 / (cov / eps_in + (1.0 - cov) / eps_out)
    nx, ny = nrm[..., 0], nrm[..., 1]
    nx2, ny2 = nx ** 2, ny ** 2
    boundary = (nx2 + ny2) > 0.5           # unit normal where set, zero elsewhere
    zero = torch.zeros_like(a * nx2)
    return (torch.where(boundary, nx2 * h + (1 - nx2) * a, a),
            torch.where(boundary, ny2 * h + (1 - ny2) * a, a),
            a,                              # n_z = 0, so zz is purely arithmetic
            torch.where(boundary, nx * ny * (h - a), zero))


def blend_tm(eps_in, eps_out, cov: torch.Tensor) -> torch.Tensor:
    """The zz entry of :func:`blend`. Needs no normal, because it cannot use one."""
    return cov * eps_in + (1.0 - cov) * eps_out


def blend_te(eps_in, eps_out, cov: torch.Tensor, nrm: torch.Tensor):
    """Kottke in-plane blend, as ``(eps_xx, eps_yy, eps_xy)``.

    `h P + a (I - P)` with the rank-1 projector `P = n n^T`, `a` arithmetic
    (parallel to the interface, where E is continuous) and `h` harmonic (normal
    to it, where D is). Derived in `symbolic_test/kottke.py`; meep assembles the
    same object in `anisotropic_averaging.cpp`, on the inverse:
    `P minveps + (I - P)/meps`.

    The OFF-DIAGONAL `(h - a) nx ny` used to be dropped. It is zero for an
    axis-aligned interface and reaches 0.67x the diagonal at 30 degrees with
    12:1 contrast, so it was never a refinement -- it was a first-order term of
    the same size, absent because the engine had nowhere to put it. Returned
    now so the rasterizer stops discarding it; see
    `tasks/port_full-tensor-eps-smoothing.md` for what has to change downstream
    before a stepper can use it.
    """
    xx, yy, _, xy = blend(eps_in, eps_out, cov, nrm)
    return xx, yy, xy


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

    It does not reach 2nd order, because this function projects onto the
    DIAGONAL and drops the off-diagonal eps_xy an interface oblique to the
    axes carries -- which is most of a circular rod's boundary. (:func:`blend_te`
    returns that term now; it is discarded here, see the slice below.) Raising
    ``supersample`` does not help (16 measures the same as 4), confirming the
    limit is the projection, not the coverage estimate.

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
    # (xx, yy) only: the mpb eigensolver this feeds carries a diagonal eps,
    # so the off-diagonal blend_te now returns has nowhere to go here. Kept as
    # a two-tuple deliberately rather than pushed onto callers that would
    # discard it -- `tasks/port_full-tensor-eps-smoothing.md` is where that
    # changes.
    return blend_te(hi, lo, cov_t, normals(cov_t))[:2]


# --------------------------------------------------------------- Kottke tau --
# The general (anisotropic) Kottke blend. `blend_te` above is the scalar
# closed form, restricted to the in-plane block; this is the tau-map route,
# which takes tensor constituents. Derived and proved
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
