# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D geometry rasterization and MPB's effective inverse-dielectric tensor.

The 2D sibling :mod:`meep_torch.mpb.raster` samples a cell in fractional
coordinates and coverage-averages epsilon per component. This module does
the same in 3D and then does the part 2D gets to skip: it builds the full
3x3 effective INVERSE dielectric tensor per cell that the planewave
operator contracts against.

Why a tensor and not a number. At a material interface the two Maxwell
continuity conditions pull in opposite directions -- ``E`` tangential is
continuous, so the tangential effective epsilon is the arithmetic mean;
``D`` normal is continuous, so the normal effective INVERSE epsilon is the
arithmetic mean of the inverses (the harmonic mean of epsilon). One scalar
cannot satisfy both, and using either mean for both components is what
makes a staircased interface converge at first order instead of second.
Kottke's construction (MPB ``maxwell_eps.c``) splits them along the local
interface normal:

    eta_eff = <1/eps> n n^T + (1/<eps>) (I - n n^T)

which reduces to ``I/eps`` exactly where a cell is uniform, so a cell in
the middle of a sphere or in the middle of the air costs nothing and is
not a special case in the code.

The normal itself is estimated from the sampled block: the direction along
which epsilon varies most, taken as the first moment of the deviation from
the block mean. A cell straddling a flat interface gets the interface
normal; a cell containing a corner gets something in between, which is the
documented limit of the method rather than a bug in it.

Conventions follow the rest of the package: the grid is uniform in
LATTICE (fractional) coordinates with cell centres at
``(i + 0.5)/n - 1/2``, object centres are read in LATTICE coordinates
along the UNIT basis vectors (see :func:`~meep_torch.mpb.raster.object_mask_2d`
for the long form of why), and the last-listed object wins.
"""
from __future__ import annotations

import math

import numpy as np

from ..mp.geometry import Block, Cylinder, Sphere
from .raster import eps_diag_of

__all__ = ["lattice_matrix_3d", "frac_min_image_3d", "cell_centres_3d",
           "object_mask_3d", "rasterize_3d", "effective_eps_inv",
           "eps_inv_from_geometry"]


def _vec3(v) -> np.ndarray:
    a = np.asarray([v.x, v.y, v.z] if hasattr(v, "x") else v, dtype=float)
    return a.ravel()[:3]


def lattice_matrix_3d(a1, a2, a3) -> np.ndarray:
    """Columns are the lattice vectors, so ``A @ (u, v, w)`` is cartesian."""
    return np.column_stack([_vec3(a1), _vec3(a2), _vec3(a3)])


def frac_min_image_3d(A: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Wrap cartesian offsets ``d`` (..., 3) to the nearest lattice image.

    Fractional, wrap each component to [-1/2, 1/2), back to cartesian. On
    an oblique cell this is not the same as wrapping each cartesian axis
    independently, and the difference is which copy of an object counts as
    the nearest one -- the fcc primitive cell is 60 degrees, so it is the
    normal case here rather than an exotic one.
    """
    f = d @ np.linalg.inv(A).T
    f -= np.round(f)
    return f @ A.T


def cell_centres_3d(a1, a2, a3, n1: int, n2: int, n3: int,
                    mesh_size: int = 1):
    """Cartesian sample points, shape ``(n1*m, n2*m, n3*m, 3)``.

    ``mesh_size`` samples each cell ``m`` times finer along every axis --
    MPB's own parameter name and meaning.
    """
    A = lattice_matrix_3d(a1, a2, a3)
    ns = (n1 * mesh_size, n2 * mesh_size, n3 * mesh_size)
    axes = [(np.arange(n) + 0.5) / n - 0.5 for n in ns]
    F = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return F @ A.T


def object_mask_3d(obj, P: np.ndarray, A: np.ndarray, centre_shift,
                   basis=None) -> np.ndarray:
    """Boolean mask of sample points ``P`` (..., 3) inside ``obj``.

    ``basis`` is the triple of UNIT lattice basis vectors; object centres
    are measured along them, not along x/y/z. See the 2D twin for the
    measurement that pinned this down.
    """
    c = _vec3(obj.center) - _vec3(centre_shift)
    if basis is not None:
        b = np.asarray([_vec3(v) for v in basis], dtype=float)   # rows b_i
        c = c @ b
    d = frac_min_image_3d(A, P - c[None, None, None, :])

    if isinstance(obj, Sphere):
        return np.sum(d * d, axis=-1) < obj.radius ** 2

    if isinstance(obj, Cylinder):
        ax = _vec3(obj.axis)
        nrm = float(np.linalg.norm(ax))
        if nrm == 0:
            raise ValueError("Cylinder.axis must be nonzero")
        ax = ax / nrm
        d_par = d @ ax
        d_perp2 = np.clip(np.sum(d * d, axis=-1) - d_par ** 2, 0.0, None)
        inside = d_perp2 < obj.radius ** 2
        if math.isfinite(obj.height):
            inside = inside & (np.abs(d_par) < obj.height / 2)
        return inside

    if isinstance(obj, Block):
        e = np.asarray([_vec3(obj.e1), _vec3(obj.e2), _vec3(obj.e3)])
        # Rotated blocks are fine here: project onto the block's own axes
        # rather than refusing, which is what the 2D path had to do because
        # its mask worked per cartesian axis. hole_slab needs an unrotated
        # slab, but nothing about the projection cares.
        nrm = np.linalg.norm(e, axis=1)
        if np.any(nrm == 0):
            raise ValueError("Block e1/e2/e3 must be nonzero")
        e = e / nrm[:, None]
        mask = np.ones(d.shape[:-1], dtype=bool)
        for axis in range(3):
            half = float(obj.size[axis]) / 2
            if math.isinf(half):
                continue
            mask &= np.abs(d @ e[axis]) < half
        return mask

    raise NotImplementedError(
        f"{type(obj).__name__} rasterization not supported in 3D "
        "(Sphere/Cylinder/Block only)")


def _scalar_eps(material) -> float:
    """Isotropic epsilon of a material, or refuse clearly."""
    ex, ey, ez = eps_diag_of(material)
    if not (abs(ex - ey) < 1e-12 and abs(ey - ez) < 1e-12):
        raise NotImplementedError(
            f"anisotropic material (eps diag {ex:g}, {ey:g}, {ez:g}) is not "
            "supported by the 3D planewave path yet: the Kottke tensor "
            "would have to blend two tensors rather than two scalars")
    return float(ex)


def rasterize_3d(a1, a2, a3, n1: int, n2: int, n3: int, geometry,
                 geometry_center, default_material, mesh_size: int = 3,
                 basis=None) -> np.ndarray:
    """Supersampled scalar epsilon, shape ``(n1*m, n2*m, n3*m)``.

    Last-listed object wins, which is meep's and MPB's documented rule and
    the opposite of what a "paint in order" reading suggests.
    """
    if mesh_size < 1:
        raise ValueError(f"mesh_size must be >= 1, got {mesh_size}")
    P = cell_centres_3d(a1, a2, a3, n1, n2, n3, mesh_size)
    A = lattice_matrix_3d(a1, a2, a3)
    fine = np.full(P.shape[:-1], _scalar_eps(default_material), dtype=float)
    for obj in geometry:
        fine = np.where(object_mask_3d(obj, P, A, geometry_center, basis),
                        _scalar_eps(obj.material), fine)
    return fine


def effective_eps_inv(fine: np.ndarray, mesh_size: int, a1, a2, a3,
                      uniform_tol: float = 1e-12) -> np.ndarray:
    """Kottke effective inverse-dielectric tensor, ``(n1, n2, n3, 3, 3)``.

    ``fine`` is the supersampled scalar epsilon from :func:`rasterize_3d`.
    Each coarse cell owns an ``m x m x m`` block; a block whose spread is
    below ``uniform_tol`` is homogeneous and gets ``I/eps`` exactly, which
    is both cheaper and avoids estimating a normal from round-off.
    """
    m = int(mesh_size)
    ns = tuple(s // m for s in fine.shape)
    if tuple(s * m for s in ns) != fine.shape:
        raise ValueError(
            f"fine grid {fine.shape} is not {ns} x mesh_size {m}")
    # (n1, n2, n3, m, m, m) view of the blocks.
    blocks = fine.reshape(ns[0], m, ns[1], m, ns[2], m).transpose(0, 2, 4, 1, 3, 5)
    flat = blocks.reshape(ns + (m ** 3,))

    mean_eps = flat.mean(axis=-1)
    mean_eta = (1.0 / flat).mean(axis=-1)

    eye = np.eye(3)
    out = np.empty(ns + (3, 3), dtype=float)
    out[...] = eye * (1.0 / mean_eps)[..., None, None]

    spread = flat.max(axis=-1) - flat.min(axis=-1)
    mixed = spread > uniform_tol
    if not np.any(mixed):
        return out

    # Sample offsets within a block, cartesian, so the moment is a real
    # direction on an oblique cell and not a fractional-space artefact.
    A = lattice_matrix_3d(a1, a2, a3)
    off = (np.arange(m) + 0.5) / m - 0.5
    F = np.stack(np.meshgrid(off / ns[0], off / ns[1], off / ns[2],
                             indexing="ij"), axis=-1).reshape(-1, 3)
    D = F @ A.T                                    # (m^3, 3)

    dev = flat[mixed] - mean_eps[mixed][..., None]  # (n_mixed, m^3)
    moment = dev @ D                                # (n_mixed, 3)
    nrm = np.linalg.norm(moment, axis=-1)
    good = nrm > 1e-30
    nhat = np.zeros_like(moment)
    nhat[good] = moment[good] / nrm[good][:, None]

    nn = nhat[:, :, None] * nhat[:, None, :]        # (n_mixed, 3, 3)
    eta_n = mean_eta[mixed][:, None, None]
    eta_t = (1.0 / mean_eps[mixed])[:, None, None]
    blend = eta_n * nn + eta_t * (eye[None] - nn)
    # A block that varies but has no net direction (a symmetric inclusion
    # dead centre) gets the isotropic harmonic mean rather than a random
    # frame: nn is zero there, which would otherwise hand it eta_t on all
    # three axes and quietly ignore the inclusion.
    blend[~good] = eye[None] * ((1.0 / 3.0) * eta_n[~good]
                                + (2.0 / 3.0) * eta_t[~good])
    out[mixed] = blend
    return out


def eps_inv_from_geometry(a1, a2, a3, n1: int, n2: int, n3: int, geometry,
                          geometry_center, default_material,
                          mesh_size: int = 3, basis=None):
    """``(eps_inv_tensor, eps_scalar)`` in one call.

    ``eps_scalar`` is the block-averaged epsilon, which is what
    ``get_epsilon()`` reports and what the examples plot; the tensor is
    what the operator runs on. Returning both keeps the two from being
    computed by different code and drifting.
    """
    fine = rasterize_3d(a1, a2, a3, n1, n2, n3, geometry, geometry_center,
                        default_material, mesh_size, basis)
    m = int(mesh_size)
    ns = (n1, n2, n3)
    blocks = fine.reshape(ns[0], m, ns[1], m, ns[2], m).transpose(0, 2, 4, 1, 3, 5)
    eps_scalar = blocks.reshape(ns + (m ** 3,)).mean(axis=-1)
    return effective_eps_inv(fine, m, a1, a2, a3), eps_scalar
