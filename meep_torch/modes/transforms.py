# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Flexcompute
# Copyright (c) 2019 Tyler Hughes
# Derived from the ceviche package (https://github.com/fancompute/ceviche).
# Adaptations (c) 2026 meep-torch contributors, MIT.
"""Coordinate transformations for the 2D mode solver.

Ported from ``tidy3d.components.mode.transforms`` (MIT) and the eps/mu Jacobian
contraction in ``tidy3d.components.mode.solver`` (MIT, lines 218-225). Plain
functions operating on numpy/torch arrays; no pydantic, no ModeSpec.

Bend (radial / cylindrical transform)
-------------------------------------
For a bend in the x-z plane the Jacobian is diagonal,
``J = diag(1, 1, R/u)`` with ``u = R + (x - center_x)`` and ``dwdz = J[2,2] = R/u``.
The eps contraction ``eps' = J eps J^T / det(J)`` (``det(J) = dwdz``) reduces to
per-component scaling:

    eps_xx' = eps_xx / dwdz,  eps_yy' = eps_yy / dwdz,  eps_zz' = eps_zz * dwdz

and the same for mu (mu = 1 -> mu_xx = mu_yy = 1/dwdz, mu_zz = dwdz). The diagonal
solver applies this scaling to the three Yee-position eps components before matrix
assembly; ``apply_bend`` returns just the cell-center (eps_zz) scaling as a
convenience.

Angled transform
----------------
Straightens a tilted waveguide so it is translationally invariant in w. The
Jacobian has off-diagonals ``J[0,2] = -tan(theta) cos(phi)``,
``J[1,2] = -tan(theta) sin(phi)`` and unit diagonal (``det(J) = 1``), so
``eps' = J eps J^T``. The input eps is treated as isotropic (eps on all three
diagonal entries). The result is the full (3, 3, Nx, Ny) tensor consumed by the
tensorial solver (Task 9).
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = ["bend_dwdz", "apply_bend", "angled_eps_tensor"]


def bend_dwdz(Nx: int, Ny: int, dx: float, radius: float, center_x: float) -> np.ndarray:
    """Diagonal Jacobian entry ``J[2,2] = R/u`` for a bend in the x-z plane.

    ``u = R + (x - center_x)`` is evaluated at cell-center x positions and
    broadcast across y; the returned ``(Nx, Ny)`` array is the per-cell ``dwdz``
    used by the diagonal eps/mu contraction.
    """
    i = np.arange(Nx)
    x_cc = (i + 0.5) * dx
    u = radius + (x_cc - center_x)
    if not np.all(u > 0):
        # u <= 0 means the cross-section reaches the bend AXIS or past it, which
        # has no arc length and therefore no bend mode. Deriving the scale factor
        # from a column norm gives R/|u|, and taking that Abs would let the
        # solver run on the degenerate geometry and return modes of a medium with
        # eps_xx < 0. Refusing is the honest option; the typed R/u silently
        # flipped sign here.
        worst = float(np.max(np.abs(x_cc - center_x)))
        raise ValueError(
            f"bend radius {radius} reaches the bend axis: the cross-section "
            f"extends {worst} from the bend centre, so radius must exceed that")
    dwdz = radius / u
    return np.broadcast_to(dwdz[:, None], (Nx, Ny)).copy()


def apply_bend(eps_cc, dx: float, radius: float, center_x: float):
    """Apply the bend Jacobian to the cell-center eps (the eps_zz component:
    ``eps_zz' = eps_zz * dwdz``). Returns the modified cell-center eps as the
    same array type as the input (torch.Tensor or numpy.ndarray).

    WARNING: this convenience helper returns ONLY the eps_zz (cell-center)
    scaling. The full per-component bend contraction (eps_xx/eps_yy divided by
    dwdz, eps_zz multiplied) is applied inside ``fd2d.solve_2d`` when
    ``bend_radius`` is passed. Do NOT combine ``apply_bend`` output with
    ``solve_2d(..., bend_radius=...)`` -- that double-scales eps_zz. Either use
    ``solve_2d(bend_radius=...)`` (full contraction, preferred) or feed
    ``apply_bend`` output into ``solve_2d`` WITHOUT ``bend_radius`` (eps_zz-only
    scaling only), but never both. Kept in ``__all__`` as an explicit
    cell-center diagnostic helper; the double-scaling hazard is documented here
    rather than dropping the function.
    """
    eps_np = np.asarray(eps_cc, dtype=np.float64)
    Nx, Ny = eps_np.shape
    dwdz = bend_dwdz(Nx, Ny, dx, radius, center_x)
    out = eps_np * dwdz
    if isinstance(eps_cc, torch.Tensor):
        return torch.from_numpy(np.ascontiguousarray(out)).to(eps_cc.dtype)
    return out


def angled_eps_tensor(eps_cc, theta: float, phi: float) -> torch.Tensor:
    """Full eps tensor ``(3, 3, Nx, Ny)`` for an angled waveguide.

    The input ``eps_cc`` is treated as isotropic (eps on all three diagonal
    entries of the local tensor). ``eps' = J eps J^T / det(J)`` with
    ``J[0,2] = -tan(theta) cos(phi)``, ``J[1,2] = -tan(theta) sin(phi)``, unit
    diagonal -> ``det(J) = 1``. Returns a ``torch.Tensor`` of dtype matching the
    input (default float64).
    """
    eps_np = np.asarray(eps_cc, dtype=np.float64)
    Nx, Ny = eps_np.shape
    t = np.tan(theta)
    J = np.array(
        [
            [1.0, 0.0, -t * np.cos(phi)],
            [0.0, 1.0, -t * np.sin(phi)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    det = float(np.linalg.det(J))  # = 1

    # eps_diag: (3, 3, N) with eps on the diagonal entries.
    N = Nx * Ny
    eps_t = np.zeros((3, 3, N), dtype=np.float64)
    eps_flat = eps_np.reshape(-1)
    for k in range(3):
        eps_t[k, k, :] = eps_flat

    # eps' = J @ eps @ J^T / det, per cell (einsum over the 3x3 tensor indices).
    Je = np.einsum("ij,jp...->ip...", J, eps_t)          # (3, 3, N)  J @ eps
    eps_prime = np.einsum("ip...,jp...->ij...", Je, J)   # (3, 3, N)  (J eps) @ J^T
    eps_prime = eps_prime / det

    out = eps_prime.reshape(3, 3, Nx, Ny)
    dtype = eps_cc.dtype if isinstance(eps_cc, torch.Tensor) else torch.float64
    return torch.from_numpy(np.ascontiguousarray(out)).to(dtype)
