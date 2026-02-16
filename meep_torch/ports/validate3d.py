# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""True-3D validation of 2D effective-index designs (fdtdx-style).

Workflow: optimize a device in 2D (lateral effective index), then
- ``extrude``: turn the 2D eps map into a 3D stack -- cells that are core
  in 2D become an h_core-thick slab of the REAL core index (e.g. SiN
  n = 2.0) in cladding; no effective-index approximation remains,
- ``ModePlaneSourceX``: launch the true quasi-TE fundamental mode of the
  3D input cross-section (full-vector ``solve_2d``) from an x-normal
  plane (dominant Ey component),
- measure guided power with ``FluxMonitor3D`` windows against a straight-
  waveguide reference run (fdtdx sparams two-run pattern).
"""
import numpy as np
import torch

from ..core.monitors_3d import PlaneSource
from ..modes.fd2d import solve_2d


def extrude(eps2d: torch.Tensor, *, n_clad: float, n_core: float,
            h_core: float, pad_z: float, res: int) -> torch.Tensor:
    """2D effective-index eps map -> 3D eps: 2D core cells (above the
    min/max midpoint) become an h_core slab of n_core at z-center."""
    core = eps2d > 0.5 * (float(eps2d.min()) + float(eps2d.max()))
    nz = int(round((h_core + 2 * pad_z) * res))
    k0 = int(round(pad_z * res))
    k1 = k0 + max(1, int(round(h_core * res)))
    eps3 = torch.full((*eps2d.shape, nz), n_clad ** 2, dtype=eps2d.dtype)
    slab = torch.where(core, n_core ** 2, n_clad ** 2)
    eps3[:, :, k0:k1] = slab[:, :, None]
    return eps3


class ModePlaneSourceX(PlaneSource):
    """Quasi-TE fundamental-mode plane source on an x-normal plane.

    Solves the REAL 3D cross-section eps3d[ix] with the full-vector 2D
    mode solver and injects its dominant transverse E (solver Ex = lab Ey)
    as the soft-current profile. Prints/exposes ``neff`` for sanity."""

    def __init__(self, eps3d: torch.Tensor, dx: float, ix: int, f: float,
                 profile, dt: float):
        nx, ny, nz = eps3d.shape
        cs = np.asarray(eps3d[ix].detach().cpu().numpy())
        m = solve_2d(cs, dx, f, k=1)
        self.neff = complex(m.neff[0])
        prof = np.real(np.asarray(m.Ex[..., 0]))       # (ny, nz), lab Ey
        prof = prof / (np.abs(prof).max() + 1e-300)
        prof_y = 0.5 * (prof[:-1, :] + prof[1:, :])    # Ey nodes (ny-1, nz)
        mask = torch.zeros(nx, ny - 1, nz, dtype=eps3d.dtype,
                           device=eps3d.device)
        mask[ix] = torch.from_numpy(prof_y).to(dtype=eps3d.dtype,
                                                  device=eps3d.device)
        self.profile = profile
        self.dt = dt
        self.src = ("Ey", mask)
