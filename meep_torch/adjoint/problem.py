# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.adjoint-shaped optimization API on torch autograd.

MaterialGrid + DesignRegion + OptimizationProblem: rho in [0,1] ->
conic filter -> tanh projection -> eps interpolation -> painted into
the base eps map -> user run function -> scalar objective. Calling the
problem returns (f0, dJ/drho) like meep.adjoint's OptimizationProblem
-- except the gradient is plain reverse-mode autograd through the FDTD
run (the W8 pattern), no separate adjoint solve.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .filters import conic_filter, tanh_projection


class MaterialGrid:
    """Design density grid (meep MaterialGrid): rho in [0, 1] linearly
    interpolates eps between eps_min and eps_max."""

    def __init__(self, shape, eps_min: float = 1.0, eps_max: float = 12.0):
        self.shape = tuple(shape)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)

    def eps(self, rho_projected: torch.Tensor) -> torch.Tensor:
        return self.eps_min + (self.eps_max - self.eps_min) * rho_projected


class DesignRegion:
    """Paints a MaterialGrid into an index box of the simulation eps map
    (bilinear upsampling from the design grid to the region cells)."""

    def __init__(self, material_grid: MaterialGrid, *, ix0, ix1, iy0, iy1):
        self.mg = material_grid
        self.box = (ix0, ix1, iy0, iy1)

    def paint(self, base_eps: torch.Tensor, rho_projected: torch.Tensor) -> torch.Tensor:
        ix0, ix1, iy0, iy1 = self.box
        up = F.interpolate(rho_projected[None, None], size=(ix1 - ix0, iy1 - iy0),
                           mode="bilinear", align_corners=False)[0, 0]
        eps = base_eps.clone()
        eps[ix0:ix1, iy0:iy1] = self.mg.eps(up)
        return eps


class OptimizationProblem:
    """opt() -> (f0, dJ/drho). rho lives on the problem (opt.rho); update
    it in-place between calls (any torch optimizer or plain ascent).
    beta may be ramped by assigning opt.beta (binarization schedule)."""

    def __init__(self, run_fn, base_eps: torch.Tensor, design_region: DesignRegion,
                 *, filter_radius: float = 0.0, dx: float = 1.0,
                 beta: float = 1.0, eta: float = 0.5, rho0=None):
        self.run_fn = run_fn
        self.base_eps = base_eps
        self.region = design_region
        self.filter_radius = filter_radius
        self.dx = dx
        self.beta = beta
        self.eta = eta
        self.rho = (rho0.clone() if rho0 is not None
                    else torch.full(design_region.mg.shape, 0.5,
                                    dtype=base_eps.dtype))

    def _objective(self, rho: torch.Tensor):
        r = rho
        if self.filter_radius > 0:
            r = conic_filter(r, self.filter_radius, self.dx)
        r = tanh_projection(r, self.beta, self.eta)
        eps = self.region.paint(self.base_eps, r)
        return self.run_fn(eps)

    def __call__(self, need_gradient: bool = True):
        rho = self.rho.detach().clone().requires_grad_(need_gradient)
        J = self._objective(rho)
        if not need_gradient:
            return float(J), None
        J.backward()
        return float(J), rho.grad.detach().numpy().copy()

    def eval_at_delta(self, i: int, j: int, h: float) -> float:
        """Objective with rho[i, j] perturbed by h (finite-difference checks)."""
        rho = self.rho.detach().clone()
        rho[i, j] += h
        with torch.no_grad():
            return float(self._objective(rho))
