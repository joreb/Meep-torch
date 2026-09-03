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


def rho_from_eps(eps: torch.Tensor, region: "DesignRegion") -> torch.Tensor:
    """Seed a design density from an existing eps map -- e.g. a rasterized
    gdsfactory component: the region's eps box is normalized to [0, 1] via
    the MaterialGrid endpoints and area-downsampled to the rho grid.
    Clamped to (0.02, 0.98) so tanh projection keeps usable gradients."""
    ix0, ix1, iy0, iy1 = region.box
    mg = region.mg
    r = (eps[ix0:ix1, iy0:iy1] - mg.eps_min) / (mg.eps_max - mg.eps_min)
    r = F.interpolate(r[None, None].to(torch.float64), size=mg.shape,
                      mode="area")[0, 0]
    return r.clamp(0.02, 0.98)


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
                 beta: float = None, eta: float = None, rho0=None):
        self.run_fn = run_fn
        self.base_eps = base_eps
        self.region = design_region
        self.filter_radius = filter_radius
        self.dx = dx
        # None: pick beta/eta up from the design grid (mp.MaterialGrid carries
        # them); a grid without them (or beta=0, meep's "no projection") falls
        # back to the engine defaults. An explicit argument always wins.
        mg = design_region.mg
        self.beta = beta if beta is not None else (getattr(mg, "beta", 0) or 1.0)
        self.eta = eta if eta is not None else (getattr(mg, "eta", None) or 0.5)
        self.rho = (rho0.clone() if rho0 is not None
                    else torch.full(design_region.mg.shape, 0.5,
                                    dtype=base_eps.dtype))

    def _objective(self, rho: torch.Tensor):
        r = rho
        if self.filter_radius > 0:
            if r.dim() == 3:
                from .filters import conic_filter3d
                r = conic_filter3d(r, self.filter_radius, self.dx)
            else:
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


class DesignRegion3D:
    """3D analog of DesignRegion: trilinear paint into an eps voxel box."""

    def __init__(self, material_grid: MaterialGrid, *, ix0, ix1, iy0, iy1,
                 iz0, iz1):
        self.mg = material_grid
        self.box = (ix0, ix1, iy0, iy1, iz0, iz1)

    def paint(self, base_eps: torch.Tensor, rho_projected: torch.Tensor) -> torch.Tensor:
        ix0, ix1, iy0, iy1, iz0, iz1 = self.box
        up = F.interpolate(rho_projected[None, None],
                           size=(ix1 - ix0, iy1 - iy0, iz1 - iz0),
                           mode="trilinear", align_corners=False)[0, 0]
        eps = base_eps.clone()
        eps[ix0:ix1, iy0:iy1, iz0:iz1] = self.mg.eps(up)
        return eps


class RobustOptimizationProblem(OptimizationProblem):
    """Worst-case (minimax) optimization.

    Scenarios: the eroded/dilated/nominal triple via projection thresholds
    `etas` (three-field robust topology optimization), and/or a LIST of
    run functions (frequencies, incidence angles, ...). The returned
    objective is the smooth soft-min over all scenarios
    (-logsumexp(-k J)/k; k -> inf gives the hard min, autograd flows to
    every active scenario) -- maximizing it maximizes the worst case.
    """

    def __init__(self, run_fns, base_eps, design_region, *,
                 etas=(0.5,), softmin_k: float = 50.0, **kw):
        self.run_fns = run_fns if isinstance(run_fns, (list, tuple)) else [run_fns]
        self.etas = tuple(etas)
        self.softmin_k = float(softmin_k)
        super().__init__(self.run_fns[0], base_eps, design_region, **kw)

    def _objective(self, rho: torch.Tensor):
        r = rho
        if self.filter_radius > 0:
            if r.dim() == 3:
                from .filters import conic_filter3d
                r = conic_filter3d(r, self.filter_radius, self.dx)
            else:
                r = conic_filter(r, self.filter_radius, self.dx)
        js = []
        for eta in self.etas:
            proj = tanh_projection(r, self.beta, eta)
            eps = self.region.paint(self.base_eps, proj)
            for fn in self.run_fns:
                js.append(fn(eps))
        J = torch.stack([j if torch.is_tensor(j) else torch.as_tensor(j)
                         for j in js])
        k = self.softmin_k
        return -torch.logsumexp(-k * J, dim=0) / k

    def scenario_values(self):
        """Hard per-scenario objectives at the current rho (reporting)."""
        vals = []
        with torch.no_grad():
            r = self.rho
            if self.filter_radius > 0:
                if r.dim() == 3:
                    from .filters import conic_filter3d
                    r = conic_filter3d(r, self.filter_radius, self.dx)
                else:
                    r = conic_filter(r, self.filter_radius, self.dx)
            for eta in self.etas:
                eps = self.region.paint(self.base_eps,
                                        tanh_projection(r, self.beta, eta))
                for fn in self.run_fns:
                    vals.append(float(fn(eps)))
        return vals
