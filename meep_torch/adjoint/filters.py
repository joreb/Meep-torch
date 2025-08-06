# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Differentiable filter/projection toolchain (meep.adjoint.filters
equivalents, torch conv2d based -- gradients flow through everything).

conic_filter / gaussian_filter: density blur enforcing a minimum
lengthscale; tanh_projection: smoothed threshold pushing densities to
0/1 (beta ramps toward binarization during optimization).
"""
import math

import torch
import torch.nn.functional as F


def _conv_norm(rho: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
    """Normalized correlation with edge handling: constant fields are
    invariant (kernel renormalized by the in-bounds weight)."""
    k = kern[None, None]
    x = rho[None, None]
    pad = (kern.shape[0] // 2, kern.shape[1] // 2)
    num = F.conv2d(x, k, padding=pad)
    den = F.conv2d(torch.ones_like(x), k, padding=pad)
    return (num / den)[0, 0]


def conic_filter(rho: torch.Tensor, radius: float, dx: float) -> torch.Tensor:
    """Conic (linear cone) blur of radius `radius` (physical units)."""
    n = max(1, int(math.ceil(radius / dx)))
    ax = torch.arange(-n, n + 1, dtype=rho.dtype, device=rho.device) * dx
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    kern = torch.clamp(1.0 - torch.sqrt(X ** 2 + Y ** 2) / radius, min=0.0)
    return _conv_norm(rho, kern)


def gaussian_filter(rho: torch.Tensor, sigma: float, dx: float) -> torch.Tensor:
    """Gaussian blur of standard deviation `sigma` (physical units)."""
    n = max(1, int(math.ceil(3 * sigma / dx)))
    ax = torch.arange(-n, n + 1, dtype=rho.dtype, device=rho.device) * dx
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    kern = torch.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    return _conv_norm(rho, kern)


def tanh_projection(rho: torch.Tensor, beta: float, eta: float = 0.5) -> torch.Tensor:
    """Smoothed Heaviside threshold at `eta` with sharpness `beta`
    (meep.adjoint convention; beta -> inf gives a hard step)."""
    num = torch.tanh(torch.as_tensor(beta * eta, dtype=rho.dtype,
                                     device=rho.device)) \
        + torch.tanh(beta * (rho - eta))
    den = math.tanh(beta * eta) + math.tanh(beta * (1 - eta))
    return num / den
