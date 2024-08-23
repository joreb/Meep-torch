import math

import torch


def cpml_profiles(
    n: int,
    npml: int,
    offset: float,
    dt: float,
    dx: float,
    m: int = 3,
    R0: float = 1e-6,
    alpha_max: float = 0.05,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """CFS-PML (CPML) 1D coefficient profiles at positions i + offset.

    Returns (b, a, kappa), each shape (n,). Recursion: psi' = b*psi + a*d,
    effective derivative = d/kappa + psi'.
    """
    x = torch.arange(n, dtype=dtype, device=device) + offset
    if npml > 0:
        u = torch.clamp((npml - x) / npml, min=0.0) + torch.clamp(
            (x - (n - 1 - npml)) / npml, min=0.0
        )
        u = torch.clamp(u, max=1.0)
    else:
        u = torch.zeros(n, dtype=dtype, device=device)
    sigma_max = 0.0
    if npml > 0:
        sigma_max = -(m + 1) * math.log(R0) / (2.0 * npml * dx)
    sigma = sigma_max * u**m
    alpha = torch.where(u > 0, alpha_max * (1.0 - u), torch.zeros_like(u))
    kappa = torch.ones_like(u)
    b = torch.exp(-(sigma / kappa + alpha) * dt)
    denom = kappa * (sigma + kappa * alpha)
    a = torch.where(sigma > 0, sigma * (b - 1.0) / denom.clamp(min=1e-300), torch.zeros_like(b))
    return b, a, kappa
