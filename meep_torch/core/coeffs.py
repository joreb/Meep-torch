import math
from typing import NamedTuple, Sequence

import torch

from .grid import Grid
from .pml import cpml_profiles


class TMCoeffs(NamedTuple):
    dt: float
    dx: float
    eps_inf: torch.Tensor
    chi3: torch.Tensor
    has_kerr: bool
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    y_pmc: bool
    b_hy: torch.Tensor
    a_hy: torch.Tensor
    k_hy: torch.Tensor
    b_hx: torch.Tensor
    a_hx: torch.Tensor
    k_hx: torch.Tensor
    b_ex: torch.Tensor
    a_ex: torch.Tensor
    k_ex: torch.Tensor
    b_ey: torch.Tensor
    a_ey: torch.Tensor
    k_ey: torch.Tensor


def _as_field(val, grid: Grid, default: float) -> torch.Tensor:
    if val is None:
        val = default
    if isinstance(val, torch.Tensor):
        if val.shape != (grid.nx, grid.ny):
            raise ValueError(f"expected shape {(grid.nx, grid.ny)}, got {tuple(val.shape)}")
        return val.to(dtype=grid.dtype, device=grid.device)
    return torch.full((grid.nx, grid.ny), float(val), dtype=grid.dtype, device=grid.device)


def build_tm_coeffs(
    grid: Grid,
    eps_inf=None,
    poles: Sequence = (),
    pole_masks: Sequence[torch.Tensor] | None = None,
    chi3=None,
    npml: int = 0,
    y_pmc: bool = False,
) -> TMCoeffs:
    dt, dx = grid.dt, grid.dx
    eps = _as_field(eps_inf, grid, 1.0)
    x3 = _as_field(chi3, grid, 0.0)
    has_kerr = bool((x3 != 0).any())

    n_poles = len(poles)
    c1 = grid.zeros(n_poles, grid.nx, grid.ny)
    c2 = grid.zeros(n_poles, grid.nx, grid.ny)
    c3 = grid.zeros(n_poles, grid.nx, grid.ny)
    for k, pole in enumerate(poles):
        fn, gamma, sigma = pole[0], pole[1], pole[2]
        w0 = 2 * math.pi * fn
        G = 2 * math.pi * gamma
        den = 1.0 + G * dt / 2.0
        mask = pole_masks[k] if pole_masks is not None else torch.ones_like(eps)
        c1[k] = (2.0 - w0 * w0 * dt * dt) / den * torch.ones_like(eps)
        c2[k] = -(1.0 - G * dt / 2.0) / den * torch.ones_like(eps)
        c3[k] = (sigma * w0 * w0 * dt * dt / den) * mask.to(eps)

    if npml * 2 >= grid.nx or (npml * 2 >= grid.ny and not y_pmc):
        raise ValueError(f"npml={npml} too large for grid {grid.nx}x{grid.ny}")

    def prof(n, offset, use_pml=True):
        return cpml_profiles(
            n, npml if use_pml else 0, offset, dt, dx, dtype=grid.dtype, device=grid.device
        )

    # H updates: dEz/dy at half-y (ny-1); dEz/dx at half-x (nx-1)
    b_hy, a_hy, k_hy = prof(grid.ny - 1, 0.5, use_pml=not y_pmc)
    b_hx, a_hx, k_hx = prof(grid.nx - 1, 0.5)
    # D update: dHy/dx at interior integer x (positions 1..nx-2)
    b_ex_f, a_ex_f, k_ex_f = prof(grid.nx, 0.0)
    b_ex, a_ex, k_ex = b_ex_f[1:-1], a_ex_f[1:-1], k_ex_f[1:-1]
    # dHx/dy at integer y: interior (PEC) or full (PMC)
    b_ey_f, a_ey_f, k_ey_f = prof(grid.ny, 0.0, use_pml=not y_pmc)
    if y_pmc:
        b_ey, a_ey, k_ey = b_ey_f, a_ey_f, k_ey_f
    else:
        b_ey, a_ey, k_ey = b_ey_f[1:-1], a_ey_f[1:-1], k_ey_f[1:-1]

    return TMCoeffs(
        dt=dt, dx=dx, eps_inf=eps, chi3=x3, has_kerr=has_kerr,
        c1=c1, c2=c2, c3=c3, y_pmc=y_pmc,
        b_hy=b_hy[None, :], a_hy=a_hy[None, :], k_hy=k_hy[None, :],
        b_hx=b_hx[:, None], a_hx=a_hx[:, None], k_hx=k_hx[:, None],
        b_ex=b_ex[:, None], a_ex=a_ex[:, None], k_ex=k_ex[:, None],
        b_ey=b_ey[None, :], a_ey=a_ey[None, :], k_ey=k_ey[None, :],
    )
