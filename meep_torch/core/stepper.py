from typing import NamedTuple

import torch

from .grid import Grid


class TMState(NamedTuple):
    Ez: torch.Tensor       # (nx, ny)        E_z at time n
    Hx: torch.Tensor       # (nx, ny-1)      at n-1/2
    Hy: torch.Tensor       # (nx-1, ny)      at n-1/2
    Dz: torch.Tensor       # (nx, ny)
    P_cur: torch.Tensor    # (n_poles, nx, ny)
    P_prev: torch.Tensor   # (n_poles, nx, ny)
    psi_Ezy: torch.Tensor  # (nx, ny-1)      CPML memory, dEz/dy in Hx update
    psi_Ezx: torch.Tensor  # (nx-1, ny)      dEz/dx in Hy update
    psi_Hyx: torch.Tensor  # (nx-2, ny-2) or (nx-2, ny) if y_pmc
    psi_Hxy: torch.Tensor  # same shape as psi_Hyx


def init_tm_state(grid: Grid, n_poles: int = 0, y_pmc: bool = False) -> TMState:
    nx, ny = grid.nx, grid.ny
    ny_e = ny if y_pmc else ny - 2
    return TMState(
        Ez=grid.zeros(nx, ny),
        Hx=grid.zeros(nx, ny - 1),
        Hy=grid.zeros(nx - 1, ny),
        Dz=grid.zeros(nx, ny),
        P_cur=grid.zeros(n_poles, nx, ny),
        P_prev=grid.zeros(n_poles, nx, ny),
        psi_Ezy=grid.zeros(nx, ny - 1),
        psi_Ezx=grid.zeros(nx - 1, ny),
        psi_Hyx=grid.zeros(nx - 2, ny_e),
        psi_Hxy=grid.zeros(nx - 2, ny_e),
    )
