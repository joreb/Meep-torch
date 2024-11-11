# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math
from typing import Sequence

import torch

from .grid import Grid


class PointProbe:
    reduce = "stack"

    def __init__(self, ix: int, iy: int):
        self.ix, self.iy = ix, iy

    def sample(self, state, t_idx: int) -> torch.Tensor:
        return state.Ez[self.ix, self.iy]


class DFTFieldsMonitor:
    """Running DFT of one full-grid field component at a set of frequencies.

    Per step accumulates dt * [cos(w t) F, -sin(w t) F] (the Re/Im of
    F * exp(-i w t), matching the FluxMonitor convention); the accumulated
    result has shape (nfreq, 2, nx, ny). Timing follows the TM stepper:
    E components sit at (t_idx+1)*dt, H components at (t_idx+1/2)*dt.
    """

    reduce = "sum"

    def __init__(self, grid: Grid, freqs: Sequence[float], component: str = "Ez"):
        self.component = component
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        F = getattr(state, self.component)
        off = 1.0 if self.component.startswith("E") else 0.5
        t = (t_idx + off) * self.dt
        w = 2 * math.pi * self.freqs[:, None, None]
        return self.dt * torch.stack(
            [torch.cos(w * t) * F, -torch.sin(w * t) * F], dim=1)


class FluxMonitorTM:
    reduce = "sum"

    def __init__(
        self,
        grid: Grid,
        freqs: Sequence[float],
        *,
        x_normal: bool,
        idx: int,
        lo: int,
        hi: int,
    ):
        self.x_normal = x_normal
        self.idx, self.lo, self.hi = idx, lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)
        if x_normal and not (1 <= idx < grid.nx - 1):
            raise ValueError(f"x-normal monitor ix={idx} must be interior")
        if not x_normal and not (1 <= idx < grid.ny - 1):
            raise ValueError(f"y-normal monitor iy={idx} must be interior")

    def sample(self, state, t_idx: int) -> torch.Tensor:
        if self.x_normal:
            E = state.Ez[self.idx, self.lo : self.hi]
            H = 0.5 * (
                state.Hy[self.idx - 1, self.lo : self.hi]
                + state.Hy[self.idx, self.lo : self.hi]
            )
        else:
            E = state.Ez[self.lo : self.hi, self.idx]
            H = 0.5 * (
                state.Hx[self.lo : self.hi, self.idx - 1]
                + state.Hx[self.lo : self.hi, self.idx]
            )
        tE = (t_idx + 1.0) * self.dt
        tH = (t_idx + 0.5) * self.dt
        w = 2 * math.pi * self.freqs[:, None]
        return self.dt * torch.stack(
            [
                torch.cos(w * tE) * E,
                -torch.sin(w * tE) * E,
                torch.cos(w * tH) * H,
                -torch.sin(w * tH) * H,
            ],
            dim=1,
        )

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReE, ImE, ReH, ImH = acc.unbind(1)
        s = (ReE * ReH + ImE * ImH).sum(-1) * dx      # Re(Ê Ĥ*) integrated
        return -0.5 * s if self.x_normal else 0.5 * s


class FluxMonitorTE:
    reduce = "sum"

    def __init__(self, grid: Grid, freqs, *, ix: int, lo: int, hi: int):
        if not (1 <= ix < grid.nx - 1):
            raise ValueError(f"ix={ix} must be interior")
        self.ix, self.lo, self.hi = ix, lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        E = state.Ey[self.ix, self.lo : self.hi]
        H = 0.5 * (state.Hz[self.ix - 1, self.lo : self.hi] + state.Hz[self.ix, self.lo : self.hi])
        # step_te advances E to (t_idx+1/2)*dt and Hz to (t_idx+1)*dt
        tE = (t_idx + 0.5) * self.dt
        tH = (t_idx + 1.0) * self.dt
        w = 2 * torch.pi * self.freqs[:, None]
        return self.dt * torch.stack(
            [torch.cos(w * tE) * E, -torch.sin(w * tE) * E,
             torch.cos(w * tH) * H, -torch.sin(w * tH) * H], dim=1)

    def flux(self, acc: torch.Tensor, dx: float) -> torch.Tensor:
        ReE, ImE, ReH, ImH = acc.unbind(1)
        return 0.5 * (ReE * ReH + ImE * ImH).sum(-1) * dx   # TE: Sx = +Ey*Hz


class FluxMonitorTEY:
    """TE y-normal line: Ex + Hz averaged to the Ex nodes ((i+1/2, iy))."""

    reduce = "sum"

    def __init__(self, grid: Grid, freqs, *, iy: int, lo: int, hi: int):
        if not (1 <= iy < grid.ny - 1):
            raise ValueError(f"iy={iy} must be interior")
        self.iy, self.lo, self.hi = iy, lo, hi
        self.dt = grid.dt
        self.freqs = torch.tensor(freqs, dtype=grid.dtype, device=grid.device)

    def sample(self, state, t_idx: int) -> torch.Tensor:
        E = state.Ex[self.lo : self.hi, self.iy]
        H = 0.5 * (state.Hz[self.lo : self.hi, self.iy - 1]
                   + state.Hz[self.lo : self.hi, self.iy])
        # step_te advances E to (t_idx+1/2)*dt and Hz to (t_idx+1)*dt
        tE = (t_idx + 0.5) * self.dt
        tH = (t_idx + 1.0) * self.dt
        w = 2 * torch.pi * self.freqs[:, None]
        return self.dt * torch.stack(
            [torch.cos(w * tE) * E, -torch.sin(w * tE) * E,
             torch.cos(w * tH) * H, -torch.sin(w * tH) * H], dim=1)
