# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

from .grid import Grid


class GaussianPulse:
    def __init__(self, fcen: float, fwidth: float, cutoff: float = 5.0):
        self.fcen = fcen
        self.w = 1.0 / fwidth
        self.t0 = cutoff * self.w
        self.t_end = 2.0 * self.t0

    def amplitude(self, t: float) -> float:
        return math.cos(2 * math.pi * self.fcen * (t - self.t0)) * math.exp(
            -((t - self.t0) ** 2) / (2 * self.w**2)
        )


class CWRamp:
    def __init__(self, fcen: float, t_ramp: float):
        self.fcen = fcen
        self.t_ramp = t_ramp

    def amplitude(self, t: float) -> float:
        ramp = min(1.0, t / self.t_ramp)
        return math.sin(2 * math.pi * self.fcen * t) * ramp


class LineSource:
    """Soft current source on the vertical segment x=ix, y in [j0, j1)."""

    def __init__(self, grid: Grid, ix: int, j0: int, j1: int, profile):
        if not 0 <= ix < grid.nx:
            raise ValueError(f"source ix={ix} outside grid nx={grid.nx}")
        if not 0 <= j0 < j1 <= grid.ny:
            raise ValueError(f"source segment j0={j0}, j1={j1} invalid for ny={grid.ny}")
        self.profile = profile
        self.dt = grid.dt
        mask = grid.zeros(grid.nx, grid.ny)
        mask[ix, j0:j1] = 1.0
        self.mask = mask

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)
