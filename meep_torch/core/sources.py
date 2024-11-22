# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import torch

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

    def __init__(self, grid: Grid, ix: int, j0: int, j1: int, profile, amp_func=None):
        if not 0 <= ix < grid.nx:
            raise ValueError(f"source ix={ix} outside grid nx={grid.nx}")
        if not 0 <= j0 < j1 <= grid.ny:
            raise ValueError(f"source segment j0={j0}, j1={j1} invalid for ny={grid.ny}")
        self.profile = profile
        self.dt = grid.dt
        mask = grid.zeros(grid.nx, grid.ny)
        if amp_func is None:
            mask[ix, j0:j1] = 1.0
        else:
            for j in range(j0, j1):
                mask[ix, j] = float(amp_func((j + 0.5) * grid.dx))
        self.mask = mask

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)


class ContinuousSource:
    """Faithful meep ContinuousSource: CW carrier with an is-1-cosine turn-on.

    ``width`` is the ramp time (meep default ~1/frequency); ``slowness`` multiplies
    it (turn-on takes ``slowness*width``). With ``width=None`` falls back to a few
    periods (the W1 CWRamp-equivalent default).
    """

    def __init__(self, frequency: float, width: float | None = None,
                 slowness: float = 3.0, cutoff: float = 5.0, fcen: float | None = None):
        self.fcen = float(fcen if fcen is not None else frequency)
        self.t_ramp = (slowness * width) if width is not None else max(1.0, slowness / self.fcen)
        self.cutoff = cutoff

    def amplitude(self, t: float) -> float:
        if self.t_ramp <= 0:
            ramp = 1.0
        else:
            s = min(1.0, t / self.t_ramp)
            ramp = 0.5 - 0.5 * math.cos(math.pi * s)
        return math.sin(2 * math.pi * self.fcen * t) * ramp


class CustomSource:
    """User time profile: ``amplitude(t) = callable(t)``."""

    def __init__(self, fn):
        self.fn = fn

    def amplitude(self, t: float) -> float:
        return float(self.fn(t))


class SoftSource:
    """General 2D soft source: ``Jz(x,y,t) = profile(t) * mask(x,y)``.

    ``mask`` is an (nx, ny) spatial-weight tensor (build it with amp_func); used
    for area sources (both polarizations via the TM stepper).
    """

    def __init__(self, grid: Grid, mask, profile):
        self.mask = mask
        self.profile = profile
        self.dt = grid.dt

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)


class ObliqueLineSource:
    """Phased real line source launching a +x planewave at angle ``theta``.

    Injects ``Jz(y, t) = env(t) * cos(k_y y - 2*pi*f*t)`` on the line x=ix,
    y in [j0, j1), with ``k_y = 2*pi*f*sin(theta)``. One real run; a soft source
    radiates both +/-x (PML absorbs both), the +x lobe is the oblique planewave.
    For a one-way (TFSF) oblique launch use the Bloch eigenmode path (W2 Wave 2).
    """

    def __init__(self, grid: Grid, ix: int, j0: int, j1: int, profile,
                 theta: float, fcen: float | None = None):
        if not 0 <= ix < grid.nx:
            raise ValueError(f"source ix={ix} outside grid nx={grid.nx}")
        self.grid = grid
        self.ix, self.j0, self.j1 = ix, j0, j1
        self.profile = profile
        self.dt = grid.dt
        self.fcen = float(fcen if fcen is not None else getattr(profile, "fcen", 0.0))
        self.ky = 2.0 * math.pi * self.fcen * math.sin(theta)
        self.omega = 2.0 * math.pi * self.fcen
        self.mask = grid.zeros(grid.nx, grid.ny)

    def amplitude(self, t_idx: int) -> float:
        t = (t_idx + 0.5) * self.dt
        env = self.profile.amplitude(t)
        ys = (torch.arange(self.j0, self.j1, dtype=torch.float64) + 0.5) * self.grid.dx
        wave = torch.cos(self.ky * ys - self.omega * t) * env
        m = self.grid.zeros(self.grid.nx, self.grid.ny)
        m[self.ix, self.j0:self.j1] = wave.to(m.dtype)
        self.mask = m
        return 1.0


class RandomSource:
    """White-noise current: fresh N(0, 1) scalar amplitude per step on a
    fixed spatial mask (seeded, reproducible). Spectrally flat drive for
    noise/thermal-emission studies.

    ponytail: one scalar per step (spatially coherent noise); per-point
    independent noise needs a per-step mask redraw -- add when an example
    (NoisyLorentzian thermal emission) demands it.
    """

    def __init__(self, grid: Grid, ix: int, j0: int, j1: int,
                 amplitude: float = 1.0, seed: int = 0):
        mask = grid.zeros(grid.nx, grid.ny)
        mask[ix, j0:j1] = 1.0
        self.mask = mask
        self.dt = grid.dt
        self.amp = float(amplitude)
        self.gen = torch.Generator().manual_seed(seed)

    def amplitude(self, t_idx: int) -> float:
        return self.amp * float(torch.randn(1, generator=self.gen).item())
