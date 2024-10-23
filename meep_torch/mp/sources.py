# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp source objects -> meep-torch LineSource / ModeSource (via Simulation).

W1 ships basic semantics: GaussianSource -> GaussianPulse; ContinuousSource ->
CWRamp (faithful width/slowness/cutoff lands in W2). Component selects
polarization (Ez/Ex(1D) -> TM, Hz -> TE). EigenModeSource wraps ports/ModeSource.
"""
import math
from dataclasses import dataclass
from typing import Sequence

from .coords import GridSpec, cell_index, cell_index_y, span_index_range, to_corner
from .vec import Vector3

# Component sentinels (subset meep exposes that the 34 examples use).
Ez = "Ez"
Ex = "Ex"
Ey = "Ey"
Hz = "Hz"
Hy = "Hy"
Dielectric = "Dielectric"

_EZ_COMPONENTS = {Ez, Ex}      # Ex used in 1D -> TM Ez in meep-torch
_HZ_COMPONENTS = {Hz, Hy}


def pol_for(component: str) -> str:
    if component in _EZ_COMPONENTS:
        return "tm"
    if component in _HZ_COMPONENTS:
        return "te"
    raise ValueError(f"unsupported component {component!r}")


@dataclass
class GaussianSource:
    frequency: float | None = None
    fwidth: float | None = None
    fcen: float | None = None
    cutoff: float = 5.0

    def __post_init__(self):
        if self.frequency is not None and self.fcen is None:
            self.fcen = self.frequency
        if self.fwidth is not None:
            self._fw = self.fwidth
        elif hasattr(self, "_fw"):
            pass
        else:
            self._fw = 0.1

    def profile(self):
        from ..core.sources import GaussianPulse
        return GaussianPulse(float(self.fcen), float(self._fw), cutoff=self.cutoff)


@dataclass
class ContinuousSource:
    frequency: float | None = None
    width: float | None = None
    fwidth: float | None = None
    slowness: float = 3.0
    cutoff: float = 5.0

    def __post_init__(self):
        self._fcen = float(self.frequency if self.frequency is not None else self.fcen)

    def profile(self):
        from ..core.sources import CWRamp
        # W1 basic: ramp over a few periods (width/slowness faithfulness -> W2).
        t_ramp = (self.width if self.width else 5.0 / self._fcen)
        return CWRamp(self._fcen, t_ramp)


@dataclass
class Source:
    src: object
    component: str = Ez
    center: Vector3 = None
    size: Vector3 = None
    amplitude: float = 1.0
    amp_func: object = None     # callable(Vector3) -> spatial weight on the source line


@dataclass
class EigenModeSource:
    src: object
    component: str = Ez
    center: Vector3 = None
    size: Vector3 = None
    direction: int = 1
    eig_band: int = 0
    amplitude: float = 1.0


def _point_or_span(center: Vector3, size: Vector3 | None, gs: GridSpec):
    """Return (ix, j0, j1) for a source line normal to x (Ez/Ex/Hy 2D).

    If size is None or its y-extent is 0 -> single-cell point at center;
    else a y-span [j0, j1). x is the propagation/normal axis here.
    """
    c = center or Vector3()
    ix = cell_index(c.x, gs)
    if size is None or size.y == 0:
        jy = max(0, min(gs.ny, cell_index_y(c.y, gs)))
        return ix, jy, jy + 1
    if math.isinf(size.y):
        return ix, 0, gs.ny          # full-y plane source (1D / infinite-height line)
    y0 = c.y - size.y / 2.0 + gs.sy / 2.0
    y1 = c.y + size.y / 2.0 + gs.sy / 2.0
    j0, j1 = span_index_range(max(0.0, y0), min(gs.sy, y1), gs, "y")
    return ix, j0, j1


def build_line_source(spec: Source, gs: GridSpec, pol: str):
    ix, j0, j1 = _point_or_span(spec.center, spec.size, gs)
    # LineSource is constructed on the Grid by Simulation (needs grid); return spec
    return ("line", pol, ix, j0, j1, spec.src.profile(), spec.amplitude, spec.amp_func,
            spec.center, gs)


def build_mode_source(spec: EigenModeSource, gs: GridSpec):
    c = spec.center or Vector3()
    # propagation axis = axis of smallest size component (meep convention)
    s = spec.size or Vector3(0, 1, 0)
    if abs(s.x) <= abs(s.y):
        axis, idx, lo, hi = "x", cell_index(c.x, gs), *_y_span(c, s, gs)
    else:
        axis, idx = "y", max(0, min(gs.ny, cell_index_y(c.y, gs)))
        x0 = c.x - s.x / 2 + gs.sx / 2
        x1 = c.x + s.x / 2 + gs.sx / 2
        lo, hi = span_index_range(x0, x1, gs, "x")
    return ("mode", axis, idx, lo, hi, spec.direction, spec.eig_band,
            spec.src.profile(), spec.amplitude)


def _y_span(c: Vector3, s: Vector3, gs: GridSpec):
    y0 = c.y - s.y / 2 + gs.sy / 2
    y1 = c.y + s.y / 2 + gs.sy / 2
    lo, hi = span_index_range(max(0.0, y0), min(gs.sy, y1), gs, "y")
    return lo, hi