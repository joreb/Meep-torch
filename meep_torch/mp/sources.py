# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp source objects -> meep-torch LineSource / ModeSource (via Simulation).

W1 ships basic semantics: GaussianSource -> GaussianPulse; ContinuousSource ->
CWRamp (faithful width/slowness/cutoff lands in W2). Component selects
polarization (Ez/Ex(1D) -> TM, Hz -> TE). EigenModeSource wraps ports/ModeSource.
"""
import cmath
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
_HZ_COMPONENTS = {Hz, Hy, Ey}  # Ey drives the 2D TE (Hz) polarization


def pol_for(component: str) -> str:
    if component in _EZ_COMPONENTS:
        return "tm"
    if component in _HZ_COMPONENTS:
        return "te"
    raise ValueError(f"unsupported component {component!r}")


class _Shifted:
    """Delayed time profile: amplitude 0 before ``start_time``, then the
    wrapped profile shifted so its t=0 lands at ``start_time``. Other
    attributes (fcen, t_end, ...) forward to the wrapped profile."""

    def __init__(self, inner, start_time: float):
        self._inner = inner
        self.start_time = float(start_time)

    def amplitude(self, t: float) -> float:
        if t < self.start_time:
            return 0.0
        return self._inner.amplitude(t - self.start_time)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Windowed:
    """meep custom_src_time semantics: src_func evaluated at t (NOT shifted)
    inside [start_time, end_time], 0 outside."""

    def __init__(self, inner, start_time: float, end_time: float):
        self._inner = inner
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def amplitude(self, t: float) -> float:
        if self.start_time <= t <= self.end_time:
            return self._inner.amplitude(t)
        return 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SourceTime:
    """meep source.py SourceTime: ``is_integrated`` only (stored; True raises
    NotImplementedError at profile() -- the engine injects soft current
    sources, there is no integrated d(current)/dt path). Subclasses supply
    ``_base_profile()`` and start/end-time gating via profile()."""

    def __init__(self, is_integrated=False):
        self.is_integrated = is_integrated

    def _checked_base_profile(self):
        if self.is_integrated:
            raise NotImplementedError(
                "is_integrated=True is not supported: the engine injects soft "
                "current sources only (no integrated d(current)/dt path)")
        return self._base_profile()


class GaussianSource(SourceTime):
    """meep GaussianSource(frequency, width, fwidth, start_time, cutoff,
    is_integrated, wavelength) -> engine GaussianPulse (real cos-modulated
    Gaussian, peak at start_time + cutoff*width). All params supported."""

    def __init__(self, frequency=None, width=0, fwidth=float("inf"),
                 start_time=0, cutoff=5.0, is_integrated=False, wavelength=None):
        if frequency is None and wavelength is None:
            raise ValueError("GaussianSource requires either frequency or wavelength")
        super().__init__(is_integrated=is_integrated)
        self.frequency = 1.0 / wavelength if wavelength else float(frequency)
        self.width = max(width, 1.0 / fwidth)
        self.start_time = start_time
        self.cutoff = cutoff

    @property
    def _fw(self):
        # private adapter: effective fwidth for the engine / EigenModeSource
        return 1.0 / float(self.width)

    def _base_profile(self):
        from ..core.sources import GaussianPulse
        if self.width == 0:
            raise ValueError("GaussianSource needs width or fwidth")
        return GaussianPulse(self.frequency, self._fw, cutoff=self.cutoff)

    def profile(self):
        p = self._checked_base_profile()
        return _Shifted(p, self.start_time) if self.start_time else p

    def fourier_transform(self, freq: float) -> complex:
        """meep's analytic FT (exp(+i*omega*t) convention) of the complex
        profile exp(-i*omega0*(t - t_peak)) * exp(-(t - t_peak)^2 / (2 w^2)):
        w*sqrt(2*pi) * exp(i*omega*t_peak) * exp(-w^2 (omega - omega0)^2 / 2).
        The engine's real cos profile transforms as 0.5*(F(f) + conj(F(-f)))."""
        w = float(self.width)
        t_peak = float(self.start_time) + self.cutoff * w
        omega = 2.0 * math.pi * freq
        omega0 = 2.0 * math.pi * self.frequency
        return (w * math.sqrt(2.0 * math.pi) * cmath.exp(1j * omega * t_peak)
                * math.exp(-0.5 * (w * (omega - omega0)) ** 2))


class ContinuousSource(SourceTime):
    """meep ContinuousSource(frequency, start_time, end_time, width, fwidth,
    cutoff, slowness, wavelength, is_integrated) -> engine ContinuousSource
    (sin carrier, 1-cos turn-on over slowness*width; width=0 falls back to a
    few periods). Not supported: end_time (no smooth turn-off path; raises on
    non-default)."""

    def __init__(self, frequency=None, start_time=0, end_time=1.0e20, width=0,
                 fwidth=float("inf"), cutoff=3.0, slowness=3.0, wavelength=None,
                 is_integrated=False):
        if frequency is None and wavelength is None:
            raise ValueError("ContinuousSource requires either frequency or wavelength")
        super().__init__(is_integrated=is_integrated)
        self.frequency = 1.0 / wavelength if wavelength else float(frequency)
        self.start_time = start_time
        self.end_time = end_time
        self.width = max(width, 1.0 / fwidth)
        self.cutoff = cutoff
        self.slowness = slowness
        self._fcen = self.frequency    # private adapter (EigenModeSource path)

    def _base_profile(self):
        from ..core.sources import ContinuousSource as _CS
        if self.end_time < 1.0e20:
            raise NotImplementedError(
                "ContinuousSource end_time is not supported (engine has no "
                "smooth turn-off); use a GaussianSource pulse instead")
        return _CS(self.frequency, width=(self.width or None),
                   slowness=self.slowness, cutoff=self.cutoff)

    def profile(self):
        p = self._checked_base_profile()
        return _Shifted(p, self.start_time) if self.start_time else p


class CustomSource(SourceTime):
    """meep CustomSource(src_func, start_time, end_time, is_integrated,
    center_frequency, fwidth): src_func(t) inside [start_time, end_time], 0
    outside (NOT shifted). center_frequency/fwidth are stored only (meep uses
    them for DFT decimation, which the engine does not do)."""

    def __init__(self, src_func, start_time=-1.0e20, end_time=1.0e20,
                 is_integrated=False, center_frequency=0, fwidth=0):
        super().__init__(is_integrated=is_integrated)
        self.src_func = src_func
        self.start_time = start_time
        self.end_time = end_time
        self.center_frequency = center_frequency
        self.fwidth = fwidth

    def profile(self):
        p = self._checked_base_profile()
        if self.start_time > -1.0e20 or self.end_time < 1.0e20:
            return _Windowed(p, self.start_time, self.end_time)
        return p

    def _base_profile(self):
        from ..core.sources import CustomSource as _CS
        return _CS(self.src_func)


@dataclass
class Source:
    src: object
    component: str = Ez
    center: Vector3 = None
    size: Vector3 = None
    amplitude: float = 1.0
    amp_func: object = None     # callable(Vector3) -> spatial weight on the source line


class GaussianBeamSource(Source):
    """meep GaussianBeamSource(src, center, volume, component, amplitude,
    amp_func, size, beam_x0, beam_kdir, beam_w0, beam_E0) -- 2D
    amplitude-only approximation.

    Ships as a line Source whose amp_func is the real Gaussian envelope
    exp(-(y^2 + z^2) / beam_w0^2) around the source center (the profile the
    hand-rolled examples/gaussian-beam.py uses), scaled by |beam_E0|'s largest
    component. The engine injects a real static spatial mask times a real
    scalar time profile, so the focusing/tilt phase cannot be applied:
    ``beam_x0`` (focus offset) and ``beam_kdir`` are accepted and stored but
    do NOT curve or steer the beam -- it launches collimated along the line
    normal. amp_func points are relative to the source center (meep
    convention). A user-supplied amp_func overrides the built-in envelope.
    Not supported: volume (raises), component default ALL_COMPONENTS (Ez
    here -- the 2D engine needs one polarization)."""

    def __init__(self, src, center=None, volume=None, component=Ez,
                 amplitude=1.0, amp_func=None, size=None, beam_x0=None,
                 beam_kdir=None, beam_w0=None, beam_E0=None):
        if volume is not None:
            raise NotImplementedError(
                "GaussianBeamSource volume= is not supported; pass center/size")
        super().__init__(src=src, component=component, center=center,
                         size=size, amplitude=amplitude, amp_func=amp_func)
        self.beam_x0 = beam_x0 if beam_x0 is not None else Vector3()
        self.beam_kdir = beam_kdir if beam_kdir is not None else Vector3()
        self.beam_w0 = beam_w0
        self.beam_E0 = beam_E0 if beam_E0 is not None else Vector3()
        if self.amp_func is None:
            if beam_w0 is None:
                raise ValueError("GaussianBeamSource requires beam_w0")
            w0 = float(beam_w0)
            e0 = max(abs(self.beam_E0.x), abs(self.beam_E0.y),
                     abs(self.beam_E0.z)) or 1.0
            self.amp_func = lambda p: e0 * math.exp(-(p.y ** 2 + p.z ** 2) / (w0 ** 2))


@dataclass
class EigenModeSource:
    src: object
    component: str = Ez
    center: Vector3 = None
    size: Vector3 = None
    direction: int = 1
    eig_band: int = 1     # meep convention: 1-based band index (1 = fundamental)
    amplitude: float = 1.0

    def __post_init__(self):
        if self.eig_band < 1:
            raise ValueError(
                f"eig_band is 1-based like meep (1 = fundamental mode), got "
                f"{self.eig_band}")


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