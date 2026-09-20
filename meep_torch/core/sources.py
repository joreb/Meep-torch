# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import cmath
import math

import torch

from .grid import Grid


def cw_amplitude(src, t_idx: int) -> complex:
    """Complex CW current amplitude of an engine source at step ``t_idx``.

    solve_cw needs the analytic (complex) current the real-field steppers only
    ever see the real part of. A profile that knows its own continuation
    supplies ``camplitude(t)`` with ``camplitude(t).real == amplitude(t)``;
    everything else falls back to the real value, which is the right answer
    for a profile that has no single carrier frequency."""
    prof = getattr(src, "profile", None)
    dt = getattr(src, "dt", None)
    if prof is None or dt is None or not hasattr(prof, "camplitude"):
        return complex(src.amplitude(t_idx))
    return prof.camplitude((t_idx + 0.5) * dt)


class GaussianPulse:
    def __init__(self, fcen: float, fwidth: float, cutoff: float = 5.0):
        self.fcen = fcen
        self.w = 1.0 / fwidth
        self.t0 = cutoff * self.w
        self.t_end = 2.0 * self.t0

    def quadrature(self, t: float) -> tuple[float, float]:
        """(Re, Im) of the analytic signal J(t) = env(t) * exp(-i phi(t)).

        meep's source is complex, J(t) = envelope * exp(-i w t); a real-valued
        engine injects Re[J] and needs Im[J] separately for anything that
        carries a phase -- the Bloch path, and a complex amp_func.

        Both quadratures come from ONE envelope and ONE phase here, and
        amplitude() is the real part of this rather than its own formula. The
        mp Bloch shim used to retype the envelope and the carrier to build the
        pair, which is the same expression living in two files: they agree
        until one is edited, and a drifted quadrature shows up as a slow phase
        error nothing raises on.
        """
        ph = 2 * math.pi * self.fcen * (t - self.t0)
        env = math.exp(-((t - self.t0) ** 2) / (2 * self.w ** 2))
        return env * math.cos(ph), -env * math.sin(ph)

    def amplitude(self, t: float) -> float:
        return self.quadrature(t)[0]


class CWRamp:
    def __init__(self, fcen: float, t_ramp: float):
        self.fcen = fcen
        self.t_ramp = t_ramp

    def amplitude(self, t: float) -> float:
        ramp = min(1.0, t / self.t_ramp)
        return math.sin(2 * math.pi * self.fcen * t) * ramp


def source_measure(spans, dx: float, ndim: int = 2) -> float:
    """The measure a source of these spans carries, derived, not typed.

    ``spans`` is the cell count along each axis of the source's own extent.
    meep's rule is one line, ``src/sources.cpp:501``::

        LOOP_OVER_DIRECTIONS(gv.dim, d)
          if (where.in_direction(d) == 0.0 && !nosize_direction(d))
            data.amp *= gv.a;          // a = resolution = 1/dx

    A meep source carries a current DENSITY, so the field it produces depends
    on the current integrated over its extent divided by the cell volume. Each
    axis the source is a DELTA in contributes one factor of 1/dx; each axis it
    EXTENDS along contributes dx of measure against the same 1/dx, cancelling.
    So the measure is dx^(-n_delta) -- and n_delta is read off the spans rather
    than declared.

    Deriving it is not tidiness. The same rule was previously typed out in
    LineSource, and typed out a second time in HLineSource when the horizontal
    orientation was added; a third copy would have arrived with the first 3D
    source. Two copies of a rule are two things that can disagree, and this one
    is invisible when it does -- a wrong measure scales the field smoothly and
    nothing raises. wiki/bugs/source-normalization-extent.md is the record of
    what that costs.

        point in 2D  spans (1, 1)   -> 1/dx^2
        line  in 2D  spans (n, 1)   -> 1/dx
        plane in 3D  spans (n, m, 1) -> 1/dx
    """
    n_delta = sum(1 for n in spans if n <= 1)
    if len(spans) < ndim:                    # unlisted axes are deltas too
        n_delta += ndim - len(spans)
    return dx ** (-n_delta)


class LineSource:
    """Soft current source on the vertical segment x=ix, y in [j0, j1)."""

    def __init__(self, grid: Grid, ix: int, j0: int, j1: int, profile,
                 amp_func=None, amps=None):
        if not 0 <= ix < grid.nx:
            raise ValueError(f"source ix={ix} outside grid nx={grid.nx}")
        if not 0 <= j0 < j1 <= grid.ny:
            raise ValueError(f"source segment j0={j0}, j1={j1} invalid for ny={grid.ny}")
        self.profile = profile
        self.dt = grid.dt
        mask = grid.zeros(grid.nx, grid.ny)
        if amps is not None:              # per-cell weights (IndexedSource)
            mask[ix, j0:j1] = torch.as_tensor(amps, dtype=grid.dtype,
                                              device=grid.device)
        elif amp_func is None:
            mask[ix, j0:j1] = 1.0
        else:
            for j in range(j0, j1):
                mask[ix, j] = float(amp_func((j + 0.5) * grid.dx))
        # meep's source is a current DENSITY integrated over its extent, so
        # the field it produces does not depend on the grid. The discrete
        # injection adds amp*mask to D, which means mask carries one 1/dx for
        # every axis the source is a delta in -- and that is derivable from
        # the extent itself, not something a caller should have to declare:
        # a segment spanning >1 cell is a line (1/dx), a single cell is a
        # point (1/dx^2). Deriving it here keeps every construction path
        # consistent by construction; passing it in as a flag let the indexed
        # and plain paths disagree.
        # Without any normalization the field scaled as dx: measured 1/40 of
        # meep at resolution 40 and 1/20 at 20.
        # One integral, not two cases: every occupied cell carries the
        # source measure it holds, divided by the cell area.
        #   extended (j1-j0 > 1): each cell holds dx of length -> dx / dx^2
        #   point    (single cell): the measure is 1        ->  1 / dx^2
        # A point and a line genuinely differ in dimension (A/m vs A/m^2 in
        # 2D), so the extent has to enter -- but it enters once, here, and is
        # read off the span rather than declared by a caller.
        self.mask = mask * source_measure((j1 - j0, 1), grid.dx)

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
        return math.sin(2 * math.pi * self.fcen * t) * self._ramp(t)

    def _ramp(self, t: float) -> float:
        if self.t_ramp <= 0:
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * min(1.0, t / self.t_ramp))

    def camplitude(self, t: float) -> complex:
        """Analytic continuation of amplitude() under the e^{-i.omega.t}
        convention: ``camplitude(t).real == amplitude(t)`` exactly, since
        Re[i e^{-i.omega.t}] = sin(omega t).

        meep's own continuous_src_time carries the current as e^{-i.omega.t}
        (real part cos), a quarter period ahead of this engine's sin. Either
        is a valid CW current; they scale the frequency-domain solution by a
        constant phase, which every |field| and every relative-error metric
        is blind to. Matching THIS engine's real convention is what keeps a
        solve_cw field comparable to the same cell's time-domain run."""
        return 1j * cmath.exp(-2j * math.pi * self.fcen * t) * self._ramp(t)


class CustomSource:
    """User time profile: ``amplitude(t) = callable(t)``."""

    def __init__(self, fn):
        self.fn = fn

    def amplitude(self, t: float) -> float:
        # meep's CustomSource src_func is routinely complex -- the chirped
        # pulse tutorial writes exp(i.2.pi.v0.t) * exp(...) directly -- and a
        # real-field run takes its real part. Doing that through float() on a
        # complex raised a ComplexWarning on every timestep, which reads like
        # a mistake rather than the intended convention.
        v = self.fn(t)
        return float(v.real) if isinstance(v, complex) else float(v)


class HLineSource:
    """Soft current source on the HORIZONTAL segment y=jy, x in [i0, i1).

    LineSource's mirror image. It exists because a source line's orientation
    is not fixed: a beam travelling along +y needs its source extended along x,
    and meep places no constraint either way.

    THE MEASURE IS THE SAME RULE, read off the extent exactly as LineSource
    reads it -- dx/dx^2 for a span of more than one cell, 1/dx^2 for a single
    one. That is the point of putting this here rather than building a mask in
    the mp shim: two orientations deriving the same measure from one piece of
    code cannot disagree, and the shim is left with nothing to get wrong.
    """

    def __init__(self, grid: Grid, jy: int, i0: int, i1: int, profile,
                 amp_func=None, amps=None):
        if not 0 <= jy < grid.ny:
            raise ValueError(f"source jy={jy} outside grid ny={grid.ny}")
        if not 0 <= i0 < i1 <= grid.nx:
            raise ValueError(
                f"source segment i0={i0}, i1={i1} invalid for nx={grid.nx}")
        self.profile = profile
        self.dt = grid.dt
        mask = grid.zeros(grid.nx, grid.ny)
        if amps is not None:
            mask[i0:i1, jy] = torch.as_tensor(amps, dtype=grid.dtype,
                                              device=grid.device)
        elif amp_func is None:
            mask[i0:i1, jy] = 1.0
        else:
            for i in range(i0, i1):
                mask[i, jy] = float(amp_func((i + 0.5) * grid.dx))
        self.mask = mask * source_measure((i1 - i0, 1), grid.dx)

    def amplitude(self, t_idx: int) -> float:
        return self.profile.amplitude((t_idx + 0.5) * self.dt)


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
        ys = (torch.arange(self.j0, self.j1, dtype=torch.float64,
                           device=self.grid.device) + 0.5) * self.grid.dx
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
