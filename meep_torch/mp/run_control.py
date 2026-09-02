# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep run-control surface: Dft return objects, Volume, step-function
combinators, stop conditions, Harminv. Adapter only (no physics) -- everything
here is bookkeeping around ``f(sim)`` callbacks; Simulation._run_stepped drives
the engine in chunks sized by these hooks (see plan_hooks / chunk_size).
"""
import math
import time
from collections import namedtuple

from .vec import Vector3


class Volume:
    """meep mp.Volume(center, size, dims) -- carries a region's box."""

    def __init__(self, center=None, size=None, dims=2):
        self.center = center or Vector3()
        self.size = size or Vector3()
        self.dims = dims


class DftObj:
    """Handle returned by add_* monitor methods. Retrieval methods accept it
    or the legacy name string interchangeably."""

    def __init__(self, sim, name, freqs):
        self.sim = sim
        self.name = name
        self.freqs = list(freqs)


class DftFlux(DftObj):
    pass


class DftFields(DftObj):
    pass


class DftNear2Far(DftObj):
    pass


class DftLdos(DftObj):
    pass


def dft_name(f):
    """Monitor key from a Dft object or a legacy name string."""
    return f.name if isinstance(f, DftObj) else f


# --- step-function combinators (meep names; each wraps fns called as f(sim)) ---
class at_beginning:
    def __init__(self, *fns):
        self.fns = fns


class at_end:
    def __init__(self, *fns):
        self.fns = fns


class at_time:
    def __init__(self, t, *fns):
        self.t = float(t)
        self.fns = fns


class at_every:
    def __init__(self, dt, *fns):
        self.every = float(dt)
        self.fns = fns


class after_time:
    def __init__(self, t, *fns):
        self.t = float(t)
        self.fns = fns


class before_time:
    def __init__(self, t, *fns):
        self.t = float(t)
        self.fns = fns


class after_sources:
    def __init__(self, *fns):
        self.fns = fns


class during_sources:
    def __init__(self, *fns):
        self.fns = fns


def with_prefix(prefix, *step_funcs):
    """meep with_prefix renames output files; no file output here -> passthrough."""
    return step_funcs[0] if len(step_funcs) == 1 else list(step_funcs)


# --- stop conditions (usable as until= / until_after_sources= values) ---
class stop_when_fields_decayed:
    """meep semantics: fed |field(pt)|^2 every step; every ``dt`` of sim time
    the window max is compared against the running global max -- stop once the
    window max fell to ``decay_by`` of it."""

    def __init__(self, dt, c, pt, decay_by):
        self.dt = float(dt)
        self.c = c
        self.pt = pt
        self.decay_by = float(decay_by)
        self._t0 = 0.0
        self._max = 0.0
        self._cur = 0.0

    def __call__(self, sim):
        self._cur = max(self._cur, abs(sim.get_field_point(self.c, self.pt)) ** 2)
        if sim.round_time() <= self._t0 + self.dt:
            return False
        cur, self._cur = self._cur, 0.0
        self._t0 = sim.round_time()
        self._max = max(self._max, cur)
        return self._max > 0 and cur <= self.decay_by * self._max


def stop_after_walltime(t):
    t0 = time.time()
    return lambda sim: (time.time() - t0) > t


Mode = namedtuple("Mode", ["freq", "decay", "Q", "amp", "err"])   # meep Harminv fields


class Harminv:
    """meep mp.Harminv(c, pt, fcen, df, mxbands): pass to run() like a step
    function. Records field(pt) every step; ``.modes`` after the run via
    modes/harminv (err is not computable from the matrix pencil -> 0.0)."""

    def __init__(self, c, pt, fcen, df, mxbands=None):
        self.c = c
        self.pt = pt
        self.fcen = float(fcen)
        self.df = float(df)
        self.mxbands = mxbands
        self.data = []
        self._dt = None
        self._modes = None

    def __call__(self, sim):
        self._dt = sim.grid.dt
        self.data.append(complex(sim.get_field_point(self.c, self.pt)))
        self._modes = None

    @property
    def modes(self):
        from ..modes.harminv import harminv
        if self._modes is None:
            ms = harminv(self.data, self._dt, self.fcen - self.df / 2.0,
                         self.fcen + self.df / 2.0,
                         max_modes=self.mxbands or 64)
            self._modes = [
                Mode(freq=m.freq, Q=m.Q, amp=m.amp, err=0.0,
                     decay=(math.pi * abs(m.freq) / m.Q if math.isfinite(m.Q) else 0.0))
                for m in ms]
        return self._modes


# --- hook machinery for Simulation._run_stepped ---
def plan_hooks(step_funcs, cutoff):
    """Normalize meep step functions.

    Returns (begin, end, per_step, scheduled): ``begin``/``end`` fire once
    around the run; ``per_step`` is (fn, t_start, t_stop) fired after every
    step whose post-step time t satisfies t >= t_start and t - dt < t_stop;
    ``scheduled`` is [next_t, period_or_None, fns] fired when t crosses next_t.
    """
    begin, end, per_step, scheduled = [], [], [], []
    for f in step_funcs:
        if isinstance(f, at_beginning):
            begin.extend(f.fns)
        elif isinstance(f, at_end):
            end.extend(f.fns)
        elif isinstance(f, at_time):
            scheduled.append([f.t, None, f.fns])
        elif isinstance(f, at_every):
            scheduled.append([0.0, f.every, f.fns])
        elif isinstance(f, after_time):
            per_step.extend((g, f.t, math.inf) for g in f.fns)
        elif isinstance(f, before_time):
            per_step.extend((g, 0.0, f.t) for g in f.fns)
        elif isinstance(f, after_sources):
            per_step.extend((g, cutoff, math.inf) for g in f.fns)
        elif isinstance(f, during_sources):
            per_step.extend((g, 0.0, cutoff) for g in f.fns)
        elif callable(f):
            per_step.append((f, 0.0, math.inf))
        else:
            raise TypeError(f"unsupported step function {f!r}")
    return begin, end, per_step, scheduled


def chunk_size(t, dt, steps_left, per_step, scheduled, has_conds):
    """Engine steps to run before the next hook/stop check must fire."""
    if has_conds:
        return 1                     # stop conditions sample every step (meep)
    k = steps_left
    for (_, a, b) in per_step:
        if t + dt >= a - 1e-9 and t < b - 1e-9:
            return 1                 # active in the coming step
        if t < a:
            k = min(k, max(1, math.ceil((a - t) / dt - 1e-9)))
    for sch in scheduled:
        k = min(k, max(1, math.ceil((sch[0] - t) / dt - 1e-9)))
    return max(1, k)


def fire_per_step(sim, per_step, t, dt):
    for (g, a, b) in per_step:
        if t >= a - 1e-9 and t - dt < b - 1e-9:
            g(sim)


def fire_scheduled(sim, scheduled, t):
    for sch in scheduled:
        while sch[0] is not None and t >= sch[0] - 1e-9:
            for g in sch[2]:
                g(sim)
            sch[0] = sch[0] + sch[1] if sch[1] else None
    scheduled[:] = [s for s in scheduled if s[0] is not None]
