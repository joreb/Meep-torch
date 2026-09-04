# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep run-control surface: Dft return objects, Volume, step-function
combinators, stop conditions, Harminv. Adapter only (no physics) -- everything
here is bookkeeping around ``f(sim)`` callbacks; Simulation._run_until drives
the engine in chunks sized by these hooks (see plan_hooks / chunk_size).
"""
import inspect
import math
import time
from collections import namedtuple

import numpy as np

from .vec import Vector3


class Volume:
    """meep mp.Volume(center, size, dims, is_cylindrical, vertices) -- carries
    a region's box. ``vertices`` computes center/size from a Vector3 list
    (meep style); is_cylindrical is stored (no cylindrical engine)."""

    def __init__(self, center=None, size=None, dims=2, is_cylindrical=False,
                 vertices=None):
        if not vertices:
            self.center = center or Vector3()
            self.size = size or Vector3()
        else:
            vs = np.array([np.array(list(Vector3(*v))) for v in vertices])
            self.center = Vector3(*np.mean(vs, axis=0))
            sizes = []
            for ax in range(3):
                u = np.unique(vs[:, ax])
                sizes.append(0.0 if u.size == 1 else float(np.abs(np.diff(u)[0])))
            self.size = Vector3(*sizes)
        self.dims = dims
        self.is_cylindrical = is_cylindrical

    def get_vertices(self):
        """Corner Vector3s of the box (collapsed dims deduplicated, meep)."""
        lo = self.center - self.size.scale(0.5)
        hi = self.center + self.size.scale(0.5)
        return [Vector3(x, y, z)
                for x in sorted({lo.x, hi.x})
                for y in sorted({lo.y, hi.y})
                for z in sorted({lo.z, hi.z})]

    def get_edges(self):
        """Vertex pairs separated by a box side length (meep get_edges)."""
        vs = self.get_vertices()

        def near(a, b):
            return a == b or abs(a - b) < 1e-10

        return [[vs[i], vs[j]]
                for i in range(len(vs)) for j in range(i + 1, len(vs))
                if near((vs[i] - vs[j]).norm(), self.size.x)
                or near((vs[i] - vs[j]).norm(), self.size.y)
                or near((vs[i] - vs[j]).norm(), self.size.z)]

    def pt_in_volume(self, pt):
        """True when ``pt`` lies inside (or on) the box (meep pt_in_volume)."""
        lo = self.center - self.size.scale(0.5)
        hi = self.center + self.size.scale(0.5)
        return (lo.x <= pt.x <= hi.x and lo.y <= pt.y <= hi.y
                and lo.z <= pt.z <= hi.z)


class DftObj:
    """meep DftObj(func, args): wrapper around the deferred monitor adder
    (meep delays func for chunk load balancing; here it runs eagerly). func
    must return the (sim, name, freqs) triple, which lands on the object for
    the retrieval methods -- they accept a DftObj or the legacy name string
    interchangeably. There is no SWIG layer: swigobj stays None and
    swigobj_attr forwards to this object's own attributes."""

    def __init__(self, func, args):
        self.func = func
        self.args = list(args)
        self.swigobj = None
        self.sim, self.name, freqs = func(*args)
        self.freqs = list(freqs)

    def swigobj_attr(self, attr):
        return getattr(self, attr)


class DftFlux(DftObj):
    """meep DftFlux(func, args), args = [freqs, fluxregions, name]."""

    def __init__(self, func, args):
        super().__init__(func, args)
        self.nfreqs = len(self.freqs)
        self.regions = args[1]
        self.num_components = 4


class DftFields(DftObj):
    """meep DftFields(func, args); args = [freqs, components, name] here."""

    def __init__(self, func, args):
        super().__init__(func, args)
        self.nfreqs = len(self.freqs)
        self.regions = []
        self.num_components = len(args[1])


class DftNear2Far(DftObj):
    """meep DftNear2Far(func, args), args = [freqs, near2farregions, name]."""

    def __init__(self, func, args):
        super().__init__(func, args)
        self.nfreqs = len(self.freqs)
        self.regions = args[1]
        self.num_components = 4

    def flux(self, direction=None, where=None, resolution=None):
        """meep DftNear2Far.flux(direction, where, resolution): far fields
        over the line ``where`` (2D; may lie outside the cell) sampled at
        ``resolution`` points per unit length, reduced to the time-averaged
        Poynting flux 0.5*Re(E x H*) component ``direction`` (mp.X / mp.Y;
        None/AUTOMATIC = the line normal), trapezoid-integrated along the
        line. Returns one value per frequency."""
        c, s = where.center, where.size
        along_y = abs(s.y) >= abs(s.x)
        length = abs(s.y) if along_y else abs(s.x)
        n = max(2, int(round(length * resolution)) + 1)
        ts = np.linspace(-length / 2.0, length / 2.0, n)
        if direction is None or direction < 0:   # AUTOMATIC: line normal
            direction = 0 if along_y else 1      # X for a y-line, Y for an x-line
        rows = []
        for t in ts:
            pt = Vector3(c.x, c.y + t) if along_y else Vector3(c.x + t, c.y)
            eh = np.atleast_2d(self.sim.get_farfield(self, pt))  # (nfreq, 6)
            ex, ey, ez, hx, hy, hz = eh.T
            if direction == 0:
                rows.append(0.5 * np.real(ey * np.conj(hz) - ez * np.conj(hy)))
            else:
                rows.append(0.5 * np.real(ez * np.conj(hx) - ex * np.conj(hz)))
        rows = np.array(rows)                    # (npts, nfreq)
        if length == 0.0:
            return rows[0].tolist()              # 0d where: flux density at the point
        return np.trapezoid(rows, ts, axis=0).tolist()


class DftLdos(DftObj):
    """meep DftLdos(func, args), args = [freqs, name]."""


class DftForce(DftObj):
    """meep DftForce(func, args): func is the deferred adder (here
    Simulation._add_force, run eagerly), args = [freqs, forceregions, name]."""

    def __init__(self, func, args):
        super().__init__(func, args)
        self.nfreqs = len(self.freqs)
        self.regions = args[1]
        self.num_components = 6

    def force(self):
        return self.sim.get_forces(self)


class DftEnergy(DftObj):
    """meep DftEnergy(func, args): func is the deferred adder (here
    Simulation._add_energy, run eagerly), args = [freqs, energyregions, name]."""

    def __init__(self, func, args):
        super().__init__(func, args)
        self.nfreqs = len(self.freqs)
        self.regions = args[1]
        self.num_components = 12

    def electric(self):
        return self.sim.get_electric_energy(self)

    def magnetic(self):
        return self.sim.get_magnetic_energy(self)

    def total(self):
        return self.sim.get_total_energy(self)


def dft_name(f):
    """Monitor key from a Dft object or a legacy name string."""
    return f.name if isinstance(f, DftObj) else f


# meep FluxData: opaque snapshot of a flux monitor's accumulated DFT fields
# (Simulation.get_flux_data / load_flux_data / load_minus_flux_data).
FluxData = namedtuple("FluxData", ["E", "H"])


# --- step-function calling convention (meep _eval_step_func) ---
def _num_args(func):
    """meep get_num_args: positional-arg count of a step function (bound
    methods and callable class instances drop self)."""
    if inspect.ismethod(func):
        return func.__code__.co_argcount - 1
    if inspect.isfunction(func):
        return func.__code__.co_argcount
    return func.__call__.__func__.__code__.co_argcount - 1


def _eval_step_func(sim, func, todo):
    """meep's step-function protocol: a 1-arg f(sim) fires only on
    todo == "step"; a 2-arg f(sim, todo) (combinators, output wrappers) is
    always called and gates itself."""
    n = _num_args(func)
    if n not in (1, 2):
        raise ValueError(f"step function {getattr(func, '__name__', func)!r} "
                         "requires 1 or 2 arguments")
    if n == 1:
        if todo == "step":
            func(sim)
    else:
        func(sim, todo)


# --- step-function combinators (meep names) ---
class _Combinator:
    """A meep f(sim, todo) step function (so it nests inside to_appended /
    in_volume / other combinators); plan_hooks also recognizes the concrete
    classes at top level for chunk-sized scheduling."""

    def _fire(self, sim, todo):
        for f in self.fns:
            _eval_step_func(sim, f, todo)


class at_beginning(_Combinator):
    def __init__(self, *fns):
        self.fns = fns
        self._done = False

    def __call__(self, sim, todo="step"):
        if not self._done:
            self._fire(sim, todo)
            self._done = True


class at_end(_Combinator):
    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, sim, todo="step"):
        if todo == "finish":
            self._fire(sim, "step")
            self._fire(sim, "finish")


class at_time(_Combinator):
    def __init__(self, t, *fns):
        self.t = float(t)
        self.fns = fns
        self._done = False

    def __call__(self, sim, todo="step"):
        if todo == "finish" or sim.round_time() >= self.t:
            if not self._done or todo == "finish":
                self._fire(sim, todo)
            self._done = self._done or todo == "step"


class at_every(_Combinator):
    def __init__(self, dt, *fns):
        self.every = float(dt)
        self.fns = fns
        self._tlast = 0.0

    def __call__(self, sim, todo="step"):
        t = sim.round_time()
        if todo == "finish" or t >= self._tlast + self.every - 0.5 * sim.grid.dt:
            self._fire(sim, todo)
            self._tlast = t


class after_time(_Combinator):
    def __init__(self, t, *fns):
        self.t = float(t)
        self.fns = fns

    def __call__(self, sim, todo="step"):
        if todo == "finish" or sim.round_time() >= self.t:
            self._fire(sim, todo)


class before_time(_Combinator):
    def __init__(self, t, *fns):
        self.t = float(t)
        self.fns = fns

    def __call__(self, sim, todo="step"):
        if todo == "finish" or sim.round_time() < self.t:
            self._fire(sim, todo)


class after_sources(_Combinator):
    def __init__(self, *fns):
        self.fns = fns

    def __call__(self, sim, todo="step"):
        if sim.round_time() >= sim._source_cutoff():
            self._fire(sim, todo)


class during_sources(_Combinator):
    def __init__(self, *fns):
        self.fns = fns
        self._finished = False

    def __call__(self, sim, todo="step"):
        if sim.round_time() < sim._source_cutoff():
            self._fire(sim, "step")
        elif not self._finished:
            self._fire(sim, "finish")
            self._finished = True


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
        self.Q_thresh = 50.0        # meep default: drop |Q| < 50 junk modes
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
                for m in ms if abs(m.Q) >= self.Q_thresh]
        return self._modes


class PadeDFT:
    """meep mp.PadeDFT(c, vol, center, size, m, n, m_frac, n_frac,
    sampling_interval, start_time, stop_time): pass to run() like a step
    function. Records the field at the region center every step; ``.dft``
    after the run is the Pade approximant (scipy.interpolate.pade) of the
    sampled series -- an analytic rational frequency response R(f).

    meep semantics kept: the series is sliced
    data[start_time : stop_time : sampling_interval] (start/stop in dt
    steps); m defaults to len(samples)*m_frac, n to len(samples)-m-1 (or
    len(samples)*n_frac); R is evaluated at z = exp(2j pi f *
    sampling_interval * dt).
    """
    # ponytail: point sample at the region center (meep DFTs the whole vol);
    # extend to array sampling when a consumer needs spatial maps

    def __init__(self, c=None, vol=None, center=None, size=None, m=None,
                 n=None, m_frac=0.5, n_frac=None, sampling_interval=1,
                 start_time=0, stop_time=None):
        self.c = c
        self.vol = vol
        self.center = center
        self.size = size
        self.m = m
        self.n = n
        self.m_frac = m_frac
        self.n_frac = n_frac
        self.sampling_interval = int(sampling_interval)
        self.start_time = start_time
        self.stop_time = stop_time
        self.data = []
        self.polys = None
        self._dt = None
        self._dft = None

    def __call__(self, sim):
        self._dt = sim.grid.dt
        pt = (self.vol.center if self.vol is not None else self.center) or Vector3()
        self.data.append(sim.get_field_point(self.c, pt))
        self._dft = None

    @property
    def dft(self):
        # ponytail: scipy.interpolate.pade is what meep calls but is slated
        # for removal in scipy 1.20; inline the ~10-line Toeplitz solve then
        from scipy.interpolate import pade
        if self._dft is None:
            samples = np.asarray(
                self.data[self.start_time:self.stop_time:self.sampling_interval])
            m = self.m or int(len(samples) * self.m_frac)
            n = self.n or (int(len(samples) * self.n_frac) if self.n_frac
                           else len(samples) - m - 1)
            P, Q = pade(samples, m, n)
            self.polys = {"P": P, "Q": Q}
            s = self.sampling_interval * self._dt
            self._dft = lambda f: P(np.exp(2j * np.pi * f * s)) / Q(np.exp(2j * np.pi * f * s))
        return self._dft

    def spectrum(self, freqs):
        """|R(f)| over ``freqs`` (peaks sit at the resonances)."""
        d = self.dft
        return np.abs([d(f) for f in freqs])


# --- hook machinery for Simulation._run_until ---
def plan_hooks(step_funcs, cutoff):
    """Normalize meep step functions.

    Returns (begin, per_step, scheduled): ``begin`` fires once before the
    loop; ``per_step`` is (fn, t_start, t_stop) fired after every step whose
    post-step time t satisfies t >= t_start and t - dt < t_stop; ``scheduled``
    is [next_t, period_or_None, fns] fired when t crosses next_t. at_end
    contributes nothing here: the run loop's final "finish" pass over the
    original step_funcs fires it (meep's run-end protocol).
    """
    begin, per_step, scheduled = [], [], []
    for f in step_funcs:
        if isinstance(f, at_beginning):
            begin.append(f)
        elif isinstance(f, at_end):
            pass                      # fired by the run-end "finish" pass
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
    return begin, per_step, scheduled


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
            _eval_step_func(sim, g, "step")


def fire_scheduled(sim, scheduled, t):
    for sch in scheduled:
        while sch[0] is not None and t >= sch[0] - 1e-9:
            for g in sch[2]:
                _eval_step_func(sim, g, "step")
            sch[0] = sch[0] + sch[1] if sch[1] else None
    scheduled[:] = [s for s in scheduled if s[0] is not None]
