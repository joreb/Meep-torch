# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Lowercase C++-binding classes (SWIG surface): sources, susceptibilities,
and the DFT monitor group.

The ``src_time`` family are genuine engine adapters over the time profiles in
:mod:`meep_torch.core.sources` (the math the stepper injects); the Lorentzian
susceptibilities carry genuine chi1 dispersion; ``dft_flux``/``dft_near2far``
are views over the shim ``DftObj`` handles a live Simulation returns (data
access raises when constructed standalone); the rest are parameter carriers
whose C++-chunk/fields-bound methods raise honestly.
"""
import cmath
import copy
import math

from ...core.sources import (ContinuousSource as _CoreCW,
                            CustomSource as _CoreCustom, GaussianPulse)
from ..run_control import DftObj

_INF = 1.0e20


def _not_ported(name, msg):
    def method(self, *args, **kwargs):
        raise NotImplementedError(f"{name}: {msg}")
    method.__name__ = name
    return method


class src_time:
    """Base source time-dependence (meep src_time). Subclasses provide
    dipole(t); current/update/last_time_max derive from it, meep-style."""

    def __init__(self):
        self._freq = 0.0
        self._cur_time = None
        self._cur_dipole = 0.0

    def _key(self):
        return (self._freq,)

    def dipole(self, time=None):
        """dipole(t); with no argument, the value cached by update()."""
        if time is None:
            return self._cur_dipole
        return self._dipole(time)

    _dipole = _not_ported("dipole", "src_time is abstract; use a subclass")

    def current(self, time, dt):
        """meep: (dipole(t + dt) - dipole(t)) / dt."""
        return (self.dipole(time + dt) - self.dipole(time)) / dt

    def update(self, time, dt):
        """Cache dipole(time) for the no-argument dipole() accessor."""
        self._cur_time = time
        self._cur_dipole = self.dipole(time)

    def frequency(self):
        return self._freq

    def set_frequency(self, f):
        self._freq = float(f)

    def get_fwidth(self):
        return 0.0

    def set_fwidth(self, fw):
        """No-op on the base class (meep base virtual is a no-op too)."""

    last_time = _not_ported("last_time", "src_time is abstract; use a subclass")

    def last_time_max(self, after=0.0):
        return max(self.last_time(), after)

    def is_equal(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def clone(self):
        return copy.copy(self)

    add_to = _not_ported(
        "src_time.add_to", "C++ linked-list plumbing for field chunks")


class gaussian_src_time(src_time):
    """meep gaussian_src_time(f, fwidth, s=5.0) or (f, width, start, end):
    the engine's GaussianPulse (real cos carrier under a Gaussian envelope,
    zero outside the window -- the profile the stepper injects)."""

    def __init__(self, f, fwidth, *args):
        super().__init__()
        self._freq = float(f)
        if len(args) == 2:            # (f, width, start_time, end_time)
            width, (st, et) = float(fwidth), (float(args[0]), float(args[1]))
            self._width = width
            self._peak = 0.5 * (st + et)
            self._cut = 0.5 * (et - st)
        else:                         # (f, fwidth, s=5.0)
            s = float(args[0]) if args else 5.0
            self._width = 1.0 / float(fwidth)
            self._peak = self._width * s
            self._cut = self._width * s
        self._rebuild()

    def _rebuild(self):
        # GaussianPulse peaks at t0 = cutoff/fwidth; pick cutoff = peak_time
        self._pulse = GaussianPulse(self._freq, 1.0 / self._width,
                                    cutoff=self._peak / self._width)

    def _key(self):
        return (self._freq, self._width, self._peak, self._cut)

    def _dipole(self, time):
        if abs(time - self._peak) > self._cut:
            return 0.0
        return self._pulse.amplitude(time)

    def last_time(self):
        return self._peak + self._cut

    def set_frequency(self, f):
        self._freq = float(f)
        self._rebuild()

    def get_fwidth(self):
        return 1.0 / self._width

    def set_fwidth(self, fw):
        self._width = 1.0 / float(fw)
        self._rebuild()

    def fourier_transform(self, f):
        """Analytic FT (exp(+i*omega*t) convention) of the complex
        profile exp(-i*omega0*tt - tt^2/(2 w^2)), tt = t - peak_time."""
        w = self._width
        omega, omega0 = 2.0 * math.pi * f, 2.0 * math.pi * self._freq
        return (w * math.sqrt(2.0 * math.pi) * cmath.exp(1j * omega * self._peak)
                * math.exp(-0.5 * (w * (omega - omega0)) ** 2))


class continuous_src_time(src_time):
    """meep continuous_src_time(f, width=0, start_time=0, end_time=inf,
    slowness=3.0): the engine's ContinuousSource (sin carrier, 1-cos
    turn-on) shifted to start_time and windowed to end_time."""

    def __init__(self, f, width=0.0, start_time=0.0, end_time=_INF,
                 slowness=3.0):
        super().__init__()
        self._freq = float(f)
        self._width = float(width)
        self._start = float(start_time)
        self._end = float(end_time)
        self._slow = float(slowness)
        self._cw = _CoreCW(self._freq, width=(self._width or None),
                           slowness=self._slow)

    def _key(self):
        return (self._freq, self._width, self._start, self._end, self._slow)

    def _dipole(self, time):
        if time < self._start or time > self._end:
            return 0.0
        return self._cw.amplitude(time - self._start)

    def last_time(self):
        return self._end


class custom_src_time(src_time):
    """meep custom_src_time(func, start_time=-inf, end_time=inf,
    center_frequency=0, fwidth=0): dipole(t) = func(t) inside the window
    (the engine's CustomSource)."""

    def __init__(self, func, start_time=-_INF, end_time=_INF,
                 center_frequency=0.0, fwidth=0.0):
        super().__init__()
        self._fn = _CoreCustom(func)
        self._start = float(start_time)
        self._end = float(end_time)
        self._freq = float(center_frequency)
        self._fwidth = float(fwidth)

    def _key(self):
        return (self._fn.fn, self._start, self._end, self._freq, self._fwidth)

    def _dipole(self, time):
        if time < self._start or time > self._end:
            return 0.0
        return self._fn.amplitude(time)

    def last_time(self):
        return self._end

    def get_fwidth(self):
        return self._fwidth

    def set_fwidth(self, fw):
        self._fwidth = float(fw)


class custom_py_src_time(custom_src_time, src_time):
    """Python-callable variant; identical to custom_src_time here."""


class gaussianbeam:
    """meep gaussianbeam(x0, kdir, w0, freq, eps, mu, E0): parameter
    carrier; get_fields (Green's-function beam evaluation) is not ported."""

    def __init__(self, x0, kdir, w0, freq, eps, mu, E0):
        self.x0, self.kdir, self.w0 = x0, kdir, float(w0)
        self.freq, self.eps, self.mu = float(freq), float(eps), float(mu)
        self.E0 = E0

    def get_E0(self):
        return self.E0

    get_fields = _not_ported(
        "gaussianbeam.get_fields",
        "needs the complex vector Green's-function evaluation "
        "(meep/src/gaussianbeam.cpp); use GaussianBeamSource for the "
        "engine's amplitude-profile beam launch")


class diffractedplanewave:
    """meep diffractedplanewave(g, axis, s, p): diffraction order g (3 ints),
    polarization axis, and s/p amplitudes. Pure data."""

    def __init__(self, g, axis, s, p):
        self._g, self._axis, self._s, self._p = g, axis, s, p

    for _n in ("g", "axis", "s", "p"):
        locals()[f"get_{_n}"] = (
            lambda self, _a=f"_{_n}": getattr(self, _a))
    del _n


_CHUNK_MSG = ("per-chunk C++ polarization-update machinery; the engine's "
              "dispersive (ADE) update is tensorized in meep_torch.core")


class susceptibility:
    """meep susceptibility base: id bookkeeping and chi1 dispersion are
    genuine; the per-chunk P-field update machinery raises."""

    def __init__(self):
        self._id = 0

    def get_id(self):
        return self._id

    def set_id(self, i):
        self._id = int(i)

    def chi1(self, freq, sigma=1.0):
        """Base susceptibility adds no dispersion."""
        return 0.0j

    def has_nonlinearities(self):
        return False

    def needs_W_prev(self):
        return False

    def needs_W_notowned(self, *args):
        return False

    needs_P = _not_ported(
        "needs_P", "queries per-chunk field ownership; the engine has no "
        "chunked fields")

    def _params(self):
        return {}

    def get_num_params(self):
        return len(self._params())

    def dump_params(self, *args):
        """The constructor-parameter dict (meep serializes these to HDF5)."""
        return self._params()

    def clone(self):
        return copy.copy(self)

    for _n in ("new_internal_data", "init_internal_data", "copy_internal_data",
               "delete_internal_data", "update_P", "subtract_P",
               "cinternal_notowned_ptr", "internal_notowned_ptr",
               "num_cinternal_notowned_needed", "num_internal_notowned_needed"):
        locals()[_n] = _not_ported(_n, _CHUNK_MSG)
    del _n


class lorentzian_susceptibility(susceptibility):
    """meep lorentzian_susceptibility(omega_0, gamma, no_omega_0_denominator):
    genuine chi1; omega_0/gamma units follow the constructor arguments."""

    def __init__(self, omega_0, gamma, no_omega_0_denominator=False):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.gamma = float(gamma)
        self.no_omega_0_denominator = bool(no_omega_0_denominator)

    def _params(self):
        return {"omega_0": self.omega_0, "gamma": self.gamma,
                "no_omega_0_denominator": self.no_omega_0_denominator}

    def chi1(self, freq, sigma=1.0):
        """sigma*omega_0^2 / (omega_0^2 - freq^2 - i*freq*gamma); the Drude
        form (no_omega_0_denominator) drops omega_0^2 in the denominator."""
        w02 = self.omega_0 * self.omega_0
        denom = -freq * freq - 1j * freq * self.gamma
        if not self.no_omega_0_denominator:
            denom += w02
        return sigma * w02 / denom


class noisy_lorentzian_susceptibility(lorentzian_susceptibility):
    """meep noisy_lorentzian_susceptibility(noise_amp, omega_0, gamma,
    no_omega_0_denominator): Lorentzian + white-noise polarization drive."""

    def __init__(self, noise_amp, omega_0, gamma, no_omega_0_denominator=False):
        super().__init__(omega_0, gamma, no_omega_0_denominator)
        self.noise_amp = float(noise_amp)

    def _params(self):
        p = super()._params()
        p["noise_amp"] = self.noise_amp
        return p


class gyrotropic_susceptibility(susceptibility):
    """meep gyrotropic_susceptibility(bias, omega_0, gamma, alpha, model):
    parameter carrier (the gyrotropic P update is chunk machinery)."""

    def __init__(self, bias=None, omega_0=0.0, gamma=0.0, alpha=0.0, model=0):
        super().__init__()
        self.bias = bias
        self.omega_0 = float(omega_0)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.model = model

    def _params(self):
        return {"bias": self.bias, "omega_0": self.omega_0,
                "gamma": self.gamma, "alpha": self.alpha, "model": self.model}


class multilevel_susceptibility(susceptibility):
    """meep multilevel_susceptibility(theta, L transitions, ...): parameter
    carrier; the rate-equation update is chunk machinery."""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def _params(self):
        return {"args": self.args}


class _dft_view:
    """View over a shim DftObj from a live Simulation monitor; data
    methods raise when constructed standalone."""

    def __init__(self, dft=None):
        self._dft = dft if isinstance(dft, DftObj) else None

    def _backed(self):
        if self._dft is None:
            raise RuntimeError(
                f"{type(self).__name__} has no live Simulation backing; pass "
                "the object returned by the matching Simulation.add_* method")
        return self._dft

    remove = _not_ported(
        "remove", "monitors are baked into the run plan and cannot be removed")
    scale_dfts = _not_ported(
        "scale_dfts", "the engine's DFT accumulators are not exposed for "
        "in-place scaling")
    save_hdf5 = _not_ported("save_hdf5", "DFT HDF5 persistence is not supported")
    load_hdf5 = _not_ported("load_hdf5", "DFT HDF5 persistence is not supported")


class dft_flux(_dft_view):
    """View over a DftFlux handle (Simulation.add_flux)."""

    def flux(self):
        d = self._backed()
        return list(d.sim.get_fluxes(d))

    complexflux = _not_ported(
        "complexflux", "needs the complex cross-spectral accumulators; the "
        "engine records real Poynting flux (use flux())")

    def invalidate_eigenmode_cache(self):
        """No-op: the engine keeps no eigenmode cache on flux monitors."""


class dft_fields(_dft_view):
    """View over a DftFields handle (Simulation.add_dft_fields)."""

    fourier_sourcedata = _not_ported(
        "fourier_sourcedata", "adjoint source construction from DFT fields "
        "is not ported; the autodiff path differentiates the run directly")


class dft_force(_dft_view):
    """View over a DftForce handle (Simulation.add_force)."""

    def force(self):
        d = self._backed()
        return list(d.sim.get_forces(d))


class dft_energy(_dft_view):
    """View over a DftEnergy handle (Simulation.add_energy)."""

    def electric(self):
        d = self._backed()
        return list(d.sim.get_electric_energy(d))

    def magnetic(self):
        d = self._backed()
        return list(d.sim.get_magnetic_energy(d))

    def total(self):
        d = self._backed()
        return list(d.sim.get_total_energy(d))


class dft_near2far(_dft_view):
    """View over a DftNear2Far handle (Simulation.add_near2far)."""

    def farfield(self, x):
        d = self._backed()
        return d.sim.get_farfield(d, x)

    farfield_lowlevel = _not_ported(
        "farfield_lowlevel", "returns raw C++ arrays; use farfield()")
    flux = _not_ported(
        "flux", "near2far far-field flux integration is not ported")
    get_farfields_array = _not_ported(
        "get_farfields_array", "not ported; evaluate farfield() per point")
    near_sourcedata = _not_ported(
        "near_sourcedata", "adjoint near-field sources are not ported")
    save_farfields = _not_ported(
        "save_farfields", "far-field HDF5 persistence is not supported")


class sourcedata:
    """meep sourcedata: adjoint source point record. Attribute carrier."""


class near_src_data(list):
    """meep near_src_data: std::vector<sourcedata> -- a list with the
    SWIG vector API."""

    push_back = list.append
    pop_back = list.pop
    size = capacity = list.__len__
    begin = iterator = list.__iter__
    rbegin = list.__reversed__

    def empty(self):
        return not self

    def reserve(self, n):
        """No-op: python lists manage capacity themselves."""

    def resize(self, n, value=None):
        if n < len(self):
            del self[n:]
        else:
            self.extend([value] * (n - len(self)))

    def assign(self, n, value):
        self[:] = [value] * n

    def front(self):
        return self[0]

    def back(self):
        return self[-1]

    def erase(self, i, j=None):
        del self[i:(j if j is not None else i + 1)]

    def swap(self, other):
        self[:], other[:] = list(other), list(self)

    def end(self):
        return iter(())

    rend = end

    def get_allocator(self):
        return None


class dft_data:
    """meep dft_data(freqs, components, volumes): plain data triple the
    structure/fields dump-load machinery passes around. Pure data."""

    def __init__(self, freqs, components, volumes):
        self.freqs = list(freqs)
        self.components = components
        self.volumes = volumes


class eigenmode_data:
    """meep eigenmode_data: opaque MPB eigenmode handle. Attribute carrier."""


class pol:
    """meep pol: per-chunk polarization linked-list node. Attribute carrier."""


class polarization_state:
    """meep polarization_state: (susceptibility, data) attribute carrier."""
