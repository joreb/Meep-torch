# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep module-level functions (parity tier). Three kinds live here:

* pure/portable helpers ported from meep/python/{simulation,geom,source}.py
  and meep/src/vec.hpp (enum helpers work on meep-torch's string components
  as well as meep's integer enum values);
* single-process implementations of meep's parallel API -- meep-torch runs
  one process by design, so am_master()=True, count_processors()=1, the
  reductions are identities and the barriers no-ops (these are the exact
  semantics of a meep built without MPI, not fakes);
* engine-backed accessors over the Dft handles / Simulation state.

Functions whose meep counterpart needs an optional binary dependency
(libGDSII, harminv-adjacent C plumbing) raise with a clear message, matching
meep's own behavior when built without that dependency.
"""
import math
import os
import random
import shutil
import signal
import tempfile
import time
import warnings

import numpy as np

from .material import Medium
from .simulation import P, R, X, Y, Z
from .vec import Vector3
from .cpp_bindings_geom import D1, D2, D3, Dcyl, vec, veccyl

_T0 = time.time()                      # module import ~ meep program start
NO_DIRECTION = 5
NO_COMPONENT = 22
E_stuff, H_stuff, D_stuff, B_stuff = 0, 1, 2, 3

# ---------------------------------------------------------------- verbosity
# single source of truth: the full verbosity_mgr port (a ``meep`` flag
# property exists there, so ``verbosity.meep`` reads/writes keep working)
from .verbosity_mgr import Verbosity, verbosity


def quiet(quietval=True):
    """Deprecated meep helper: quiet(True) silences diagnostics."""
    verbosity(int(not quietval))
    warnings.warn("quiet has been deprecated; use the Verbosity class instead",
                  RuntimeWarning)


# --------------------------------------------------- printing (+ callbacks)
_printf_callback = [None]
_printf_stderr_callback = [None]


def set_meep_printf_callback(func):
    _printf_callback[0] = func


def set_meep_printf_stderr_callback(func):
    _printf_stderr_callback[0] = func


def set_ctl_printf_callback(callback):
    _printf_callback[0] = callback


def set_mpb_printf_callback(callback):
    _printf_callback[0] = callback


def _cfmt(args):
    return (args[0] % tuple(args[1:])) if len(args) > 1 else str(args[0])


def master_printf(*args):
    s = _cfmt(args)
    if _printf_callback[0]:
        _printf_callback[0](s)
    else:
        print(s, end="", flush=True)


def master_printf_stderr(*args):
    import sys
    s = _cfmt(args)
    if _printf_stderr_callback[0]:
        _printf_stderr_callback[0](s)
    else:
        print(s, end="", file=sys.stderr, flush=True)


def debug_printf(*args):
    if verbosity.meep >= 3:
        master_printf(*args)


def master_fopen(name, mode):
    return open(name, mode)


def master_fclose(f):
    f.close()


def master_fprintf(f, *args):
    f.write(_cfmt(args))


def abort(*args):
    raise RuntimeError("meep: " + (_cfmt(args) if args else "abort"))


# ------------------------------------------- single-process parallel layer
def am_master():
    return True


def am_really_master():
    return True


def my_rank():
    return 0


def my_global_rank():
    return 0


def count_processors():
    return 1


def with_mpi():
    return False


def get_num_groups():
    return 1


def get_group_masters():
    return [0]


def divide_parallel_processes(numgroups):
    if numgroups < 1:
        raise ValueError("numgroups must be >= 1")
    return 0                            # this (only) process is in group 0


def end_divide_parallel():
    pass


def merge_subgroup_data(data):
    return data                         # one group: nothing to merge


def begin_global_communications():
    pass


def end_global_communications():
    pass


def all_wait():
    pass


def begin_critical_section(tag):
    pass


def end_critical_section(tag):
    pass


def send(_from=0, to=0, data=None, size=None):
    pass                                # rank 0 -> rank 0 transfer is a no-op


def broadcast(*args):
    return args[-1] if args else None   # broadcast(from, data) -> data


def sum_to_all(x):
    return x


def sum_to_master(x):
    return x


def partial_sum_to_all(x):
    return x


def max_to_all(x):
    return x


def max_to_master(x):
    return x


def min_to_all(x):
    return x


def or_to_all(x):
    return x


def and_to_all(x):
    return x


# -------------------------------- component / direction / dimension helpers
# meep C enum order (vec.hpp); meep-torch components are these name strings.
_COMPONENTS = ["Ex", "Ey", "Er", "Ep", "Ez", "Hx", "Hy", "Hr", "Hp", "Hz",
               "Dx", "Dy", "Dr", "Dp", "Dz", "Bx", "By", "Br", "Bp", "Bz",
               "Dielectric", "Permeability", "NO_COMPONENT"]
_DERIVED = ["Sx", "Sy", "Sr", "Sp", "Sz",
            "EnergyDensity", "D_EnergyDensity", "H_EnergyDensity"]  # 100..107
_NAMES = {"Dielectric": "eps", "Permeability": "mu", "NO_COMPONENT": "none",
          "EnergyDensity": "energy", "D_EnergyDensity": "denergy",
          "H_EnergyDensity": "henergy"}


def _cidx(c):
    """meep enum value of a component given as string or int."""
    if isinstance(c, str):
        if c in _COMPONENTS:
            return _COMPONENTS.index(c)
        if c in _DERIVED:
            return 100 + _DERIVED.index(c)
        raise ValueError(f"unknown component {c!r}")
    return int(c)


def _cstr(i):
    return _DERIVED[i - 100] if i >= 100 else _COMPONENTS[i]


def is_derived(c):
    return _cidx(c) >= 100


def is_electric(c):
    return _cidx(c) < 5


def is_magnetic(c):
    return 5 <= _cidx(c) < 10


def is_D(c):
    return 10 <= _cidx(c) < 15


def is_B(c):
    return 15 <= _cidx(c) < 20


def is_E_or_D(c):
    return is_electric(c) or is_D(c)


def is_H_or_B(c):
    return is_magnetic(c) or is_B(c)


def is_poynting(c):
    return 100 <= _cidx(c) < 105


def is_energydensity(c):
    return _cidx(c) >= 105


def is_tm(c):
    """2D TM (Ez) polarization components: Ez/Dz plus in-plane H/B."""
    return _cstr(_cidx(c)) in ("Ez", "Dz", "Hx", "Hy", "Hr", "Hp",
                               "Bx", "By", "Br", "Bp")


def meep_type(c):
    i = _cidx(c)
    if i >= 20:
        raise ValueError(f"invalid field in type: {c!r}")
    return (E_stuff, H_stuff, D_stuff, B_stuff)[i // 5]


def first_field_component(ft):
    return _COMPONENTS[ft * 5]          # Ex / Hx / Dx / Bx


def component_name(c):
    s = _cstr(_cidx(c))
    return _NAMES.get(s, s.lower())


def component_index(c):
    """0/1/2 for the x/y/z (or r/phi = 0/1) part; -1 for scalars."""
    i = _cidx(c)
    if i >= 100:
        i = (i - 100) if i < 105 else 20
    if i >= 20:
        return -1
    return {0: 0, 1: 1, 2: 0, 3: 1, 4: 2}[i % 5]


def component_direction(c):
    i = _cidx(c)
    if i >= 100:
        i = (i - 100) if i < 105 else 20
    if i >= 20:
        return NO_DIRECTION
    return (X, Y, R, P, Z)[i % 5]


def direction_component(c, d):
    """Component of c's field type pointing along direction d."""
    i = _cidx(c)
    off = {X: 0, Y: 1, R: 2, P: 3, Z: 4}
    if i >= 105 and d == NO_DIRECTION:
        return _cstr(i)                 # energy densities are directionless
    if i >= 100:
        return _DERIVED[off[d]]         # poynting block
    if i >= 20:
        if d == NO_DIRECTION:
            return _cstr(i)
        raise ValueError(f"unknown field component {c!r}")
    return _COMPONENTS[(i // 5) * 5 + off[d]]


def field_type_component(ft, c):
    return direction_component(first_field_component(ft), component_direction(c))


def direction_name(d):
    return ("x", "y", "z", "r", "phi", "no_direction")[d]


def dimension_name(dim):
    return ("1D", "2D", "3D", "Cylindrical")[dim]


def number_of_directions(dim):
    return int(dim + 1 - 2 * (dim == Dcyl))


def start_at_direction(dim):
    return Z if dim in (D1, Dcyl) else X


def stop_at_direction(dim):
    return int(dim + 1 + 2 * (dim == D1))


def has_direction(dim, d):
    return start_at_direction(dim) <= d < stop_at_direction(dim) or \
        (dim == Dcyl and d in (R, Z))


def has_field_direction(dim, d):
    return has_direction(dim, d) or (dim == D2 and d == Z) or \
        (dim == Dcyl and d == P)


def coordinate_mismatch(dim, d):
    """True if d is polar while dim is cartesian, or vice versa."""
    if not isinstance(d, int) or d > NO_DIRECTION:
        d = component_direction(d)
    return d != NO_DIRECTION and (
        (D1 <= dim <= D3 and d not in (X, Y, Z)) or
        (dim == Dcyl and d not in (R, P, Z)))


def cycle_direction(dim, d, shift):
    start = 2 if dim == Dcyl else 0
    return (d - start + shift + 99) % 3 + start


def cycle_component(dim, c, shift):
    return direction_component(c, cycle_direction(dim, component_direction(c), shift))


def flip(s):
    """Flip a boundary side (meep Low <-> High)."""
    from .material import High, Low
    return High if s == Low else Low


def pml(thickness=1.0, direction=None, side=None, strength=1.0):
    """C-binding pml(): a PML boundary_region band."""
    from .cpp_bindings_core import boundary_region
    return boundary_region(boundary_region.PML, thickness, direction, side,
                           strength)


# ------------------------------------------------------------- vec helpers
def make_vector3(x=0.0, y=0.0, z=0.0):
    return Vector3(float(x), float(y), float(z))


def vec_to_vector3(pt):
    return Vector3(pt.x(), pt.y(), pt.z())


def vector3_to_vec(v3):
    return vec(v3.x, v3.y, v3.z)


def py_v3_to_vec(dims, iterable, is_cylindrical=False):
    v3 = Vector3(*iterable)
    if dims == 1:
        return vec(v3.z)
    if dims == 2:
        if is_cylindrical:
            return veccyl(v3.x, v3.z)
        v = vec(v3.x, v3.y)
        v.set_direction(Z, v3.z)        # for special_kz handling
        return v
    if dims == 3:
        return vec(v3.x, v3.y, v3.z)
    raise ValueError(f"Invalid dimensions in Volume: {dims}")


def vec_abs(pt):
    return math.sqrt(sum(pt.in_direction(d) ** 2 for d in (X, Y, Z, R, P)))


def vec_max(a, b):
    r = vec._make(a.dim, a.t)
    r.t = [max(x, y) for x, y in zip(a.t, b.t)]
    return r


def vec_min(a, b):
    r = vec._make(a.dim, a.t)
    r.t = [min(x, y) for x, y in zip(a.t, b.t)]
    return r


def vec_dim(di):
    return vec._make(di)                # zero vec of dimension di


def vec_dim_val(di, val):
    r = vec._make(di)
    r.t = [float(val)] * 5
    return r


# ------------------------------------------------------- math / rng / time
def BesselJ(m, kr):
    from scipy.special import jv
    return float(jv(m, kr))


def cdouble(*args):
    if len(args) == 1:
        return complex(args[0])
    return complex(args[0], args[1])


def complexarray(re, im):
    z = im * 1j
    z += re
    return z


def linspace(freq_min, freq_max, Nfreq):
    return np.linspace(freq_min, freq_max, Nfreq).tolist()


def interpolate(n, nums):
    """meep interpolate: insert n evenly spaced values between every pair of
    consecutive entries (numbers or Vector3s)."""
    res = []
    if isinstance(nums[0], Vector3):
        for low, high in zip(nums, nums[1:]):
            xs = np.linspace(low.x, high.x, n + 1, endpoint=False)
            ys = np.linspace(low.y, high.y, n + 1, endpoint=False)
            zs = np.linspace(low.z, high.z, n + 1, endpoint=False)
            res.extend(Vector3(x, y, z) for x, y, z in zip(xs, ys, zs))
    else:
        for low, high in zip(nums, nums[1:]):
            res.extend(np.linspace(low, high, n + 1, endpoint=False).tolist())
    return res + [nums[-1]]


def _mirrorindex(i, n):
    return 2 * n - 1 - i if i >= n else (-1 - i if i < 0 else i)


def linear_interpolate(rx, ry, rz, data, nx, ny, nz, stride=1):
    """Trilinear interpolation on a row-major grid, coords in [0,1] with
    mirror boundaries (port of meep::linear_interpolate)."""
    def axis(r, n):
        r = -r if r < 0.0 else (1.0 - r if r > 1.0 else r)
        i1 = _mirrorindex(int(r * n), n)
        d = r * n - i1 - 0.5
        i2 = _mirrorindex(i1 + 1 if d >= 0.0 else i1 - 1, n)
        return i1, i2, abs(d)

    x1, x2, dx = axis(rx, nx)
    y1, y2, dy = axis(ry, ny)
    z1, z2, dz = axis(rz, nz)

    def D(x, y, z):
        return data[((x * ny + y) * nz + z) * stride]

    return (((D(x1, y1, z1) * (1 - dx) + D(x2, y1, z1) * dx) * (1 - dy) +
             (D(x1, y2, z1) * (1 - dx) + D(x2, y2, z1) * dx) * dy) * (1 - dz) +
            ((D(x1, y1, z2) * (1 - dx) + D(x2, y1, z2) * dx) * (1 - dy) +
             (D(x1, y2, z2) * (1 - dx) + D(x2, y2, z2) * dx) * dy) * dz)


def gaussian_random(mean, stddev):
    return random.gauss(mean, stddev)


def uniform_random(a, b):
    return random.uniform(a, b)


def random_int(a, b):
    return random.randint(int(a), int(b))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)


def restore_random_seed():
    random.seed()
    np.random.seed()


def wall_time():
    return time.time()


def report_elapsed_time():
    dt = time.time() - _T0
    master_printf("\nElapsed run time = %0.4f s\n", dt)
    return dt


def is_single_precision():
    import torch
    return torch.get_default_dtype() == torch.float32


def get_realnum_size():
    return 4 if is_single_precision() else 8


def set_zero_subnormals(iszero):
    import torch
    torch.set_flush_denormal(bool(iszero))


def pml_quadratic_profile(u, *_):
    return u * u


def src_times_equal(t1, t2):
    return t1 is t2 or t1 == t2


# ------------------------------------------------------------ geom helpers
def check_nonnegative(prop, val):
    if val < 0:
        raise ValueError(f"{prop} cannot be negative. Got {val}")
    return val


def check_positive(prop, val):
    if val <= 0:
        raise ValueError(f"{prop} must be positive. Got {val}")
    return val


def make_dielectric(epsilon):
    return Medium(epsilon=epsilon)


_default_material = [None]
_dimensions = [2]


def set_default_material(m):
    _default_material[0] = m


def unset_default_material():
    _default_material[0] = None


def set_dimensions(dims):
    old, _dimensions[0] = _dimensions[0], int(dims)
    return old


def get_center_and_size(vol):
    """(center, size) of a Volume / meep::volume-like object."""
    if hasattr(vol, "center") and hasattr(vol, "size"):
        return vol.center, vol.size
    rmin, rmax = vol.get_min_corner(), vol.get_max_corner()
    lo = Vector3(rmin.x(), rmin.y(), rmin.z())
    hi = Vector3(rmax.x(), rmax.y(), rmax.z())
    return 0.5 * (lo + hi), hi - lo


def geometric_object_duplicates(shift_vector, min_multiple, max_multiple, go):
    shift_vector = Vector3(*shift_vector)
    return [go.shift(shift_vector.scale(m))
            for m in range(max_multiple, min_multiple - 1, -1)]


def geometric_objects_duplicates(shift_vector, min_multiple, max_multiple, go_list):
    dups = []
    for go in go_list:
        dups += geometric_object_duplicates(shift_vector, min_multiple,
                                            max_multiple, go)
    return dups


def geometric_objects_lattice_duplicates(lat, go_list, *usize):
    from .geometry import cartesian_to_lattice
    u1 = usize[0] if usize else 1
    u2 = usize[1] if len(usize) >= 2 else 1
    u3 = usize[2] if len(usize) >= 3 else 1
    s = lat.size

    def lat_to_lattice(v):
        return cartesian_to_lattice(lat.basis * v, lat)

    b1 = lat_to_lattice(Vector3(u1))
    b2 = lat_to_lattice(Vector3(0, u2, 0))
    b3 = lat_to_lattice(Vector3(0, 0, u3))
    n1 = math.ceil((s.x or 1e-20) / u1)
    n2 = math.ceil((s.y or 1e-20) / u2)
    n3 = math.ceil((s.z or 1e-20) / u3)
    d3 = geometric_objects_duplicates(b3, -math.floor((n3 - 1) / 2),
                                      math.ceil((n3 - 1) / 2), go_list)
    d2 = geometric_objects_duplicates(b2, -math.floor((n2 - 1) / 2),
                                      math.ceil((n2 - 1) / 2), d3)
    return geometric_objects_duplicates(b1, -math.floor((n1 - 1) / 2),
                                        math.ceil((n1 - 1) / 2), d2)


def display_geometric_object_info(indentby, o):
    pad = " " * indentby
    master_printf("%s%s, center = (%g,%g,%g)\n", pad, type(o).__name__,
                  o.center.x, o.center.y, o.center.z)


def is_point_in_object(p, o):
    """Point-membership test for the axis-aligned solids the raster supports."""
    from .geometry import Block, Cylinder, Ellipsoid, Sphere
    q = Vector3(*p) - o.center
    if isinstance(o, Sphere):
        return q.norm() <= o.radius
    if isinstance(o, Cylinder):
        a = o.axis.unit()
        h = q.dot(a)
        rad = (q - a * h).norm()
        return abs(h) <= o.height / 2.0 and rad <= o.radius
    if isinstance(o, Ellipsoid):
        return sum(((c / (s / 2.0)) ** 2 if s else 0.0)
                   for c, s in zip((q.x, q.y, q.z),
                                   (o.size.x, o.size.y, o.size.z))) <= 1.0
    if isinstance(o, Block):
        return all(math.isinf(s) or abs(q.dot(e.unit())) <= s / 2.0
                   for s, e in zip((o.size.x, o.size.y, o.size.z),
                                   (o.e1, o.e2, o.e3)))
    raise NotImplementedError(f"is_point_in_object: unsupported {type(o).__name__}")


def is_point_in_periodic_object(p, o):
    return is_point_in_object(p, o)     # no lattice periodicity applied here


def memoize(f):
    cache = {}

    def _mf(x):
        if x not in cache:
            cache[x] = f(x)
        return cache[x]
    return _mf


def find_root_deriv(f, tol, x_min, x_max, x_guess=None):
    """Newton-with-bisection-fallback root finder for f(x) -> (value, deriv)
    (verbatim meep semantics, meep/python/geom.py)."""
    f_memo = memoize(f)

    def lazy(x):
        return x if isinstance(x, (int, float)) else x()

    def pick_bound(which):
        def _pb():
            fmin, fmax = f_memo(x_min)[0], f_memo(x_max)[0]
            if which(fmin):
                return x_min
            if which(fmax):
                return x_max
            raise ValueError("failed to bracket the root in find_root_deriv")
        return _pb

    def in_bounds(x, fx, df, a, b):
        return (fx - df * (x - a)) * (fx - df * (x - b)) < 0

    def newton(x, a, b, dx):
        while abs(dx) >= abs(tol * x):
            fx, df = f_memo(x)
            if fx == 0:
                return x
            a_prime = x if fx < 0 else a
            b_prime = x if fx > 0 else b
            if (dx != x_max - x_min and dx * (fx / df) < 0
                    and f_memo(lazy(a_prime))[0] * f_memo(lazy(b_prime))[0] > 0):
                raise ValueError("failed to bracket the root in find_root_deriv")
            if df != 0 and in_bounds(x, fx, df, lazy(a_prime), lazy(b_prime)):
                dx, x = -fx / df, x - fx / df
                a, b = a_prime, b_prime
            else:
                av, bv = lazy(a_prime), lazy(b_prime)
                x_mid = 0.5 * (av + bv)
                dx, x = bv - av, x_mid
                a, b = av, bv
        return x

    x0 = x_guess if x_guess is not None else 0.5 * (x_min + x_max)
    return newton(x0, pick_bound(lambda v: v < 0), pick_bound(lambda v: v > 0),
                  x_max - x_min)


# --------------------------------------------------- Dft handle accessors
def get_eigenmode_freqs(f):
    return list(f.freqs)


def get_energy_freqs(f):
    return list(f.freqs)


def get_force_freqs(f):
    return list(f.freqs)


def _scale_dfts(s, handle):
    """Scale the accumulated DFT tensors of a monitor handle by real s."""
    s = complex(s)
    if s.imag:
        raise NotImplementedError("complex scale factors are not supported "
                                  "(the accumulators store re/im channels)")
    sim, key = handle.sim, handle.name
    hits = [k for k in sim._last_acc if k == key or k.startswith(key + ":")]
    if not hits:
        raise RuntimeError("scale_*_fields: run() before scaling (no "
                           f"accumulated data for {key!r})")
    for k in hits:
        acc, mon, meta = sim._last_acc[k]
        sim._last_acc[k] = (acc * s.real, mon, meta)


def scale_flux_fields(s, flux):
    _scale_dfts(s, flux)


def scale_energy_fields(s, ef):
    _scale_dfts(s, ef)


def scale_force_fields(s, force):
    _scale_dfts(s, force)


def scale_near2far_fields(s, near2far):
    _scale_dfts(s, near2far)


def get_forces(f):
    if hasattr(f, "force"):
        return f.force()
    raise NotImplementedError("the engine has no dft_force monitor")


def get_electric_energy(f):
    if hasattr(f, "electric"):
        return f.electric()
    raise NotImplementedError("the engine has no dft_energy monitor")


def get_magnetic_energy(f):
    if hasattr(f, "magnetic"):
        return f.magnetic()
    raise NotImplementedError("the engine has no dft_energy monitor")


def get_total_energy(f):
    if hasattr(f, "total"):
        return f.total()
    raise NotImplementedError("the engine has no dft_energy monitor")


class _LdosObj:
    """Frequency carrier returned by Ldos() (meep _dft_ldos analogue)."""

    def __init__(self, freqs):
        self.freqs = list(freqs)
        self.freq = self.freqs          # meep exposes .freq


def Ldos(*args):
    """Ldos(fcen, df, nfreq) or Ldos(freqs): LDOS frequency spec."""
    if len(args) == 3:
        fcen, df, nfreq = args
        return _LdosObj(np.linspace(fcen - df / 2, fcen + df / 2, nfreq))
    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        return _LdosObj(args[0])
    raise TypeError("Ldos takes (fcen, df, nfreq) or (freqs)")


def dft_ldos(*args, **kwargs):
    raise NotImplementedError(
        "module-level dft_ldos cannot register a monitor mid-run here; use "
        "sim.dft_ldos(fcen, df, nfreq) before run() and sim.get_ldos() after")


# --------------------------------------- step functions & stop conditions
def combine_step_funcs(*step_funcs):
    def _combine(sim):
        for f in step_funcs:
            f(sim)
    return _combine


def when_true(cond, *step_funcs):
    def _when_true(sim):
        if cond(sim):
            for f in step_funcs:
                f(sim)
    return _when_true


def when_false(cond, *step_funcs):
    return when_true(lambda sim: not cond(sim), *step_funcs)


def after_sources_and_time(t, *step_funcs):
    def _after(sim):
        if sim.round_time() >= sim._source_cutoff() + t:
            for f in step_funcs:
                f(sim)
    return _after


def synchronized_magnetic(*step_funcs):
    """meep synchronizes B/H to the E time before firing; the engine's
    stepped fields are read as-is (half-step offset, O(dt) difference)."""
    return combine_step_funcs(*step_funcs)


def display_progress(t0, t, dt):
    t_0 = wall_time()
    closure = {"tlast": t_0}

    def _disp(sim):
        t1 = wall_time()
        if t1 - closure["tlast"] >= dt:
            done = sim.meep_time() - t0
            el = t1 - t_0
            togo = (el * (t / done) - el) if done else 0
            if verbosity.meep > 0:
                print(f"Meep progress: {done}/{t} = {done / (0.01 * t):.1f}% "
                      f"done in {el:.1f}s, {togo:.1f}s to go")
            closure["tlast"] = t1
    return _disp


def stop_on_interrupt():
    shutting_down = [False]

    def _handler(sig, frame):
        print("WARNING: System requested termination. Time stepping aborted.")
        shutting_down[0] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    return lambda sim: shutting_down[0]


def _field_norm(sim):
    """L2 norm over the live state tensors (decay proxy for energy/DFT)."""
    st = sim._state
    if st is None:
        return 0.0
    import torch
    tot = 0.0
    for name in dir(st):
        v = getattr(st, name)
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            tot += float((v.detach() ** 2).sum())
    return math.sqrt(tot)


def stop_when_energy_decayed(dt=None, decay_by=None):
    """Stop once total field energy decayed to decay_by of its max, sampled
    every dt sim-time units (energy = squared field norm of the state)."""
    if dt is None or decay_by is None:
        raise ValueError("dt and decay_by are all required.")
    closure = {"max_abs": 0.0, "t0": 0.0}

    def _stop(sim):
        if sim.round_time() <= dt + closure["t0"]:
            return False
        cur = _field_norm(sim) ** 2
        closure["max_abs"] = max(closure["max_abs"], cur)
        closure["t0"] = sim.round_time()
        if closure["max_abs"] and verbosity.meep > 0:
            print(f"energy decay(t = {sim.meep_time()}): {cur} / "
                  f"{closure['max_abs']} = {cur / closure['max_abs']}")
        return bool(closure["max_abs"]) and cur <= closure["max_abs"] * decay_by

    return _stop


def _stop_when_decayed(sample, tol, minimum_run_time, maximum_run_time, dt):
    closure = {"prev": 0.0, "t0": 0.0, "maxchange": 0.0}

    def _stop(sim):
        if maximum_run_time and sim.round_time() > maximum_run_time:
            return True
        if sim.round_time() <= dt + closure["t0"]:
            return False
        cur = sample(sim)
        change = abs(closure["prev"] - cur)
        closure["maxchange"] = max(closure["maxchange"], change)
        first = closure["prev"] == 0.0
        closure["prev"], closure["t0"] = cur, sim.round_time()
        if first or not closure["maxchange"]:
            return False
        return (change / closure["maxchange"] <= tol
                and sim.round_time() >= minimum_run_time)

    return _stop


def stop_when_dft_decayed(tol=1e-11, minimum_run_time=0, maximum_run_time=None):
    """Stop once the DFT accumulators stop changing. The engine's running
    accumulators are not readable mid-run, so the (equivalent-in-the-limit)
    live field norm is monitored instead; once fields decay the DFTs freeze."""
    return _stop_when_decayed(_field_norm, tol, minimum_run_time,
                              maximum_run_time, dt=1.0)


def stop_when_flux_decayed(flux, tol=1e-11, minimum_run_time=0,
                           maximum_run_time=None):
    """Same proxy as stop_when_dft_decayed: flux spectra are frozen once the
    fields through the monitor have decayed."""
    return _stop_when_decayed(_field_norm, tol, minimum_run_time,
                              maximum_run_time, dt=1.0)


# ----------------------------------------------------- eigenmode handles
def get_k(vedata):
    return vedata.k


def get_group_velocity(vedata):
    return vedata.group_velocity


def destroy_eigenmode_data(vedata, destroy_mdata=True):
    pass                                # GC-managed; nothing to free


def py_do_harminv(vals, dt, f_min, f_max, maxbands=100, spectral_density=1.1,
                  Q_thresh=50.0, rel_err_thresh=math.inf, err_thresh=0.01,
                  rel_amp_thresh=-1.0, amp_thresh=-1.0):
    """Harmonic inversion of a time series; returns meep-shaped bands
    [(complex_freq, amp, err), ...] via the engine's harminv."""
    from ..modes.harminv import harminv as _harminv
    ms = _harminv(list(vals), dt, f_min, f_max, max_modes=maxbands or 100)
    out = []
    for m in ms:
        fim = -m.freq / (2.0 * m.Q) if math.isfinite(m.Q) and m.Q else 0.0
        out.append((complex(m.freq, fim), m.amp, 0.0))
    return out


# --------------------------------------------------- sources / visualization
def get_equiv_sources(field, normal_vec, time_src, center, size):
    """Equivalence-principle currents K = n x H, N = -n x E from a field
    slice, as Source objects (meep source.py port; the gridded amplitude is
    carried on ``.amp_data``)."""
    from .sources import Source
    ex, ey, ez, hx, hy, hz = field
    n = normal_vec
    currents = {
        "Ex": n[1] * hz - n[2] * hy, "Ey": n[2] * hx - n[0] * hz,
        "Ez": n[0] * hy - n[1] * hx,
        "Hx": n[2] * ey - n[1] * ez, "Hy": n[0] * ez - n[2] * ex,
        "Hz": n[1] * ex - n[0] * ey,
    }
    sources = []
    for comp, arr in currents.items():
        if np.sum(np.abs(arr)):
            s = Source(time_src, comp, center=center, size=size)
            s.amp_data = arr
            sources.append(s)
    return sources


def plot_fields(*args, **kwargs):
    """meep plot_fields (delegates to mp.visualization.plot_fields)."""
    from .visualization import plot_fields as _pf
    return _pf(*args, **kwargs)


def plot3D(sim, save_to_image=False, image_name="sim.png", **kwargs):
    raise NotImplementedError("plot3D (vispy 3D viewer) is not ported; "
                              "use plot2D / plot_fields")


# ------------------------------------------------------ files / GDSII / io
def make_output_directory(*args):
    path = args[0] if args else tempfile.mkdtemp(prefix="meep-torch-")
    os.makedirs(path, exist_ok=True)
    return path


def delete_directory(path):
    shutil.rmtree(path)


def trash_output_directory(dirname):
    shutil.rmtree(dirname, ignore_errors=True)
    os.makedirs(dirname, exist_ok=True)


def create_output_file(dirname, fname):
    from .cpp_bindings_core import h5file
    os.makedirs(dirname, exist_ok=True)
    return h5file(os.path.join(dirname, fname), h5file.WRITE)


def with_libGDSII():
    return False


def _no_gdsii(*_a, **_k):
    raise RuntimeError("meep-torch was built without libGDSII support; "
                       "GDSII import is unavailable")


GDSII_layers = GDSII_prisms = GDSII_vol = _no_gdsii
get_GDSII_layers = get_GDSII_prism = get_GDSII_prisms = _no_gdsii
get_GDSII_volume = set_geometry_from_GDSII = _no_gdsii


# export everything public except imported modules (for mp's star import)
import types as _types
__all__ = [n for n, v in list(globals().items())
           if not n.startswith("_") and not isinstance(v, _types.ModuleType)]
