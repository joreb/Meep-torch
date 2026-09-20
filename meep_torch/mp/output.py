# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep HDF5 output family: output_epsilon / output_efield_z / ... plus the
in_volume / in_point / to_appended output combinators.

Files are real: written with h5py when it is importable (meep writes .h5),
otherwise as numpy ``.npz`` with the same dataset names and array content (a
documented fallback -- install h5py for byte-level meep compatibility).
Complex arrays follow meep's convention of paired ``<name>.r`` / ``<name>.i``
datasets. Filenames follow meep: ``<prefix>-<tag>-<t:09.2f>``.
"""
import math
import os
import shutil

import numpy as np

from .run_control import Volume, _eval_step_func
from .cyl_shim import needs_cyl
from .functions import component_name, verbosity

_E = ("Ex", "Ey", "Er", "Ep", "Ez")
_D = ("Dx", "Dy", "Dr", "Dp", "Dz")
_H = ("Hx", "Hy", "Hr", "Hp", "Hz")
_B = ("Bx", "By", "Br", "Bp", "Bz")


def _write(sim, tag, datasets):
    """Write datasets to <prefix>-<tag>-<time> (h5 or npz); returns the path.
    Honors the to_appended target (accumulate along a new time axis)."""
    append = getattr(sim, "_append_target", None)
    if append is not None:
        fname, store = append
        for k, v in datasets.items():
            store.setdefault(k, []).append(np.asarray(v))
        # TRAILING TIME AXIS, because that is where meep puts it. meep's h5
        # writer grows the LAST dimension when appending, so a 160x160 slice
        # over 333 steps lands as (160, 160, 333). np.stack defaults to axis=0
        # and gave (333, 160, 160) -- the same numbers, transposed, which any
        # reader slicing [..., t] gets silently wrong. Caught by comparing
        # bent-waveguide's -ez.h5 against the oracle's: port[333,160,160]
        # against oracle[160,160,333].
        datasets = {k: np.stack(v, axis=-1) for k, v in store.items()}
        base = _outdir(sim, f"{_prefix(sim)}-{fname}")
    else:
        base = _outdir(sim, f"{_prefix(sim)}-{tag}-{sim.meep_time():09.2f}")
    out = {}
    for k, v in datasets.items():
        v = np.asarray(v)
        if np.iscomplexobj(v):
            out[k + ".r"], out[k + ".i"] = v.real, v.imag
        else:
            out[k] = v
    try:
        import h5py
    except ImportError:
        path = base + ".npz"
        np.savez(path, **out)
    else:
        path = base + ".h5"
        with h5py.File(path, "w") as f:
            for k, v in out.items():
                f.create_dataset(k, data=v)
    if verbosity.meep > 0:
        print(f"creating output file \"{path}\"...")
    sim.last_output_file = path
    if tag == "eps":
        sim.last_eps_filename = path
    return path


def _prefix(sim):
    """meep's filename prefix: filename_prefix, else the script name.

    This used to return a literal "meep_torch", so output files were named
    nothing like meep's -- the ring tutorial writes ring-ez-*.h5, not
    meep_torch-ez-*.h5, and a tutorial that reads its own output back by
    name would not find it.
    """
    get = getattr(sim, "get_filename_prefix", None)
    return get() if callable(get) else (
        getattr(sim, "filename_prefix", None) or "meep_torch")


def _outdir(sim, base: str) -> str:
    """Prepend Simulation.use_output_directory()'s directory, if one is set.

    Without this every output_* call writes to the process working
    directory, which for a script run from the repo root means the repo
    root: one run of the ring tutorial drops 21 .h5 files there.
    """
    d = getattr(sim, "_output_directory", None)
    if not d:
        return base
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, base)


def _volume_slice(sim, arr):
    """Restrict a full-cell 2D array to sim's in_volume region, if any."""
    v = getattr(sim, "_output_volume", None)
    if v is None or arr.ndim != 2:
        return arr
    from .coords import cell_index, cell_index_y
    c, s = v.center, v.size
    gs = sim.gs
    i0 = 0 if math.isinf(s.x) else cell_index(c.x - s.x / 2.0, gs)
    i1 = arr.shape[0] if math.isinf(s.x) else cell_index(c.x + s.x / 2.0, gs)
    j0 = 0 if math.isinf(s.y) else cell_index_y(c.y - s.y / 2.0, gs)
    j1 = arr.shape[1] if math.isinf(s.y) else cell_index_y(c.y + s.y / 2.0, gs)
    return arr[max(i0, 0):max(i1, i0 + 1), max(j0, 0):max(j1, j0 + 1)]


def _get(sim, component):
    """Component array, or None when the engine does not expose it.

    A 0-d or object-dtype result counts as NOT EXPOSED. get_array does not
    always raise for a component the run lacks -- a TM run asked for Hz hands
    back a 0-d object array, which is not None, so it sailed past the caller's
    `arr is None` check and reached h5py as
    ``TypeError: Object dtype dtype('O') has no native HDF5 equivalent``.
    """
    try:
        arr = np.asarray(sim.get_array(component=component))
    except (ValueError, RuntimeError, AttributeError):
        return None
    if arr.ndim == 0 or arr.dtype == object:
        return None
    return _volume_slice(sim, arr)


#: The 2D field components, split by polarisation. In a 2D run the two sets
#: decouple completely, so a component from the OTHER set is not missing -- it
#: is identically zero, and that is physics rather than an unimplemented
#: feature.
#: The 2D field components by polarisation. In a 2D run the two sets decouple,
#: so a component of the OTHER set is identically zero.
_TM_SET = frozenset("ez hx hy dz bz".split())
_TE_SET = frozenset("hz ex ey dx dy bx by".split())


def _is_zero_in_this_run(sim, component):
    """True only when the component belongs to the OTHER 2D polarisation.

    Decided from the run's own polarisation, which is the only signal that
    means what it looks like. Two earlier attempts did not:

    - matching the tag against a list of ALL nine 2D field names cannot tell
      "absent because the other polarisation" from "present but get_array
      cannot return it", and wrote zeros over both;
    - `fields.have_component` is a per-implementation constant (`c in
      _TM_COMPONENTS`), not a per-run fact, so it reports the TM set even for
      a TE run.

    Measured availability, which follows neither: a TM run exposes Ez, Hx, Hy
    AND Ex; a TE run exposes Hz and Hy but NOT its own Ex, Ey. So "get_array
    failed" cannot mean "the field is zero" -- Ex and Ey in a TE run are real,
    non-zero, and unavailable, and zeroing them was the bug this replaces.

    Under this rule they raise, which is the honest answer: the engine has the
    field and cannot hand it over. Only the complementary set is zero-filled,
    which is the case that motivated any of this -- cyl-ellipsoid.py drives Hz
    and ends with mp.output_efield_z, and Ez in a TE run is zero everywhere.
    """
    pol = getattr(sim, "_pol", None)
    if pol not in ("tm", "te"):
        return False                 # 3D or cylindrical: no such decoupling
    tag = component_name(component)
    return tag in (_TE_SET if pol == "tm" else _TM_SET)


def _zero_of_run(sim):
    """Full-grid zeros, shaped as meep shapes its output.

    From the dielectric grid, not from a field: meep returns the FULL grid for
    every component, while this engine returns Yee-native shapes, so a TE run
    donating Hz gave (59, 59) where meep writes (60, 60).
    """
    try:
        eps = np.asarray(sim.get_epsilon())
        if eps.ndim >= 2:
            return np.zeros(eps.shape, dtype=float)
    except Exception:
        pass
    return None


def _resolve(sim, component):
    """The array to write for one component, or a raise.

    ASKING A TE RUN FOR Ez IS NOT AN ERROR. cyl-ellipsoid.py drives Hz and
    ends with mp.output_efield_z; upstream meep runs it to completion, because
    in a TE run Ez is zero everywhere rather than absent. Raising there made a
    legal tutorial fail against a shim whose job is to accept what meep
    accepts. Anything the engine does hand over wins over the zero fill.
    """
    arr = _get(sim, component)
    if arr is None and _is_zero_in_this_run(sim, component):
        arr = _zero_of_run(sim)
    if arr is None:
        raise ValueError(f"the engine does not expose component {component!r} "
                         "for this run (see Simulation.get_array)")
    return arr


def _output_component(sim, component):
    tag = component_name(component)
    return _write(sim, tag, {tag: _resolve(sim, component)})


def _in_this_coordinate_system(sim, component):
    """False for a component of the coordinate system this run is not using.

    The family tuples above are the union over coordinate systems -- _E holds
    Er and Ep beside Ex and Ey -- so a cartesian run must drop the cylindrical
    pair outright rather than zero-fill or demand them. They are not zero in a
    cartesian run; they do not exist in it.
    """
    axis = component_name(component)[1:]
    return (axis in ("r", "p", "z")) if needs_cyl(sim) else (axis in ("x", "y", "z"))


def _output_family(sim, comps, tag):
    """Write a whole field family into one file, all components or none.

    Dropping the components _get cannot return was silent corruption: a TE
    output_efield wrote a file holding only the zero-filled ez and quietly lost
    ex and ey, the two that carry the run. A family file that names itself
    after the family has to contain it, so an unavailable component of THIS
    run's polarisation raises here exactly as it does on its own.
    """
    comps = [c for c in comps if _in_this_coordinate_system(sim, c)]
    return _write(sim, tag, {component_name(c): _resolve(sim, c) for c in comps})


def _mk_component_fn(c, name):
    # a real 1-arg def, not a defaulted-arg lambda: the meep step-function
    # protocol dispatches on positional-arg count (run_control._eval_step_func)
    def _out(sim):
        return _output_component(sim, c)
    _out.__name__ = name
    return _out


def _mk_family_fn(comps, tag, name):
    def _out(sim):
        return _output_family(sim, comps, tag)
    _out.__name__ = name
    return _out


def _mk_single(comps, letter):
    """output_<f>field_<d> functions for one family (x, y, z, r, p)."""
    return {name: _mk_component_fn(c, name) for c in comps
            for name in (f"output_{letter}field_{c[1].lower()}",)}


for _comps, _letter in ((_E, "e"), (_D, "d"), (_H, "h"), (_B, "b")):
    globals().update(_mk_single(_comps, _letter))
    globals()[f"output_{_letter}field"] = _mk_family_fn(
        _comps, _letter, f"output_{_letter}field")
del _comps, _letter


def output_epsilon(sim=None, *step_func_args, **kwargs):
    """Output the relative permittivity (eps_inf; frequency must be 0)."""
    if sim is None:
        return lambda s: output_epsilon(s, *step_func_args, **kwargs)
    frequency = kwargs.pop("frequency", 0.0)
    arr = _volume_slice(sim, np.asarray(sim.get_epsilon(frequency=frequency)))
    return _write(sim, "eps", {"eps": arr})


def output_mu(sim=None, *step_func_args, **kwargs):
    """Output the relative permeability (mu_inf; frequency must be 0)."""
    if sim is None:
        return lambda s: output_mu(s, *step_func_args, **kwargs)
    frequency = kwargs.pop("frequency", 0.0)
    arr = _volume_slice(sim, np.asarray(sim.get_mu(frequency=frequency)))
    return _write(sim, "mu", {"mu": arr})


def _epwr(sim):
    eps = np.asarray(sim.get_epsilon())
    arrs = [a for c in _E if (a := _get(sim, c)) is not None]
    if not arrs:
        raise ValueError("no electric field components available")
    return 0.5 * eps * sum(np.abs(a) ** 2 for a in arrs)


def _hpwr(sim):
    arrs = [a for c in _H if (a := _get(sim, c)) is not None]
    if not arrs:
        raise ValueError("no magnetic field components available")
    return 0.5 * sum(np.abs(a) ** 2 for a in arrs)


def output_dpwr(sim):
    """Output the electric-field energy density eps |E|^2 / 2."""
    return _write(sim, "denergy", {"denergy": _epwr(sim)})


def output_hpwr(sim):
    """Output the magnetic-field energy density |H|^2 / 2."""
    return _write(sim, "henergy", {"henergy": _hpwr(sim)})


def output_tot_pwr(sim):
    """Output the total energy density (electric + magnetic)."""
    return _write(sim, "energy", {"energy": _epwr(sim) + _hpwr(sim)})


def _poynting(sim, d):
    """S_d from E x H* (needs both transverse field pairs on the engine)."""
    pairs = {"x": ("Ey", "Hz", "Ez", "Hy"), "y": ("Ez", "Hx", "Ex", "Hz"),
             "z": ("Ex", "Hy", "Ey", "Hx"), "r": ("Ep", "Hz", "Ez", "Hp"),
             "p": ("Ez", "Hr", "Er", "Hz")}[d]
    a = [_get(sim, c) for c in pairs]
    if any(v is None for v in a):
        missing = [c for c, v in zip(pairs, a) if v is None]
        raise NotImplementedError(
            f"output_poynting_{d}: the engine does not expose {missing} "
            "for this run, so the cross product cannot be formed")
    return np.real(a[0] * np.conj(a[1]) - a[2] * np.conj(a[3]))


def _mk_poynting(d, name):
    def _out(sim):
        return _write(sim, f"s{d}", {f"s{d}": _poynting(sim, d)})
    _out.__name__ = name
    return _out


for _d in "xyzrp":
    globals()[f"output_poynting_{_d}"] = _mk_poynting(_d, f"output_poynting_{_d}")
    globals()[f"output_sfield_{_d}"] = _mk_poynting(_d, f"output_sfield_{_d}")
del _d


def output_poynting(sim):
    """Output every computable Poynting component into one file."""
    data = {}
    for d in "xyzrp":
        try:
            data[f"s{d}"] = _poynting(sim, d)
        except NotImplementedError:
            pass
    if not data:
        raise NotImplementedError(
            "output_poynting: the engine exposes no transverse field pair")
    return _write(sim, "s", data)


output_sfield = output_poynting


# ------------------------------------------------------ output combinators
# Each wrapper is a meep 2-arg step function f(sim, todo) whose inner funcs go
# through _eval_step_func, so wrappers and run_control combinators nest freely
# in either order (meep's protocol).
def in_volume(v, *step_funcs):
    """Restrict output functions among step_funcs to the Volume v."""
    def _in_volume(sim, todo="step"):
        save = getattr(sim, "_output_volume", None)
        sim._output_volume = v
        try:
            for f in step_funcs:
                _eval_step_func(sim, f, todo)
        finally:
            sim._output_volume = save
    return _in_volume


def in_point(pt, *step_funcs):
    """Restrict output functions among step_funcs to the single point pt."""
    return in_volume(Volume(center=pt), *step_funcs)


def to_appended(fname, *step_funcs):
    """Make output functions among step_funcs append along a new TRAILING time
    axis into one file named <prefix>-<fname> instead of per-time files.

    Trailing because meep's h5 writer grows the last dimension; see _write."""
    store = {}

    def _to_appended(sim, todo="step"):
        save = getattr(sim, "_append_target", None)
        sim._append_target = (fname, store)
        try:
            for f in step_funcs:
                _eval_step_func(sim, f, todo)
        finally:
            sim._append_target = save
    return _to_appended


def with_prefix(prefix, *step_funcs):
    """Prepend ``prefix`` to the file names output functions among step_funcs
    write (meep with_prefix)."""
    def _with_prefix(sim, todo="step"):
        save = getattr(sim, "filename_prefix", None)
        sim.filename_prefix = prefix + _prefix(sim)
        try:
            for f in step_funcs:
                _eval_step_func(sim, f, todo)
        finally:
            sim.filename_prefix = save
    return _with_prefix


def output_png(compnt, options, rm_h5=True):
    """Step function: output compnt then convert with h5topng (h5utils)."""
    if shutil.which("h5topng") is None:
        raise RuntimeError("output_png requires the h5topng binary from "
                           "h5utils; install h5utils (and h5py)")

    def _output_png(sim):
        _output_component(sim, compnt)
        fname = sim.last_output_file
        if not fname.endswith(".h5"):
            raise RuntimeError("output_png needs .h5 output; install h5py")
        cmd = f"h5topng {options} {fname}"
        if os.system(cmd) != 0:
            raise RuntimeError(f"error running h5topng: {cmd}")
        if rm_h5:
            os.remove(fname)
    return _output_png


# export everything public except imported modules (for mp's star import)
import types as _types
__all__ = [n for n, v in list(globals().items())
           if not n.startswith("_") and not isinstance(v, _types.ModuleType)]
