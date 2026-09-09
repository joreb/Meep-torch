# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb module-level band functions.

These are the callables users pass to ``ms.run_tm(...)`` and friends. Each
takes ``(ms, which_band)`` -- except the thunks, which take ``(ms)`` and run
once per k-point -- and the run loop dispatches on the argument count.

Bodies mirror meep/python/mpb/solver.py so the surface is
substitutable: every one of them fetches the field with
``bloch_phase=False`` (the periodic envelope) and lets the output writer
multiply the Bloch phase back in, which is what puts the phase in the
file exactly once.
"""
from ..mp.verbosity_mgr import verbosity

__all__ = [
    "output_hfield", "output_hfield_x", "output_hfield_y", "output_hfield_z",
    "output_bfield", "output_bfield_x", "output_bfield_y", "output_bfield_z",
    "output_dfield", "output_dfield_x", "output_dfield_y", "output_dfield_z",
    "output_efield", "output_efield_x", "output_efield_y", "output_efield_z",
    "output_bpwr", "output_dpwr", "output_tot_pwr", "output_dpwr_in_objects",
    "fix_hfield_phase", "fix_bfield_phase", "fix_dfield_phase",
    "fix_efield_phase", "combine_band_functions", "output_at_kpoint",
    "apply_band_func", "apply_band_func_thunk", "output_epsilon",
    "display_yparities", "display_zparities", "display_group_velocities",
]


def _getter(ms, kind):
    return getattr(ms, f"get_{kind}field")


def _make(kind, suffix):
    def _fn(ms, which_band):
        _getter(ms, kind)(which_band, False)
        getattr(ms, f"output_field{suffix}")()
    _fn.__name__ = f"output_{kind}field{suffix}"
    _fn.__doc__ = (f"Band function: write the {kind.upper()} field"
                   f"{' component ' + suffix[1:] if suffix else ''} of "
                   f"``which_band`` to HDF5.")
    return _fn


output_hfield = _make("h", "")
output_hfield_x = _make("h", "_x")
output_hfield_y = _make("h", "_y")
output_hfield_z = _make("h", "_z")
output_bfield = _make("b", "")
output_bfield_x = _make("b", "_x")
output_bfield_y = _make("b", "_y")
output_bfield_z = _make("b", "_z")
output_dfield = _make("d", "")
output_dfield_x = _make("d", "_x")
output_dfield_y = _make("d", "_y")
output_dfield_z = _make("d", "_z")
output_efield = _make("e", "")
output_efield_x = _make("e", "_x")
output_efield_y = _make("e", "_y")
output_efield_z = _make("e", "_z")


def output_epsilon(ms):
    """Thunk: write the dielectric function once per k-point."""
    ms.output_epsilon()


def output_bpwr(ms, which_band):
    ms.get_bfield(which_band, False)
    ms.compute_field_energy()
    ms.output_field()


def output_dpwr(ms, which_band):
    ms.get_dfield(which_band, False)
    ms.compute_field_energy()
    ms.output_field()


def output_tot_pwr(ms, which_band):
    ms.get_tot_pwr(which_band)
    ms.output_field_to_file(-1, f"{ms.get_filename_prefix()}tot.")


def output_dpwr_in_objects(output_func, min_energy, objects=[]):
    """Return a band function that calls ``output_func`` only for bands with
    more than ``min_energy`` of their D energy inside ``objects``."""
    def _output(ms, which_band):
        ms.get_dfield(which_band, False)
        ms.compute_field_energy()
        energy = ms.compute_energy_in_objects(objects)
        if verbosity.mpb > 0:
            print(f"dpwr:, {which_band}, {ms.freqs[which_band - 1]}, {energy} ")
        if energy >= min_energy:
            apply_band_func(ms, output_func, which_band)
    return _output


def _fix(kind):
    def _fn(ms, which_band):
        _getter(ms, kind)(which_band, False)
        ms.fix_field_phase()
    _fn.__name__ = f"fix_{kind}field_phase"
    _fn.__doc__ = (f"Band function: pick MPB's canonical phase for "
                   f"``which_band`` from its {kind.upper()} field.")
    return _fn


fix_hfield_phase = _fix("h")
fix_bfield_phase = _fix("b")
fix_dfield_phase = _fix("d")
fix_efield_phase = _fix("e")


def display_yparities(ms):
    """Band function (thunk): MPB's ``yparity:, <k>, p1, p2, ...`` line."""
    ms.display_kpoint_data("yparity", ms.compute_yparities())


def display_zparities(ms):
    """Band function (thunk): MPB's ``zparity:, <k>, p1, p2, ...`` line.

    The numbers are ``<h|M_z|h>`` on each eigenvector: +1 for an exactly
    even mode, -1 for odd. A run inside a parity sector prints 1.0 for
    every band, which is what makes the line a check rather than a
    decoration -- anything else means the sector was not held.
    """
    ms.display_kpoint_data("zparity", ms.compute_zparities())


def display_group_velocities(ms):
    """Band function (thunk): MPB's ``velocity:, <k>, v1, v2, ...`` line."""
    ms.display_kpoint_data("velocity", ms.compute_group_velocities())


def _num_args(fn):
    import inspect
    return len(inspect.signature(fn).parameters)


def apply_band_func_thunk(ms, band_func, which_band, eval_thunk):
    """Dispatch a band function that may be a thunk (one argument, evaluated
    once per k-point) or a per-band function (two arguments)."""
    if _num_args(band_func) == 1:
        if eval_thunk:
            band_func(ms)
    else:
        band_func(ms, which_band)


def apply_band_func(ms, band_func, which_band):
    apply_band_func_thunk(ms, band_func, which_band, which_band == 1)


def combine_band_functions(*band_funcs):
    """Combine zero or more band functions into one."""
    def _combine(ms, which_band):
        for f in band_funcs:
            apply_band_func(ms, f, which_band)
    return _combine


def output_at_kpoint(kpoint, *band_funcs):
    """Only invoke the given band functions at the specified k-point."""
    band_func = combine_band_functions(*band_funcs)

    def _output_at_kpoint(ms, which_band):
        if ms.current_k.close(kpoint, tol=1e-8 * kpoint.norm()):
            band_func(ms, which_band)
    return _output_at_kpoint
