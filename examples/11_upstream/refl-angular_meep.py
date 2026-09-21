# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/refl-angular.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
OUT = pathlib.Path(__file__).parent / "output/refl-angular/"
OUT.mkdir(exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/


def _dump_plot_data(_fig, _path):
    """Write the NUMBERS behind a figure next to it, as <name>.plotdata.json.

    Several tutorials compute a result and only plot it -- binary_grating_
    phasemap, gaussian-beam and refl-quartz print nothing numeric and write no
    data file, so a stdout-and-files comparison has nothing to say about them
    at all. The numbers are right there in the axes.

    This lives in the generated preamble and not in the body, which is the
    whole point: it is emitted identically into both twins, so the
    byte-comparability that makes a twin difference mean anything survives. A
    record() call in the body would not.
    """
    import json as _json
    out, n = {}, 0
    for ai, ax in enumerate(_fig.get_axes()):
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs):
                out[f"ax{ai}.line{li}.x"] = [float(v) for v in xs]
                out[f"ax{ai}.line{li}.y"] = [float(v) for v in ys]
                n += 1
        for ii, im in enumerate(ax.get_images()):
            a = im.get_array()
            if a is not None and a.size:
                import numpy as _np
                a = _np.asarray(a, dtype=float)
                out[f"ax{ai}.image{ii}.shape"] = list(a.shape)
                out[f"ax{ai}.image{ii}.stats"] = [float(a.min()), float(a.max()),
                                                  float(a.mean()),
                                                  float((a ** 2).mean() ** 0.5)]
                n += 1
    if n:
        # PREFIXED WITH THE SCRIPT STEM, because the figure's name comes from
        # upstream's body and is therefore IDENTICAL on both twins. Without the
        # prefix both write Ez_angle0.plotdata.json into one directory and the
        # second run overwrites the first -- the same defect that made the two
        # mpb twins clobber each other's h5 output.
        _p = pathlib.Path(str(_path))
        _p = _p.with_name("refl-angular_meep-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    # SUFFIXED WITH THE TWIN TAG, for the same reason the plotdata is prefixed
    # with the stem: the figure's name comes from upstream's body and is
    # identical on both twins, so `_mp` and `_meep` wrote the same PNG into one
    # directory and whichever ran second won.
    _p = pathlib.Path(str(fname))
    r = _savefig(self, _p.with_name(_p.stem + "_meep" + _p.suffix), *a, **k)
    try:
        # the ORIGINAL name: the plotdata already carries the "<stem>-" prefix
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump
# --- end preamble ----------------------------------------------------------

import argparse
import math

import meep as mp


def main(args):
    resolution = args.res

    dpml = 1.0  # PML thickness
    sz = 10 + 2 * dpml  # size of computational cell (without PMLs)
    cell_size = mp.Vector3(0, 0, sz)
    pml_layers = [mp.PML(dpml)]

    wvl_min = 0.4  # min wavelength
    wvl_max = 0.8  # max wavelength
    fmin = 1 / wvl_max  # min frequency
    fmax = 1 / wvl_min  # max frequency
    fcen = 0.5 * (fmin + fmax)  # center frequency
    df = fmax - fmin  # frequency width
    nfreq = 50  # number of frequency bins

    # rotation angle (in degrees) of source: CCW around Y axis, 0 degrees along +Z axis
    theta_r = math.radians(args.theta)

    # plane of incidence is XZ
    k = mp.Vector3(math.sin(theta_r), 0, math.cos(theta_r)).scale(fmin)

    # if normal incidence, force number of dimensions to be 1
    dimensions = 1 if theta_r == 0 else 3
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.5 * sz + dpml),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        dimensions=dimensions,
        resolution=resolution,
    )

    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, -0.25 * sz))
    refl = sim.add_flux(fcen, df, nfreq, refl_fr)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(0, 0, -0.5 * sz + dpml), 1e-9
        )
    )

    empty_flux = mp.get_fluxes(refl)
    empty_data = sim.get_flux_data(refl)
    sim.reset_meep()

    # add a block with n=3.5 for the air-dielectric interface
    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, 0.5 * sz),
            center=mp.Vector3(0, 0, 0.25 * sz),
            material=mp.Medium(index=3.5),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        dimensions=dimensions,
        resolution=resolution,
    )

    refl = sim.add_flux(fcen, df, nfreq, refl_fr)
    sim.load_minus_flux_data(refl, empty_data)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(0, 0, -0.5 * sz + dpml), 1e-9
        )
    )

    refl_flux = mp.get_fluxes(refl)
    freqs = mp.get_flux_freqs(refl)

    for i in range(nfreq):
        print(
            f"refl:, {k.x}, {1 / freqs[i]}, {math.degrees(math.asin(k.x/freqs[i]))}, {-refl_flux[i] / empty_flux[i]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-res", type=int, default=200, help="resolution (default: 200 pixels/um)"
    )
    parser.add_argument(
        "-theta",
        type=float,
        default=0,
        help="angle of incident planewave (default: 0 degrees)",
    )
    args = parser.parse_args()
    main(args)
