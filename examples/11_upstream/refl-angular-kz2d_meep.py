# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/refl-angular-kz2d.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/refl-angular-kz2d/"
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
        _p = _p.with_name("refl-angular-kz2d_meep-" + _p.stem)
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


# plt.show() IS A FIGURE TOO. Several tutorials build a figure and only call
# show(): upstream is interactive, the harness is headless, and under Agg
# show() is a no-op, so the figure -- and the numbers in it -- vanished. Those
# scripts scored "no comparable output" in the parity sweep while actually
# having plenty to compare. Route show() through the same savefig hook so they
# emit a PNG and a .plotdata.json like everything else.
import matplotlib.pyplot as _plt

_pltshow = _plt.show
_shown = set()


def _show_and_dump(*a, **k):
    for _i, _num in enumerate(_plt.get_fignums()):
        _f = _plt.figure(_num)
        if id(_f) in _shown:
            continue      # already written by an explicit savefig
        _shown.add(id(_f))
        try:
            _savefig_and_dump(_f, f"figure{_i}.png")
        except Exception:
            pass          # a figure we cannot write must not kill the tutorial
    return None


_plt.show = _show_and_dump
# --- end preamble ----------------------------------------------------------

import math

import meep as mp


def refl_planar(theta, kz_2d):
    resolution = 100

    dpml = 1.0
    sx = 10
    sx = 10 + 2 * dpml
    cell_size = mp.Vector3(sx)
    pml_layers = [mp.PML(dpml)]

    fcen = 1.0

    # plane of incidence is XZ
    k = mp.Vector3(z=math.sin(theta)).scale(fcen)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=0.2 * fcen),
            component=mp.Ey,
            center=mp.Vector3(-0.5 * sx + dpml),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        kz_2d=kz_2d,
        resolution=resolution,
    )

    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.25 * sx))
    refl = sim.add_flux(fcen, 0, 1, refl_fr)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ey, mp.Vector3(-0.5 * sx + dpml), 1e-9
        )
    )

    input_flux = mp.get_fluxes(refl)
    input_data = sim.get_flux_data(refl)
    sim.reset_meep()

    # add a block with n=3.5 for the air-dielectric interface
    geometry = [
        mp.Block(
            size=mp.Vector3(0.5 * sx, mp.inf, mp.inf),
            center=mp.Vector3(0.25 * sx),
            material=mp.Medium(index=3.5),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        kz_2d=kz_2d,
        resolution=resolution,
    )

    refl = sim.add_flux(fcen, 0, 1, refl_fr)
    sim.load_minus_flux_data(refl, input_data)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ey, mp.Vector3(-0.5 * sx + dpml), 1e-9
        )
    )

    refl_flux = mp.get_fluxes(refl)
    freqs = mp.get_flux_freqs(refl)

    return -refl_flux[0] / input_flux[0]


# rotation angle of source: CCW around Y axis, 0 degrees along +X axis
theta_r = math.radians(19.4)

Rmeep_real_imag = refl_planar(theta_r, "real/imag")
Rmeep_complex = refl_planar(theta_r, "complex")
Rmeep_3d = refl_planar(theta_r, "3d")

n1 = 1
n2 = 3.5

# compute angle of refracted planewave in medium n2
# for incident planewave in medium n1 at angle theta_in
theta_out = lambda theta_in: math.asin(n1 * math.sin(theta_in) / n2)

# compute Fresnel reflectance for S-polarization in medium n2
# for incident planewave in medium n1 at angle theta_in
Rfresnel = (
    lambda theta_in: math.fabs(
        (n2 * math.cos(theta_out(theta_in)) - n1 * math.cos(theta_in))
        / (n2 * math.cos(theta_out(theta_in)) + n1 * math.cos(theta_in))
    )
    ** 2
)

print(
    f"refl:, {Rmeep_real_imag} (real/imag), {Rmeep_complex} (complex), {Rmeep_3d} (3d), {Rfresnel(theta_r)} (analytic)"
)
