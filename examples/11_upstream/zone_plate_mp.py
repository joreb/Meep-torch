"""Computes the diffraction spectra of a zone plate in cylindrical coords."""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/zone_plate.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
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
OUT = pathlib.Path(__file__).parent / "output/zone_plate/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
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
        _p = _p.with_name("zone_plate_mp-" + _p.stem)
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
    r = _savefig(self, _p.with_name(_p.stem + "_mp" + _p.suffix), *a, **k)
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

import matplotlib.pyplot as plt
import meep_torch.mp as mp
import numpy as np


resolution_um = 25

pml_um = 1.0
substrate_um = 2.0
padding_um = 2.0
height_um = 0.5
focal_length_um = 200
scan_length_z_um = 100
farfield_resolution_um = 10

pml_layers = [mp.PML(thickness=pml_um)]

wavelength_um = 0.5
frequency = 1 / wavelength_um
frequench_width = 0.2 * frequency

# The number of zones in the zone plate.
# Odd-numbered zones impart a π phase shift and
# even-numbered zones impart no phase shift.
num_zones = 25

# Specify the radius of each zone using the equation
# from https://en.wikipedia.org/wiki/Zone_plate.
zone_radius_um = np.zeros(num_zones)
for n in range(1, num_zones + 1):
    zone_radius_um[n - 1] = math.sqrt(
        n * wavelength_um * (focal_length_um + n * wavelength_um / 4)
    )

size_r_um = zone_radius_um[-1] + padding_um + pml_um
size_z_um = pml_um + substrate_um + height_um + padding_um + pml_um
cell_size = mp.Vector3(size_r_um, 0, size_z_um)

# Specify a (linearly polarized) planewave at normal incidence.
sources = [
    mp.Source(
        mp.GaussianSource(frequency, fwidth=frequench_width, is_integrated=True),
        component=mp.Er,
        center=mp.Vector3(0.5 * size_r_um, 0, -0.5 * size_z_um + pml_um),
        size=mp.Vector3(size_r_um),
    ),
    mp.Source(
        mp.GaussianSource(frequency, fwidth=frequench_width, is_integrated=True),
        component=mp.Ep,
        center=mp.Vector3(0.5 * size_r_um, 0, -0.5 * size_z_um + pml_um),
        size=mp.Vector3(size_r_um),
        amplitude=-1j,
    ),
]

glass = mp.Medium(index=1.5)

# Add the substrate.
geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(size_r_um, 0, pml_um + substrate_um),
        center=mp.Vector3(
            0.5 * size_r_um, 0, -0.5 * size_z_um + 0.5 * (pml_um + substrate_um)
        ),
    )
]

# Add the zone plates starting with the ones with largest radius.
for n in range(num_zones - 1, -1, -1):
    geometry.append(
        mp.Block(
            material=glass if n % 2 == 0 else mp.vacuum,
            size=mp.Vector3(zone_radius_um[n], 0, height_um),
            center=mp.Vector3(
                0.5 * zone_radius_um[n],
                0,
                -0.5 * size_z_um + pml_um + substrate_um + 0.5 * height_um,
            ),
        )
    )

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution_um,
    sources=sources,
    geometry=geometry,
    dimensions=mp.CYLINDRICAL,
    m=-1,
)

# Add the near-field monitor (must be entirely in air).
n2f_monitor = sim.add_near2far(
    frequency,
    0,
    1,
    mp.Near2FarRegion(
        center=mp.Vector3(0.5 * (size_r_um - pml_um), 0, 0.5 * size_z_um - pml_um),
        size=mp.Vector3(size_r_um - pml_um, 0, 0),
    ),
    mp.Near2FarRegion(
        center=mp.Vector3(
            size_r_um - pml_um,
            0,
            0.5 * size_z_um - pml_um - 0.5 * (height_um + padding_um),
        ),
        size=mp.Vector3(0, 0, height_um + padding_um),
    ),
)

fig, ax = plt.subplots()
sim.plot2D(ax=ax)
if mp.am_master():
    fig.savefig("zone_plate_layout.png", bbox_inches="tight", dpi=150)

# Timestep the fields until they have sufficiently decayed away.
sim.run(
    until_after_sources=mp.stop_when_fields_decayed(
        50.0, mp.Er, mp.Vector3(0.5 * size_r_um, 0, 0), 1e-6
    )
)

farfields_r = sim.get_farfields(
    n2f_monitor,
    farfield_resolution_um,
    center=mp.Vector3(
        0.5 * (size_r_um - pml_um),
        0,
        -0.5 * size_z_um + pml_um + substrate_um + height_um + focal_length_um,
    ),
    size=mp.Vector3(size_r_um - pml_um, 0, 0),
)

farfields_z = sim.get_farfields(
    n2f_monitor,
    farfield_resolution_um,
    center=mp.Vector3(
        0, 0, -0.5 * size_z_um + pml_um + substrate_um + height_um + focal_length_um
    ),
    size=mp.Vector3(0, 0, scan_length_z_um),
)

intensity_r = (
    np.absolute(farfields_r["Ex"]) ** 2
    + np.absolute(farfields_r["Ey"]) ** 2
    + np.absolute(farfields_r["Ez"]) ** 2
)
intensity_z = (
    np.absolute(farfields_z["Ex"]) ** 2
    + np.absolute(farfields_z["Ey"]) ** 2
    + np.absolute(farfields_z["Ez"]) ** 2
)

# Plot the intensity data and save the result to disk.
fig, ax = plt.subplots(ncols=2)

ax[0].semilogy(np.linspace(0, size_r_um - pml_um, intensity_r.size), intensity_r, "bo-")
ax[0].set_xlim(-2, 20)
ax[0].set_xticks(np.arange(0, 25, 5))
ax[0].grid(True, axis="y", which="both", ls="-")
ax[0].set_xlabel(r"$r$ coordinate (μm)")
ax[0].set_ylabel(r"energy density of far fields, |E|$^2$")

ax[1].semilogy(
    np.linspace(
        focal_length_um - 0.5 * scan_length_z_um,
        focal_length_um + 0.5 * scan_length_z_um,
        intensity_z.size,
    ),
    intensity_z,
    "bo-",
)
ax[1].grid(True, axis="y", which="both", ls="-")
ax[1].set_xlabel(r"$z$ coordinate (μm)")
ax[1].set_ylabel(r"energy density of far fields, |E|$^2$")

fig.suptitle(f"binary-phase zone plate with focal length $z$ = {focal_length_um} μm")

if mp.am_master():
    fig.savefig("zone_plate_farfields.png", dpi=200, bbox_inches="tight")
