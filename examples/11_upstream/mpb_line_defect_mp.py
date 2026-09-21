# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_line_defect.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mpb_line_defect/"
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
        _p = _p.with_name("mpb_line_defect_mp-" + _p.stem)
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
# --- end preamble ----------------------------------------------------------

import math

import meep_torch.mp as mp
from meep_torch import mpb

# A line_defect waveguide in a 2d triangular lattice of dielectric
# rods (c.f. tri_rods.ctl), formed by a row of missing rods along the
# "x" direction.  (Here, "x" and "y" refer to the first and second
# basis directions.)  This structure supports a single guided band
# within the band gap, much like the analogous waveguide in a square
# lattice of rods (see "Photonic Crystals" by Joannopoulos et al.).

supercell_y = 7  # the (odd) number of lateral supercell periods

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, supercell_y),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

eps = 12  # the dielectric constant of the rods
r = 0.2  # the rod radius in the bulk crystal

geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=eps))]

# duplicate the bulk crystal rods over the supercell:
geometry = mp.geometric_objects_lattice_duplicates(geometry_lattice, geometry)

# add a rod of air, to erase a row of rods and form a waveguide:
geometry += [mp.Cylinder(r, material=mp.air)]

Gamma = mp.Vector3()
K_prime = mp.lattice_to_reciprocal(
    mp.Vector3(0.5), geometry_lattice
)  # edge of Brillouin zone.
k_points = mp.interpolate(4, [Gamma, K_prime])

# the bigger the supercell, the more bands you need to compute to get
# to the defect modes (the lowest band is "folded" supercell_y times):
extra_bands = 5  # number of extra bands to compute above the gap
num_bands = supercell_y + extra_bands

resolution = 32

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
)
ms.device = "cuda"
ms.filename_prefix = str(OUT / "mpb_line_defect_mp")


def main():
    # Compute the TM modes, outputting the Ez field in the *middle* of the
    # band.  (In general, the guided mode in such an air defect may have
    # exited the gap by the time it reaches the edge of the Brillouin
    # zone at K_prime.)
    ms.run_tm(
        mpb.output_at_kpoint(k_points[len(k_points) // 2]),
        ms.fix_efield_phase,
        mpb.output_efield_z,
    )


if __name__ == "__main__":
    main()
