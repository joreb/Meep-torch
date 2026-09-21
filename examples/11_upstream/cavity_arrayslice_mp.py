# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/cavity_arrayslice.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/cavity_arrayslice/"
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
        _p = _p.with_name("cavity_arrayslice_mp-" + _p.stem)
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

import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

# set up the geometry
eps = 13
w = 1.2
r = 0.36
d = 1.4
N = 3
sy = 6
pad = 2
dpml = 1
sx = (2 * (pad + dpml + N)) + d - 1
fcen = 0.25
df = 0.2
nfreq = 500

cell = mp.Vector3(sx, sy, 0)

blk = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=mp.Medium(epsilon=eps))

geometry = [blk]

geometry.extend(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)) for i in range(3))
geometry.extend(mp.Cylinder(r, center=mp.Vector3(d / -2 - i)) for i in range(3))

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=[],
    boundary_layers=[mp.PML(dpml)],
    resolution=20,
)

# add sources
sim.sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Hz, mp.Vector3())]

# run until sources are finished (and no later)
sim._run_sources_until(0, [])

# get 1D and 2D array slices
xMin = -0.25 * sx
xMax = +0.25 * sx
yMin = -0.15 * sy
yMax = +0.15 * sy

# 1D slice of Hz data
size_1d = mp.Vector3(xMax - xMin)
center_1d = mp.Vector3((xMin + xMax) / 2)
slice1d = sim.get_array(mp.Volume(center_1d, size=size_1d), component=mp.Hz)

# 2D slice of Hz data
size_2d = mp.Vector3(xMax - xMin, yMax - yMin)
center_2d = mp.Vector3((xMin + xMax) / 2, (yMin + yMax) / 2)
slice2d = sim.get_array(mp.Volume(center_2d, size=size_2d), component=mp.Hz)

# plot 1D slice
plt.subplot(1, 2, 1)
x1d = np.linspace(xMin, xMax, len(slice1d))
plt.plot(x1d, slice1d)

# plot 2D slice
plt.subplot(1, 2, 2)
dy = (yMax - yMin) / slice2d.shape[1]
dx = (xMax - xMin) / slice2d.shape[0]
(x2d, y2d) = np.mgrid[slice(xMin, xMax, dx), slice(yMin, yMax, dy)]
plt.contourf(x2d, y2d, slice2d)
plt.colorbar()
plt.show()
