# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/metal-cavity-ldos.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/metal-cavity-ldos/"
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
        _p = _p.with_name("metal-cavity-ldos_meep-" + _p.stem)
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

import matplotlib.pyplot as plt
import numpy as np

import meep as mp


def metal_cavity(w):
    resolution = 50
    sxy = 2
    dpml = 1
    sxy += 2 * dpml
    cell = mp.Vector3(sxy, sxy)

    pml_layers = [mp.PML(dpml)]
    a = 1
    t = 0.1
    geometry = [
        mp.Block(mp.Vector3(a + 2 * t, a + 2 * t, mp.inf), material=mp.metal),
        mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
    ]

    geometry.append(
        mp.Block(
            center=mp.Vector3(a / 2), size=mp.Vector3(2 * t, w, mp.inf), material=mp.air
        )
    )

    fcen = math.sqrt(0.5) / a
    df = 0.2
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3()
        )
    ]

    symmetries = [mp.Mirror(mp.Y)]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=symmetries,
        resolution=resolution,
    )

    h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=500)

    m = h.modes[0]
    f = m.freq
    Q = m.Q
    Vmode = 0.25 * a * a
    ldos_1 = Q / Vmode / (2 * math.pi * f * math.pi * 0.5)

    sim.reset_meep()

    T = 2 * Q * (1 / f)
    sim.run(mp.dft_ldos(f, 0, 1), until_after_sources=T)
    ldos_2 = sim.ldos_data[0]

    return ldos_1, ldos_2


ws = np.arange(0.2, 0.5, 0.1)
ldos_1 = np.zeros(len(ws))
ldos_2 = np.zeros(len(ws))

for j in range(len(ws)):
    ldos_1[j], ldos_2[j] = metal_cavity(ws[j])
    print(f"ldos:, {ldos_1[j]}, {ldos_2[2]}")

plt.figure(dpi=150)
plt.semilogy(1 / ws, ldos_1, "bo-", label="2Q/(πωV)")
plt.semilogy(1 / ws, ldos_2, "rs-", label="LDOS")
plt.xlabel("a/w")
plt.ylabel("2Q/(πωW) or LDOS")
plt.show()
