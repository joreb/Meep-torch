# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/oblique-source.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/oblique-source/"
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
        _p = _p.with_name("oblique-source_meep-" + _p.stem)
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

import matplotlib.pyplot as plt
import numpy as np

import meep as mp

resolution = 50  # pixels/μm

cell_size = mp.Vector3(14, 14)

pml_layers = [mp.PML(thickness=2)]

# rotation angle (in degrees) of waveguide, counter clockwise (CCW) around z-axis
rot_angle = np.radians(20)

w = 1.0  # width of waveguide

geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
        e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle),
        e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
        material=mp.Medium(epsilon=12),
    )
]

fsrc = 0.15  # frequency of eigenmode or constant-amplitude source
bnum = 1  # band number of eigenmode

kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle)

compute_flux = True  # compute flux (True) or plot the field profile (False)

eig_src = True  # eigenmode (True) or constant-amplitude (False) source

if eig_src:
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc)
            if compute_flux
            else mp.ContinuousSource(fsrc),
            center=mp.Vector3(),
            size=mp.Vector3(y=3 * w),
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            eig_band=bnum,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]
else:
    sources = [
        mp.Source(
            src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc)
            if compute_flux
            else mp.ContinuousSource(fsrc),
            center=mp.Vector3(),
            size=mp.Vector3(y=3 * w),
            component=mp.Ez,
        )
    ]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    sources=sources,
    geometry=geometry,
    symmetries=[mp.Mirror(mp.Y)] if rot_angle == 0 else [],
)

if compute_flux:
    tran = sim.add_flux(
        fsrc, 0, 1, mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(y=14))
    )
    sim.run(until_after_sources=50)
    res = sim.get_eigenmode_coefficients(
        tran,
        [1],
        eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
        direction=mp.NO_DIRECTION,
        kpoint_func=lambda f, n: kpoint,
    )
    print(
        "flux:, {:.6f}, {:.6f}".format(
            mp.get_fluxes(tran)[0], abs(res.alpha[0, 0, 0]) ** 2
        )
    )
else:
    sim.run(until=100)
    sim.plot2D(
        output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(10, 10)),
        fields=mp.Ez,
        field_parameters={"alpha": 0.9},
    )
    plt.show()
