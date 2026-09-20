# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/stochastic_emitter_line.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/stochastic_emitter_line/"
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
        _p = _p.with_name("stochastic_emitter_line_meep-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    r = _savefig(self, fname, *a, **k)
    try:
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump
# --- end preamble ----------------------------------------------------------

import argparse

import numpy as np
from meep.materials import Ag

import meep as mp

parser = argparse.ArgumentParser()
parser.add_argument("-res", type=int, default=50, help="resolution (pixels/um)")
parser.add_argument("-nf", type=int, default=500, help="number of frequencies")
parser.add_argument(
    "-nsrc",
    type=int,
    default=15,
    help="number of line sources with cosine Fourier series amplitude function (method 3)",
)
parser.add_argument(
    "-textured",
    action="store_true",
    default=False,
    help="flat (default) or textured surface",
)
parser.add_argument(
    "-method",
    type=int,
    choices=[2, 3],
    default=2,
    help="type of method: (2) single dipole with 1 run per dipole or (3) line source with cosine Fourier series amplitude function",
)
args = parser.parse_args()

resolution = args.res

dpml = 1.0
dair = 0.9
hrod = 0.6
wrod = 0.8
dsub = 5.4
dAg = 0.4

sx = 1.5
sy = dpml + dair + hrod + dsub + dAg

cell_size = mp.Vector3(sx, sy)

pml_layers = [mp.PML(direction=mp.Y, thickness=dpml, side=mp.High)]

fcen = 1.0
df = 0.2
nfreq = args.nf
nsrc = args.nsrc
ndipole = int(sx * resolution)
run_time = 2 * nfreq / df

geometry = [
    mp.Block(
        material=mp.Medium(index=3.45),
        center=mp.Vector3(0, 0.5 * sy - dpml - dair - hrod - 0.5 * dsub),
        size=mp.Vector3(mp.inf, dsub, mp.inf),
    ),
    mp.Block(
        material=Ag,
        center=mp.Vector3(0, -0.5 * sy + 0.5 * dAg),
        size=mp.Vector3(mp.inf, dAg, mp.inf),
    ),
]

if args.textured:
    geometry.append(
        mp.Block(
            material=mp.Medium(index=3.45),
            center=mp.Vector3(0, 0.5 * sy - dpml - dair - 0.5 * hrod),
            size=mp.Vector3(wrod, hrod, mp.inf),
        )
    )


def src_amp_func(n):
    def _src_amp_func(p):
        if n == 0:
            return 1 / np.sqrt(sx)
        else:
            return np.sqrt(2 / sx) * np.cos(n * np.pi * (p.x + 0.5 * sx) / sx)

    return _src_amp_func


def compute_flux(m, n):
    if m == 2:
        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(
                    sx * (-0.5 + n / ndipole), -0.5 * sy + dAg + 0.5 * dsub
                ),
            )
        ]
    else:
        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(0, -0.5 * sy + dAg + 0.5 * dsub),
                size=mp.Vector3(sx, 0),
                amp_func=src_amp_func(n),
            )
        ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        k_point=mp.Vector3(),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
    )

    flux_mon = sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0.5 * sy - dpml), size=mp.Vector3(sx)),
    )

    sim.run(until=run_time)

    flux = mp.get_fluxes(flux_mon)
    freqs = mp.get_flux_freqs(flux_mon)

    return freqs, flux


if args.method == 2:
    fluxes = np.zeros((nfreq, ndipole))
    for d in range(ndipole):
        freqs, fluxes[:, d] = compute_flux(2, d)
else:
    fluxes = np.zeros((nfreq, nsrc))
    for d in range(nsrc):
        freqs, fluxes[:, d] = compute_flux(3, d)


if mp.am_master():
    with open(
        f'method{args.method}_{"textured" if args.textured else "flat"}_res{resolution}_nfreq{nfreq}_{"ndipole" if args.method == 2 else "nsrc"}{ndipole if args.method == 2 else nsrc}.npz',
        "wb",
    ) as f:
        np.savez(f, freqs=freqs, fluxes=fluxes)
