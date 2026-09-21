# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/holey-wvg-cavity.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/holey-wvg-cavity/"
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
        _p = _p.with_name("holey-wvg-cavity_mp-" + _p.stem)
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

# Meep Tutorial: Hz-polarized transmission and reflection through a cavity
# formed by a periodic sequence of holes in a dielectric waveguide,
# with a defect formed by a larger spacing between one pair of holes.
# This structure is based on one analyzed in:
#    S. Fan, J. N. Winn, A. Devenyi, J. C. Chen, R. D. Meade, and
#    J. D. Joannopoulos, "Guided and defect modes in periodic dielectric
#    waveguides," J. Opt. Soc. Am. B, 12 (7), 1267-1272 (1995).
import argparse

import meep_torch.mp as mp


def main(args):
    resolution = 20  # pixels/um

    eps = 13  # dielectric constant of waveguide
    w = 1.2  # width of waveguide
    r = 0.36  # radius of holes
    d = 1.4  # defect spacing (ordinary spacing = 1)
    N = args.N  # number of holes on either side of defect

    sy = args.sy  # size of cell in y direction (perpendicular to wvg.)
    pad = 2  # padding between last hole and PML edge
    dpml = 1  # PML thickness

    sx = 2 * (pad + dpml + N) + d - 1  # size of cell in x direction

    cell = mp.Vector3(sx, sy, 0)

    blk = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=mp.Medium(epsilon=eps))

    geometry = [blk]

    for i in range(N):
        geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
        geometry.append(mp.Cylinder(r, center=mp.Vector3(-(d / 2 + i))))

    fcen = args.fcen  # pulse center frequency
    df = args.df  # pulse frequency width
    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[],
        boundary_layers=[mp.PML(dpml)],
        resolution=20,
    )

    if args.resonant_modes:
        sim.sources.append(
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df), component=mp.Hz, center=mp.Vector3()
            )
        )

        sim.symmetries.append(mp.Mirror(mp.Y, phase=-1))
        sim.symmetries.append(mp.Mirror(mp.X, phase=-1))

        sim.run(
            mp.at_beginning(mp.output_epsilon),
            mp.after_sources(mp.Harminv(mp.Hz, mp.Vector3(), fcen, df)),
            until_after_sources=400,
        )

        sim.run(mp.at_every(1 / fcen / 20, mp.output_hfield_z), until=1 / fcen)
    else:
        sim.sources.append(
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ey,
                center=mp.Vector3(-0.5 * sx + dpml),
                size=mp.Vector3(0, w),
            )
        )

        sim.symmetries.append(mp.Mirror(mp.Y, phase=-1))

        freg = mp.FluxRegion(
            center=mp.Vector3(0.5 * sx - dpml - 0.5), size=mp.Vector3(0, 2 * w)
        )

        nfreq = 500  # number of frequencies at which to compute flux

        # transmitted flux
        trans = sim.add_flux(fcen, df, nfreq, freg)

        vol = mp.Volume(mp.Vector3(), size=mp.Vector3(sx))

        sim.run(
            mp.at_beginning(mp.output_epsilon),
            mp.during_sources(
                mp.in_volume(
                    vol,
                    mp.to_appended("hz-slice", mp.at_every(0.4, mp.output_hfield_z)),
                )
            ),
            until_after_sources=mp.stop_when_fields_decayed(
                50, mp.Ey, mp.Vector3(0.5 * sx - dpml - 0.5), 1e-3
            ),
        )

        sim.display_fluxes(trans)  # print out the flux spectrum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resonant_modes",
        action="store_true",
        default=False,
        help="Compute resonant modes. Default is transmission spectrum.",
    )
    parser.add_argument(
        "-N", type=int, default=3, help="number of holes on either side of defect"
    )
    parser.add_argument(
        "-sy",
        type=int,
        default=6,
        help="size of cell in y direction (perpendicular to wvg.)",
    )
    parser.add_argument(
        "-fcen", type=float, default=0.25, help="pulse center frequency"
    )
    parser.add_argument("-df", type=float, default=0.2, help="pulse frequency width")
    args = parser.parse_args()
    main(args)
