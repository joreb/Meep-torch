# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/coupler.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/coupler/"
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
        _p = _p.with_name("coupler_mp-" + _p.stem)
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

import argparse

import gdstk
import meep_torch.mp as mp


gds_file = "coupler.gds"
CELL_LAYER = 0
PORT1_LAYER = 1
PORT2_LAYER = 2
PORT3_LAYER = 3
PORT4_LAYER = 4
SOURCE_LAYER = 5
UPPER_BRANCH_LAYER = 31
LOWER_BRANCH_LAYER = 32
default_d = 0.3

t_oxide = 1.0
t_Si = 0.22
t_air = 0.78

dpml = 1
cell_thickness = dpml + t_oxide + t_Si + t_air + dpml

oxide = mp.Medium(epsilon=2.25)
silicon = mp.Medium(epsilon=12)

fcen = 1 / 1.55
df = 0.2 * fcen


def get_gds_cell(fname):
    """Returns the (single) top-level cell of the GDS file `fname`."""
    return gdstk.read_gds(fname).top_level()[0]


def get_gds_prisms(material, cell, layer, datatype=0, zmin=0.0, zmax=0.0):
    """Returns a list of `mp.Prism`s, one for each polygon on (`layer`, `datatype`)."""
    prisms = []
    for poly in cell.get_polygons(layer=layer, datatype=datatype):
        vertices = [mp.Vector3(x, y, zmin) for x, y in poly.points]
        prisms.append(
            mp.Prism(
                vertices,
                height=zmax - zmin,
                axis=mp.Vector3(0, 0, 1),
                material=material,
            )
        )
    return prisms


def get_gds_vol(cell, layer, datatype=0, zmin=0.0, zmax=0.0):
    """Returns an `mp.Volume` spanning the bounding box of (`layer`, `datatype`)."""
    polygons = cell.get_polygons(layer=layer, datatype=datatype)
    xs = [x for poly in polygons for x, y in poly.points]
    ys = [y for poly in polygons for x, y in poly.points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    center = mp.Vector3(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
    size = mp.Vector3(xmax - xmin, ymax - ymin, zmax - zmin)
    dims = 2 if (zmin == 0 and zmax == 0) else 3
    return mp.Volume(center=center, size=size, dims=dims)


def main(args):
    cell_zmax = 0.5 * cell_thickness if args.three_d else 0
    cell_zmin = -0.5 * cell_thickness if args.three_d else 0
    si_zmax = 0.5 * t_Si if args.three_d else 10
    si_zmin = -0.5 * t_Si if args.three_d else -10

    # read cell size, volumes for source region and flux monitors,
    # and coupler geometry from the GDS file using gdstk
    gds_cell = get_gds_cell(gds_file)

    upper_branch = get_gds_prisms(
        silicon, gds_cell, UPPER_BRANCH_LAYER, zmin=si_zmin, zmax=si_zmax
    )
    lower_branch = get_gds_prisms(
        silicon, gds_cell, LOWER_BRANCH_LAYER, zmin=si_zmin, zmax=si_zmax
    )

    cell = get_gds_vol(gds_cell, CELL_LAYER, zmin=cell_zmin, zmax=cell_zmax)
    p1 = get_gds_vol(gds_cell, PORT1_LAYER, zmin=si_zmin, zmax=si_zmax)
    p2 = get_gds_vol(gds_cell, PORT2_LAYER, zmin=si_zmin, zmax=si_zmax)
    p3 = get_gds_vol(gds_cell, PORT3_LAYER, zmin=si_zmin, zmax=si_zmax)
    p4 = get_gds_vol(gds_cell, PORT4_LAYER, zmin=si_zmin, zmax=si_zmax)
    src_vol = get_gds_vol(gds_cell, SOURCE_LAYER, zmin=si_zmin, zmax=si_zmax)

    # displace upper and lower branches of coupler (as well as source and flux regions)
    if args.d != default_d:
        delta_y = 0.5 * (args.d - default_d)
        delta = mp.Vector3(y=delta_y)
        p1.center += delta
        p2.center -= delta
        p3.center += delta
        p4.center -= delta
        src_vol.center += delta
        cell.size += 2 * delta
        for np in range(len(lower_branch)):
            lower_branch[np].center -= delta
            for nv in range(len(lower_branch[np].vertices)):
                lower_branch[np].vertices[nv] -= delta
        for np in range(len(upper_branch)):
            upper_branch[np].center += delta
            for nv in range(len(upper_branch[np].vertices)):
                upper_branch[np].vertices[nv] += delta

    geometry = upper_branch + lower_branch

    if args.three_d:
        oxide_center = mp.Vector3(z=-0.5 * t_oxide)
        oxide_size = mp.Vector3(cell.size.x, cell.size.y, t_oxide)
        oxide_layer = [mp.Block(material=oxide, center=oxide_center, size=oxide_size)]
        geometry = geometry + oxide_layer

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            volume=src_vol,
            eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z,
        )
    ]

    sim = mp.Simulation(
        resolution=args.res,
        cell_size=cell.size,
        boundary_layers=[mp.PML(dpml)],
        sources=sources,
        geometry=geometry,
    )

    mode1 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p1))
    mode2 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p2))
    mode3 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p3))
    mode4 = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(volume=p4))

    sim.run(until_after_sources=100)

    # S parameters
    p1_coeff = sim.get_eigenmode_coefficients(
        mode1, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 0]
    p2_coeff = sim.get_eigenmode_coefficients(
        mode2, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 1]
    p3_coeff = sim.get_eigenmode_coefficients(
        mode3, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 0]
    p4_coeff = sim.get_eigenmode_coefficients(
        mode4, [1], eig_parity=mp.NO_PARITY if args.three_d else mp.EVEN_Y + mp.ODD_Z
    ).alpha[0, 0, 0]

    # transmittance
    p2_trans = abs(p2_coeff) ** 2 / abs(p1_coeff) ** 2
    p3_trans = abs(p3_coeff) ** 2 / abs(p1_coeff) ** 2
    p4_trans = abs(p4_coeff) ** 2 / abs(p1_coeff) ** 2

    print(f"trans:, {args.d:.2f}, {p2_trans:.6f}, {p3_trans:.6f}, {p4_trans:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-res", type=int, default=50, help="resolution (default: 50 pixels/um)"
    )
    parser.add_argument(
        "-d", type=float, default=0.1, help="branch separation (default: 0.1 um)"
    )
    parser.add_argument(
        "--three_d",
        action="store_true",
        default=False,
        help="d calculation? (default: False)",
    )
    args = parser.parse_args()
    main(args)
