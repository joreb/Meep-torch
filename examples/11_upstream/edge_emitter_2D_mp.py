"""Computes the extraction efficiency of a cleaved stack using a 2D sim."""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/edge_emitter_2D.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/edge_emitter_2D/"
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
        _p = _p.with_name("edge_emitter_2D_mp-" + _p.stem)
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


import argparse
from typing import Tuple

import meep_torch.mp as mp
import matplotlib.pyplot as plt
import numpy as np


RESOLUTION_UM = 25
WAVELENGTH_UM = 1.0
CLADDING_UM = 0.5
ACTIVE_UM = 0.3
SUBSTRATE_UM = 1.0
PML_UM = 0.5
AIR_UM = 1.0
SIDE_UM = 3.0
N_SUBSTRATE = 3.2
N_CLADDING = 1.5
N_ACTIVE = 2.3
FIELD_DECAY_PERIOD = 25.0
FIELD_DECAY_TOL = 1e-6
DELTA_KZ = 0.02
DIPOLE_FLUX_TOL = 0.02
DEBUG_OUTPUT = 0


def line_current_in_cleaved_stack(
    dipole_pol: str, dipole_pos_um: float, kz: float
) -> Tuple[float, float]:
    """Computes the flux of a line current in a cleaved multilayer stack.

    Args:
        dipole_pol: polarization of the dipole ("x", "y", or "z").
        dipole_pos_um: position of the dipole relative to the edge of the
          active region.
        kz: z-component of the Bloch-periodic wavevector.

    Returns:
        The flux emitted by a line current in the active region and into air as
          a 2-tuple.
    """
    if dipole_pol == "x":
        src_cmpt = mp.Ex
    elif dipole_pol == "y":
        src_cmpt = mp.Ey
    elif dipole_pol == "z":
        src_cmpt = mp.Ez
    else:
        raise ValueError("dipole_pol must be x, y, or z.")

    cell_x_um = PML_UM + SIDE_UM + AIR_UM + PML_UM
    cell_y_um = (
        PML_UM + SUBSTRATE_UM + CLADDING_UM + ACTIVE_UM + CLADDING_UM + AIR_UM + PML_UM
    )
    cell_size = mp.Vector3(cell_x_um, cell_y_um, 0)

    boundary_layers = [mp.PML(thickness=PML_UM)]

    frequency = 1 / WAVELENGTH_UM

    src_pos = mp.Vector3(
        -0.5 * cell_x_um + PML_UM + SIDE_UM - dipole_pos_um,
        -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM + CLADDING_UM + 0.5 * ACTIVE_UM,
        0,
    )
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1 * frequency),
            center=src_pos,
            component=src_cmpt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=N_SUBSTRATE),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um + 0.5 * (PML_UM + SUBSTRATE_UM),
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, PML_UM + SUBSTRATE_UM, mp.inf),
        ),
        mp.Block(
            material=mp.Medium(index=N_CLADDING),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um + PML_UM + SUBSTRATE_UM + 0.5 * CLADDING_UM,
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, CLADDING_UM, mp.inf),
        ),
        mp.Block(
            material=mp.Medium(index=N_ACTIVE),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um
                + PML_UM
                + SUBSTRATE_UM
                + CLADDING_UM
                + 0.5 * ACTIVE_UM,
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, ACTIVE_UM, mp.inf),
        ),
        mp.Block(
            material=mp.Medium(index=N_CLADDING),
            center=mp.Vector3(
                -0.5 * cell_x_um + 0.5 * (PML_UM + SIDE_UM),
                -0.5 * cell_y_um
                + PML_UM
                + SUBSTRATE_UM
                + CLADDING_UM
                + ACTIVE_UM
                + 0.5 * CLADDING_UM,
                0,
            ),
            size=mp.Vector3(PML_UM + SIDE_UM, CLADDING_UM, mp.inf),
        ),
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        k_point=mp.Vector3(0, 0, kz),
        kz_2d="real/imag",
        force_complex_fields=False,
        sources=sources,
        geometry=geometry,
        boundary_layers=boundary_layers,
    )

    flux_mon = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(
                -0.5 * cell_x_um + PML_UM + 0.5 * (cell_x_um - 2 * PML_UM),
                0.5 * cell_y_um - PML_UM,
                0,
            ),
            size=mp.Vector3(cell_x_um - 2 * PML_UM, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                0.5 * cell_x_um - PML_UM,
                0.5 * cell_y_um - PML_UM - 0.5 * (cell_y_um - 2 * PML_UM),
                0,
            ),
            size=mp.Vector3(0, cell_y_um - 2 * PML_UM, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(
                0.5 * cell_x_um - PML_UM - +0.5 * AIR_UM, -0.5 * cell_y_um + PML_UM, 0
            ),
            size=mp.Vector3(AIR_UM, 0, 0),
            weight=-1.0,
        ),
    )

    if DEBUG_OUTPUT:
        if mp.am_master():
            fig, ax = plt.subplots()
            sim.plot2D(ax=ax)
            fig.savefig("edge_emitter_layout.png", dpi=150, bbox_inches="tight")

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD,
            src_cmpt,
            src_pos,
            FIELD_DECAY_TOL,
        ),
    )

    pixel_area = (1 / RESOLUTION_UM) ** 2
    dipole_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * pixel_area

    air_flux = mp.get_fluxes(flux_mon)[0]

    return dipole_flux, air_flux


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dipole_pol",
        type=str,
        choices=["x", "y", "z"],
        help="polarization of the electric dipole (x, y, z)",
    )
    parser.add_argument(
        "dipole_pos_um",
        type=float,
        help="position of the dipole relative to the edge of the cleaved facet",
    )
    args = parser.parse_args()

    num_kz = 0
    max_dipole_flux = 0
    dipole_flux = []
    air_flux = []

    # Brillouin-zone integration over kz.
    while True:
        kz = num_kz * DELTA_KZ
        dipole_flux_kz, air_flux_kz = line_current_in_cleaved_stack(
            args.dipole_pol, args.dipole_pos_um, kz
        )
        print(f"kz:, {num_kz}, {kz:.2f}, {dipole_flux_kz:.2f}, " f"{air_flux_kz:.2f}")

        dipole_flux.append(dipole_flux_kz)
        air_flux.append(air_flux_kz)
        num_kz += 1

        if dipole_flux_kz > max_dipole_flux:
            max_dipole_flux = dipole_flux_kz
        elif (dipole_flux_kz / max_dipole_flux) < DIPOLE_FLUX_TOL:
            break

    dipole_flux = np.array(dipole_flux)
    air_flux = np.array(air_flux)
    extraction_efficiency = (air_flux[0] + 2 * np.sum(air_flux[1:])) / (
        dipole_flux[0] + 2 * np.sum(dipole_flux[1:])
    )

    print(
        f"extraction_efficiency:, {args.dipole_pol}, {args.dipole_pos_um:.2f},"
        f" {100 * extraction_efficiency:.2f}%"
    )
