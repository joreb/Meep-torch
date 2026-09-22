"""Radiation pattern of a dipole antenna above a ground plane using Brillouin-zone integration.

Tutorial reference:

https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#antenna-above-a-perfect-electric-conductor-ground-plane
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/antenna_pec_ground_plane_1D.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/antenna_pec_ground_plane_1D/"
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
        _p = _p.with_name("antenna_pec_ground_plane_1D_mp-" + _p.stem)
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
from typing import Tuple

import matplotlib.pyplot as plt
import meep_torch.mp as mp
import numpy as np


RESOLUTION_UM = 50
PML_UM = 1.0
BULK_UM = 10.0
N_BACKGROUND = 1.2
ANTENNA_HEIGHT_UM = 1.25
WAVELENGTH_UM = 0.65
NUM_POLAR = 50
FIELD_DECAY_THRESHOLD = 1e-6
FIELD_DECAY_PERIOD = 25

frequency = 1 / WAVELENGTH_UM


def planewave_above_ground_plane(kx: float, ky: float, kz: float) -> float:
    """
    Returns the flux from a linearly polarized planewave above a ground plane.

    Args:
        kx, ky, kz: the wavevector components of the planewave.

    Returns:
        The radiated flux in z.
    """
    size_z_um = BULK_UM + PML_UM
    pml_layers = [mp.PML(thickness=PML_UM, direction=mp.Z, side=mp.High)]
    src_pt = mp.Vector3(0, 0, -0.5 * size_z_um + ANTENNA_HEIGHT_UM)

    cell_size = mp.Vector3(0, 0, size_z_um)

    src_cmpt = mp.Ey
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.1 * frequency),
            component=src_cmpt,
            center=src_pt,
            size=mp.Vector3(),
        )
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        default_material=mp.Medium(index=N_BACKGROUND),
        cell_size=cell_size,
        sources=sources,
        boundary_layers=pml_layers,
        k_point=mp.Vector3(kx, ky, kz),
    )

    sim.set_boundary(mp.Low, mp.Z, mp.Metallic)
    sim.set_boundary(mp.High, mp.Z, mp.Metallic)

    mon_pt = mp.Vector3(0, 0, 0.5 * size_z_um - PML_UM)
    dft_flux_z = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(center=mon_pt, size=mp.Vector3(), direction=mp.Z),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, src_cmpt, mon_pt, FIELD_DECAY_THRESHOLD
        )
    )

    flux_z = mp.get_fluxes(dft_flux_z)[0]

    return flux_z


def planewave_wavevector(polar_rad: float) -> Tuple[float, float, float]:
    """Returns the wavevector of the outgoing planewave in free space.

    Args:
        polar_rad: polar angle of the wavevector for φ = 0 (xz plane). 0 is
            the +z direction.

    Returns:
        The wavevector of the outgoing planewave.
    """
    kx = N_BACKGROUND * frequency * np.sin(polar_rad)
    ky = 0
    kz = N_BACKGROUND * frequency * np.cos(polar_rad)

    return kx, ky, kz


if __name__ == "__main__":
    polar_rad = np.linspace(0, 0.5 * np.pi, NUM_POLAR)
    radial_flux_meep = np.zeros(NUM_POLAR)

    for i in range(NUM_POLAR):
        kx, ky, kz = planewave_wavevector(polar_rad[i])

        # Skip wavevectors which are close to the light cone
        # due to poor absorption by PML (i.e. glancing-angle waves).
        if kx > (0.95 * N_BACKGROUND * frequency):
            continue

        flux_z = planewave_above_ground_plane(kx, ky, kz)
        radial_flux_meep[i] = np.cos(polar_rad[i]) * flux_z

    # The radiation pattern of a two-element antenna
    # array is equivalent to the radiation pattern of
    # a single antenna multiplied by its array factor
    # as derived in Section 6.2 "Two-Element Array" of
    # Antenna Theory: Analysis and Design, Fourth Edition
    # (2016) by C.A. Balanis.
    k_free_space = 2 * math.pi / (WAVELENGTH_UM / N_BACKGROUND)
    radial_flux_analytic = (
        np.sin(k_free_space * ANTENNA_HEIGHT_UM * np.cos(polar_rad)) ** 2
    )

    # Find the smallest angle at which the radiation pattern is a maximum using
    # the formula for the radiation pattern. This angle is used to compute the
    # radial flux which is then used for normalization of the radiation pattern.
    polar_rad_max = math.acos(9 * math.pi / (2 * k_free_space * ANTENNA_HEIGHT_UM))
    kx, ky, kz = planewave_wavevector(polar_rad_max)
    flux_z = planewave_above_ground_plane(kx, ky, kz)
    radial_flux_meep_max = np.cos(polar_rad_max) * flux_z

    radial_flux_meep = radial_flux_meep / radial_flux_meep_max

    rel_err = np.linalg.norm(radial_flux_meep - radial_flux_analytic) / np.linalg.norm(
        radial_flux_analytic
    )
    print(f"error:, {rel_err:.6f}")

    fig, ax = plt.subplots()
    ax.plot(np.degrees(polar_rad), radial_flux_meep, "b-", label="Meep")
    ax.plot(np.degrees(polar_rad), radial_flux_analytic, "r-", label="theory")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("radiation pattern")
    ax.set_title("antenna above PEC ground plane\n using Brillouin-zone integration")
    ax.set_xlim(0, 90.0)
    ax.set_ylim(0, 1.0)
    ax.legend()
    if mp.am_master():
        fig.savefig("antenna_radiation_pattern_1D.png", dpi=150, bbox_inches="tight")
