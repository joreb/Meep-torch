"""Radiation pattern of a dipole antenna above a ground plane.

Tutorial reference:

https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#antenna-above-a-perfect-electric-conductor-ground-plane
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/antenna_pec_ground_plane.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/antenna_pec_ground_plane/"
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
        _p = _p.with_name("antenna_pec_ground_plane_meep-" + _p.stem)
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


import math

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


RESOLUTION_UM = 100
PML_UM = 0.5
BULK_UM = 6.0
N_BACKGROUND = 1.2
ANTENNA_HEIGHT_UM = 1.25
WAVELENGTH_UM = 0.65
NUM_POLAR = 50
FARFIELD_RADIUS_UM = 1000 * WAVELENGTH_UM
GREENCYL_TOL = 1e-8

frequency = 1 / WAVELENGTH_UM
polar_rad = np.linspace(0, 0.5 * math.pi, NUM_POLAR)


def radiation_pattern(sim: mp.Simulation, n2f_mon: mp.DftNear2Far) -> np.ndarray:
    """Computes the radiation pattern from the far fields.

    Args:
        sim: a `Simulation` object.
        n2f_mon: a `DftNear2Far` object returned by `Simulation.add_near2far`.

    Returns:
        Array of radial Poynting flux, one for each point on the circumference of
        a quarter circle with angular range of [0, π/2] rad. 0 rad is the +y
        direction and π/2 is +x.
    """
    e_field = np.zeros((NUM_POLAR, 3), dtype=np.complex128)
    h_field = np.zeros((NUM_POLAR, 3), dtype=np.complex128)
    for i in range(NUM_POLAR):
        far_field = sim.get_farfield(
            n2f_mon,
            mp.Vector3(
                FARFIELD_RADIUS_UM * math.sin(polar_rad[i]),
                FARFIELD_RADIUS_UM * math.cos(polar_rad[i]),
                0,
            ),
            GREENCYL_TOL,
        )
        e_field[i, :] = [far_field[j] for j in range(3)]
        h_field[i, :] = [far_field[j + 3] for j in range(3)]

    flux_x = np.real(
        np.conj(e_field[:, 1]) * h_field[:, 2] - np.conj(e_field[:, 2]) * h_field[:, 1]
    )
    flux_y = np.real(
        np.conj(e_field[:, 2]) * h_field[:, 0] - np.conj(e_field[:, 0]) * h_field[:, 2]
    )
    flux_r = np.sqrt(np.square(flux_x) + np.square(flux_y))

    return flux_r


def antenna_above_ground_plane(dipole_polarization: int) -> np.ndarray:
    """Returns the radiation pattern of an antenna above a ground plane.

    Args:
        dipole_polarization: polarization of the dipole antenna (Ez or Hz).

    Returns:
        The radiation pattern as a 1D array.
    """
    if dipole_polarization not in (mp.Ez, mp.Hz):
        raise ValueError("dipole_polarization must be either Ez or Hz")

    sxy = PML_UM + BULK_UM + PML_UM
    cell_size = mp.Vector3(sxy, sxy, 0)
    boundary_layers = [mp.PML(PML_UM)]

    # The near-to-far field transformation feature can only be applied
    # to structures which can be fully enclosed. It cannot be applied to
    # the ground plane which extends to infinity. As a workaround, we use
    # two antennas of opposite sign surrounded by a single near2far box
    # which encloses both antennas. We then use an odd mirror symmetry to
    # divide the computational cell in half which is effectively
    # equivalent to a PEC boundary condition on one side.
    # Note: This setup means that the radiation pattern can only
    # be measured in the top half above the dipole antenna.
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            component=dipole_polarization,
            center=mp.Vector3(0, ANTENNA_HEIGHT_UM),
        ),
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            component=dipole_polarization,
            center=mp.Vector3(0, -ANTENNA_HEIGHT_UM),
            amplitude=-1.0 if dipole_polarization == mp.Ez else +1.0,
        ),
    ]

    if dipole_polarization == mp.Hz:
        symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y)]
    else:
        symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y, phase=-1)]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=sources,
        symmetries=symmetries,
        default_material=mp.Medium(index=N_BACKGROUND),
    )

    nearfield_box = sim.add_near2far(
        frequency,
        0,
        1,
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0.5 * BULK_UM), size=mp.Vector3(BULK_UM, 0)
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -0.5 * BULK_UM), size=mp.Vector3(BULK_UM, 0), weight=-1
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0.5 * BULK_UM, 0), size=mp.Vector3(0, BULK_UM)
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(-0.5 * BULK_UM, 0), size=mp.Vector3(0, BULK_UM), weight=-1
        ),
    )

    fig, ax = plt.subplots()
    sim.plot2D(ax=ax)
    if mp.am_master():
        fig.savefig(
            "antenna_above_ground_plane_layout.png", dpi=150, bbox_inches="tight"
        )

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    return radiation_pattern(sim, nearfield_box)


if __name__ == "__main__":
    dipole_polarization = mp.Ez
    radial_flux_meep = antenna_above_ground_plane(dipole_polarization)
    # Normalize radiation pattern to lie in range [0, 1].
    radial_flux_meep = radial_flux_meep / np.max(radial_flux_meep)

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

    rel_err = np.linalg.norm(radial_flux_meep - radial_flux_analytic) / np.linalg.norm(
        radial_flux_analytic
    )
    print(f"error:, {rel_err:.6f}")

    fig, ax = plt.subplots()
    ax.plot(np.degrees(polar_rad), radial_flux_meep, "b-", label="Meep")
    ax.plot(np.degrees(polar_rad), radial_flux_analytic, "r-", label="theory")
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("radiation pattern")
    ax.set_title(
        f"antenna with {'E' if dipole_polarization==mp.Ez else 'H'}$_z$ "
        "polarization above PEC ground plane"
    )
    ax.set_xlim(0, 90.0)
    ax.set_ylim(0, 1.0)
    ax.legend()
    if mp.am_master():
        fig.savefig(
            "antenna_above_ground_plane_radiation_pattern.png",
            dpi=150,
            bbox_inches="tight",
        )
