"""Radiation pattern of a dipole in vacuum using Brillouin-zone integration."""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/dipole_in_vacuum_1D.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/dipole_in_vacuum_1D/"
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
        _p = _p.with_name("dipole_in_vacuum_1D_mp-" + _p.stem)
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
from enum import Enum
from typing import Tuple
import math

import matplotlib.pyplot as plt
import meep_torch.mp as mp
import numpy as np


RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
NUM_POLAR = 30
NUM_AZIMUTH = 50
FIELD_DECAY_THRESHOLD = 1e-6

frequency = 1 / WAVELENGTH_UM


def planewave_in_vacuum(dipole_pol: str, kx: float, ky: float, kz: float) -> float:
    """
    Returns the Poynting flux of a linearly polarized planewave in vacuum.

    Args:
        dipole_pol: the polarization of the electric dipole. Either x or y.
        kx, ky, kz: the wavevector components of the planewave.

    Returns:
        The Poynting flux in z.
    """
    pml_um = 1.0
    air_um = 10.0
    size_z_um = pml_um + air_um + pml_um
    cell_size = mp.Vector3(0, 0, size_z_um)
    pml_layers = [mp.PML(thickness=pml_um, direction=mp.Z)]
    k_point = mp.Vector3(kx, ky, kz)

    if dipole_pol == "x":
        src_cmpt = mp.Ex
    elif dipole_pol == "y":
        src_cmpt = mp.Ey

    src_pt = mp.Vector3(0, 0, -0.5 * air_um)
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
        cell_size=cell_size,
        sources=sources,
        boundary_layers=pml_layers,
        k_point=k_point,
    )

    mon_pt = mp.Vector3(0, 0, 0.5 * air_um)
    dft_flux_z = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(center=mon_pt, size=mp.Vector3(), direction=mp.Z),
    )

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            25, src_cmpt, mon_pt, FIELD_DECAY_THRESHOLD
        )
    )

    flux_z = mp.get_fluxes(dft_flux_z)[0]

    return flux_z


def spherical_to_cartesian(polar_rad, azimuth_rad) -> Tuple[float, float, float]:
    """Converts a point on the unit sphere from spherical to Cartesian coords.

    Args:
        polar_rad: polar angle of the point. 0° is +z.
        azimuth_rad: azimuthal angle of the point. 0° is +x.

    Returns:
        The x,y,z coordinates of the point as a 3-tuple.
    """
    x = np.sin(polar_rad) * np.cos(azimuth_rad)
    y = np.sin(polar_rad) * np.sin(azimuth_rad)
    z = np.cos(polar_rad)

    return x, y, z


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dipole_pol",
        type=str,
        choices=["x", "y"],
        help="polarization of the electric dipole (x or y)",
    )
    args = parser.parse_args()

    # Radial flux is defined on the surface of a hemisphere.
    polar_rad = np.linspace(0, 0.5 * np.pi, NUM_POLAR)
    azimuth_rad = np.linspace(0, 2 * np.pi, NUM_AZIMUTH)
    radial_flux = np.zeros((NUM_POLAR, NUM_AZIMUTH))

    for i in range(NUM_POLAR):
        for j in range(NUM_AZIMUTH):
            rx, ry, rz = spherical_to_cartesian(polar_rad[i], azimuth_rad[j])
            # Specify the components of the wavevector of the outgoing
            # planewave in vacuum.
            kx = frequency * rx
            ky = frequency * ry
            kz = frequency * rz

            # Skip wavevectors which are close to the light cone
            # due to poor absorption by PML (i.e. glancing-angle waves).
            if np.sqrt(kx**2 + ky**2) > (0.95 * frequency):
                continue

            flux_z = planewave_in_vacuum(args.dipole_pol, kx, ky, kz)
            radial_flux[i, j] = rz * flux_z

    if mp.am_master():
        np.savez(
            "dipole_radiation_pattern.npz",
            NUM_AZIMUTH=NUM_AZIMUTH,
            NUM_POLAR=NUM_POLAR,
            RESOLUTION_UM=RESOLUTION_UM,
            WAVELENGTH_UM=WAVELENGTH_UM,
            azimuth_rad=azimuth_rad,
            dipole_pol=args.dipole_pol,
            polar_rad=polar_rad,
            radial_flux=radial_flux,
        )
