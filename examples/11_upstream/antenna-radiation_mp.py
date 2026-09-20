"""Radiation pattern of a dipole antenna in free space.

Tutorial reference:

https://meep.readthedocs.io/en/latest/Python_Tutorials/Near_to_Far_Field_Spectra/#radiation-pattern-of-an-antenna
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/antenna-radiation.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/antenna-radiation/"
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
        _p = _p.with_name("antenna-radiation_mp-" + _p.stem)
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


import math

import matplotlib.pyplot as plt
import meep_torch.mp as mp
import numpy as np


RESOLUTION_UM = 50
PML_UM = 1.0
WAVELENGTH_UM = 1.0
NUM_POLAR = 100
FARFIELD_RADIUS_UM = 1000 * WAVELENGTH_UM
FARFIELD_RESOLUTION_UM = 1
GREENCYL_TOL = 1e-8

frequency = 1 / WAVELENGTH_UM
polar_rad = np.linspace(0, 2 * math.pi, NUM_POLAR)


def radiation_pattern(sim: mp.Simulation, n2f_mon: mp.DftNear2Far) -> np.ndarray:
    """Computes the radiation pattern from the far fields.

    Args:
        sim: a `Simulation` object.
        n2f_mon: a `DftNear2Far` object returned by `Simulation.add_near2far`.

    Returns:
        Array of radial Poynting flux, one for each point on the circumference of
        a circle with angular range of [0, 2π] rad. 0 rad is the +x
        direction and π/2 is +y.
    """
    e_field = np.zeros((NUM_POLAR, 3), dtype=np.complex128)
    h_field = np.zeros((NUM_POLAR, 3), dtype=np.complex128)
    for i in range(NUM_POLAR):
        far_field = sim.get_farfield(
            n2f_mon,
            mp.Vector3(
                FARFIELD_RADIUS_UM * math.cos(polar_rad[i]),
                FARFIELD_RADIUS_UM * math.sin(polar_rad[i]),
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


def antenna_radiation_pattern(dipole_polarization: int) -> np.ndarray:
    """Returns the radiation pattern of a dipole antenna.

    Args:
        dipole_polarization: polarization of the dipole antenna (Ex, Ey, or Ez).

    Returns:
        The radiation pattern as a 1D array.
    """
    if dipole_polarization not in (mp.Ex, mp.Ey, mp.Ez):
        raise ValueError("dipole_polarization must be Ex, Ey, or Ez.")

    cell_um = 4.0
    sxy = PML_UM + cell_um + PML_UM
    cell_size = mp.Vector3(sxy, sxy, 0)
    boundary_layers = [mp.PML(PML_UM)]

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            center=mp.Vector3(),
            component=dipole_polarization,
        )
    ]

    if dipole_polarization == mp.Ex:
        symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)]
    elif dipole_polarization == mp.Ey:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=-1)]
    else:
        symmetries = [mp.Mirror(mp.X, phase=+1), mp.Mirror(mp.Y, phase=+1)]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=sources,
        symmetries=symmetries,
    )

    nearfield_box = sim.add_near2far(
        frequency,
        0,
        1,
        mp.Near2FarRegion(
            center=mp.Vector3(0, 0.5 * cell_um), size=mp.Vector3(cell_um, 0)
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0, -0.5 * cell_um), size=mp.Vector3(cell_um, 0), weight=-1
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(0.5 * cell_um, 0), size=mp.Vector3(0, cell_um)
        ),
        mp.Near2FarRegion(
            center=mp.Vector3(-0.5 * cell_um, 0), size=mp.Vector3(0, cell_um), weight=-1
        ),
    )

    flux_box = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(center=mp.Vector3(0, 0.5 * cell_um), size=mp.Vector3(cell_um, 0)),
        mp.FluxRegion(
            center=mp.Vector3(0, -0.5 * cell_um), size=mp.Vector3(cell_um, 0), weight=-1
        ),
        mp.FluxRegion(center=mp.Vector3(0.5 * cell_um, 0), size=mp.Vector3(0, cell_um)),
        mp.FluxRegion(
            center=mp.Vector3(-0.5 * cell_um, 0), size=mp.Vector3(0, cell_um), weight=-1
        ),
    )

    sim.run(until_after_sources=mp.stop_when_dft_decayed())

    flux_near = mp.get_fluxes(flux_box)[0]

    flux_far = (
        nearfield_box.flux(
            mp.Y,
            mp.Volume(
                center=mp.Vector3(0, FARFIELD_RADIUS_UM, 0),
                size=mp.Vector3(2 * FARFIELD_RADIUS_UM, 0, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
        - nearfield_box.flux(
            mp.Y,
            mp.Volume(
                center=mp.Vector3(0, -FARFIELD_RADIUS_UM, 0),
                size=mp.Vector3(2 * FARFIELD_RADIUS_UM, 0, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
        + nearfield_box.flux(
            mp.X,
            mp.Volume(
                center=mp.Vector3(FARFIELD_RADIUS_UM, 0, 0),
                size=mp.Vector3(0, 2 * FARFIELD_RADIUS_UM, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
        - nearfield_box.flux(
            mp.X,
            mp.Volume(
                center=mp.Vector3(-FARFIELD_RADIUS_UM, 0, 0),
                size=mp.Vector3(0, 2 * FARFIELD_RADIUS_UM, mp.inf),
            ),
            FARFIELD_RESOLUTION_UM,
        )[0]
    )

    radial_flux = radiation_pattern(sim, nearfield_box)
    flux_radiation_pattern = np.trapz(radial_flux * FARFIELD_RADIUS_UM, x=polar_rad)

    print(
        f"flux:, {mp.component_name(dipole_polarization)} (dipole), "
        f"{flux_near:.6f} (near), {flux_far:.6f} (far), "
        f"{flux_radiation_pattern:.6f} (radiation pattern)"
    )

    return radial_flux


if __name__ == "__main__":
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(18, 6), ncols=3)

    for i, dipole_polarization in enumerate([mp.Ex, mp.Ey, mp.Ez]):
        radial_flux = antenna_radiation_pattern(dipole_polarization)

        # Analytic formulas for the radiation pattern of an electric dipole
        # in vacuum. From Section 4.2 "Infinitesimal Dipole"
        # of Antenna Theory: Analysis and Design, 4th Edition (2016) by C. Balanis.
        if dipole_polarization == mp.Ex:
            flux_theory = np.sin(polar_rad) ** 2
        elif dipole_polarization == mp.Ey:
            flux_theory = np.cos(polar_rad) ** 2
        elif dipole_polarization == mp.Ez:
            flux_theory = np.ones((NUM_POLAR,))

        ax[i].plot(polar_rad, radial_flux / max(radial_flux), "b-", label="Meep")
        ax[i].plot(polar_rad, flux_theory, "r--", label="theory")
        ax[i].set_rmax(1)
        ax[i].set_rticks([0, 0.5, 1])
        ax[i].grid(True)
        ax[i].set_rlabel_position(22)
        ax[i].set_title(
            r"$\mathcal{J}$"
            f"$_{{{mp.direction_name(mp.component_direction(dipole_polarization))}}}$"
            " dipole"
        )
        ax[i].legend(loc="upper right")

    fig.subplots_adjust(wspace=0.2)
    if mp.am_master():
        fig.savefig(
            f"antenna_radiation_pattern.png",
            dpi=150,
            bbox_inches="tight",
        )
