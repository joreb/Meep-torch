# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/planar_cavity_ldos.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/planar_cavity_ldos/"
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
        _p = _p.with_name("planar_cavity_ldos_meep-" + _p.stem)
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

# Computes the Purcell enhancement factor of an in-plane dipole in a planar
# dielectric cavity with lossless metallic walls. The result is computed in
# cylindrical and 3D coordinates and compared with the analytic result from:
# I. Abram et al., IEEE J. Quantum Electronics, Vol. 34, pp. 71-76 (1998).

# tutorial reference:
# https://meep.readthedocs.io/en/latest/Python_Tutorials/Local_Density_of_States/#planar-cavity-with-lossless-metallic-walls

from typing import Optional

import matplotlib.pyplot as plt
import meep as mp
import numpy as np


# Note: Meep may round the cell dimensions to an integer number of pixels which
# could modify the cavity structure.
RESOLUTION_UM = 71

PML_UM = 0.5
BULK_UM = 6.0
N_CAVITY = 2.4
WAVELENGTH_UM = 1.0
FIELD_DECAY_TOL = 1e-6
FIELD_DECAY_PERIOD = 20

frequency = 1 / WAVELENGTH_UM


def ldos_cyl(cavity_um: Optional[float] = None) -> float:
    """Computes the LDOS of a dipole in a cavity or bulk media in cyl. coords.

    Args:
        cavity_um: thickness of the cavity. If None, bulk media is used.

    Returns:
        The LDOS of the dipole.
    """
    if cavity_um is None:
        cell_z_um = BULK_UM + 2 * PML_UM
        pml_layers = [mp.PML(thickness=PML_UM)]
    else:
        cell_z_um = cavity_um
        pml_layers = [mp.PML(thickness=PML_UM, direction=mp.R)]

    cell_r_um = BULK_UM + PML_UM
    cell_size = mp.Vector3(cell_r_um, 0, cell_z_um)

    # An Er source at r = 0 and m=±1 needs to be slightly offset.
    # https://github.com/NanoComp/meep/issues/2704
    dipole_rpos_um = 1.5 / RESOLUTION_UM

    src_pt = mp.Vector3(dipole_rpos_um, 0, 0)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            component=mp.Er,
            center=src_pt,
        )
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        default_material=mp.Medium(index=N_CAVITY),
    )

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, mp.Er, src_pt, FIELD_DECAY_TOL
        ),
    )

    return sim.ldos_data[0]


def ldos_3d(cavity_um: Optional[float] = None) -> float:
    """Computes the LDOS of a dipole in a cavity or bulk media in 3D coords.

    Args:
        cavity_um: thickness of the cavity. If None, bulk media is used.

    Returns:
        The LDOS of the dipole.
    """
    if cavity_um is None:
        size_z_um = BULK_UM + 2 * PML_UM
        pml_layers = [mp.PML(thickness=PML_UM)]
    else:
        size_z_um = cavity_um
        pml_layers = [
            mp.PML(thickness=PML_UM, direction=mp.X),
            mp.PML(thickness=PML_UM, direction=mp.Y),
        ]

    size_xy_um = BULK_UM + 2 * PML_UM
    cell_size = mp.Vector3(size_xy_um, size_xy_um, size_z_um)

    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.2 * frequency),
            component=mp.Ex,
            center=mp.Vector3(),
        )
    ]

    symmetries = [
        mp.Mirror(direction=mp.X, phase=-1),
        mp.Mirror(direction=mp.Y),
        mp.Mirror(direction=mp.Z),
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=symmetries,
        default_material=mp.Medium(index=N_CAVITY),
    )

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(
            FIELD_DECAY_PERIOD, mp.Ex, mp.Vector3(), FIELD_DECAY_TOL
        ),
    )

    return sim.ldos_data[0]


if __name__ == "__main__":
    ldos_bulk_cyl = ldos_cyl()
    ldos_bulk_3d = ldos_3d()

    cavity_um = np.arange(0.50, 2.55, 0.05)
    vacuum_cavity_um = cavity_um * WAVELENGTH_UM / N_CAVITY

    num_cavity_um = cavity_um.shape[0]
    ldos_cavity_cyl = np.zeros(num_cavity_um)
    ldos_cavity_3d = np.zeros(num_cavity_um)

    for j in range(num_cavity_um):
        ldos_cavity_cyl[j] = ldos_cyl(vacuum_cavity_um[j])
        ldos_cavity_3d[j] = ldos_3d(vacuum_cavity_um[j])
        purcell_cyl = ldos_cavity_cyl[j] / ldos_bulk_cyl
        purcell_3d = ldos_cavity_3d[j] / ldos_bulk_3d
        print(f"purcell:, {cavity_um[j]:.3f}, {purcell_cyl:.6f}, {purcell_3d:.6f}")

    # Purcell enhancement factor (relative to bulk medium)
    purcell_meep_cyl = ldos_cavity_cyl / ldos_bulk_cyl
    purcell_meep_3d = ldos_cavity_3d / ldos_bulk_3d

    # Equation 7 of 1998 reference.
    purcell_theory = 3 * np.fix(cavity_um + 0.5) / (4 * cavity_um) + (
        4 * np.power(np.fix(cavity_um + 0.5), 3) - np.fix(cavity_um + 0.5)
    ) / (16 * np.power(cavity_um, 3))

    if mp.am_master():
        fig, ax = plt.subplots()
        ax.plot(cavity_um, purcell_meep_3d, "b-", label="Meep (3d)")
        ax.plot(cavity_um, purcell_meep_cyl, "r-", label="Meep (cylin.)")
        ax.plot(cavity_um, purcell_theory, "g-", label="theory")
        ax.plot(cavity_um, np.ones(len(cavity_um)), "k--")
        ax.set_xlabel("cavity thickness (in media)")
        ax.set_ylabel("Purcell enhancement factor")
        ax.set_title(
            "in-plane dipole at λ=1.0 μm in a planar cavity\n"
            "with n=2.4 and lossless metallic walls"
        )
        ax.axis([0.5, 2.5, 0.4, 3.1])
        ax.legend()
        fig.savefig(
            "cavity_purcell_factor_vs_thickness.png", dpi=150, bbox_inches="tight"
        )
