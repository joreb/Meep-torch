"""Tutorial example for dipole current sources in cylindrical coordinates.

tutorial reference:
https://meep.readthedocs.io/en/latest/Python_Tutorials/Cylindrical_Coordinates/#nonaxisymmetric-dipole-sources
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/point_dipole_cyl.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/point_dipole_cyl/"
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
        _p = _p.with_name("point_dipole_cyl_mp-" + _p.stem)
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


from typing import Tuple

import meep_torch.mp as mp
import numpy as np


RESOLUTION_UM = 50
WAVELENGTH_UM = 1.0
N_SLAB = 2.4
SLAB_THICKNESS_UM = 0.7 * WAVELENGTH_UM / N_SLAB


def dipole_in_slab(zpos: float, rpos_um: float, m: int) -> Tuple[float, float]:
    """Computes the flux from a dipole in a slab.

    Args:
      zpos: position of dipole as a fraction of layer thickness.
      rpos_um: position of source in radial direction.
      m: angular φ dependence of the fields exp(imφ).

    Returns:
      A 2-tuple of the radiated and total flux.
    """
    pml_um = 1.0  # thickness of PML
    padding_um = 1.0  # thickness of air padding
    r_um = 20.0  # length of cell in r

    frequency = 1 / WAVELENGTH_UM  # center frequency of source/monitor

    # runtime termination criteria
    flux_decay_threshold = 1e-4

    size_r = r_um + pml_um
    size_z = SLAB_THICKNESS_UM + padding_um + pml_um
    cell_size = mp.Vector3(size_r, 0, size_z)

    boundary_layers = [
        mp.PML(pml_um, direction=mp.R),
        mp.PML(pml_um, direction=mp.Z, side=mp.High),
    ]

    src_pt = mp.Vector3(rpos_um, 0, -0.5 * size_z + zpos * SLAB_THICKNESS_UM)
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency, fwidth=0.05 * frequency),
            component=mp.Er,
            center=src_pt,
        ),
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=N_SLAB),
            center=mp.Vector3(0, 0, -0.5 * size_z + 0.5 * SLAB_THICKNESS_UM),
            size=mp.Vector3(mp.inf, mp.inf, SLAB_THICKNESS_UM),
        )
    ]

    sim = mp.Simulation(
        resolution=RESOLUTION_UM,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=m,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
        force_complex_fields=True,
    )

    flux_mon = sim.add_flux(
        frequency,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * r_um, 0, 0.5 * size_z - pml_um),
            size=mp.Vector3(r_um, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(r_um, 0, 0.5 * size_z - pml_um - 0.5 * padding_um),
            size=mp.Vector3(0, 0, padding_um),
        ),
    )

    sim.run(
        mp.dft_ldos(frequency, 0, 1),
        until_after_sources=mp.stop_when_dft_decayed(tol=flux_decay_threshold),
    )

    radiated_flux = mp.get_fluxes(flux_mon)[0]

    # volume of the ring current source
    delta_vol = 2 * np.pi * rpos_um / (RESOLUTION_UM**2)

    # total flux from point source via LDOS
    source_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * delta_vol

    print(
        f"flux-cyl:, {rpos_um:.2f}, {m:3d}, " f"{source_flux:.6f}, {radiated_flux:.6f}"
    )

    return radiated_flux, source_flux


if __name__ == "__main__":
    dipole_height = 0.5

    # An Er source at r = 0 needs to be slightly offset.
    # https://github.com/NanoComp/meep/issues/2704
    dipole_rpos_um = 1.5 / RESOLUTION_UM

    # Er source at r = 0 requires a single simulation with m = ±1.
    m = 1
    radiated_flux, source_flux = dipole_in_slab(
        dipole_height,
        dipole_rpos_um,
        m,
    )
    extraction_efficiency = radiated_flux / source_flux
    print(f"exteff:, {dipole_rpos_um}, {extraction_efficiency:.6f}")

    # Er source at r > 0 requires Fourier-series expansion of φ.

    # Threshold flux to determine when to truncate expansion.
    flux_decay_threshold = 1e-2

    dipole_rpos_um = [3.5, 6.7, 9.5]
    for rpos_um in dipole_rpos_um:
        source_flux_total = 0
        radiated_flux_total = 0
        radiated_flux_max = 0
        m = 0
        while True:
            radiated_flux, source_flux = dipole_in_slab(
                dipole_height,
                rpos_um,
                m,
            )
            radiated_flux_total += radiated_flux * (1 if m == 0 else 2)
            source_flux_total += source_flux * (1 if m == 0 else 2)

            if radiated_flux > radiated_flux_max:
                radiated_flux_max = radiated_flux

            if m > 0 and (radiated_flux / radiated_flux_max) < flux_decay_threshold:
                break
            else:
                m += 1

        extraction_efficiency = radiated_flux_total / source_flux_total
        print(f"exteff:, {rpos_um}, {extraction_efficiency:.6f}")
