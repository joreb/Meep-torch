"""Computes the extraction efficiency in 3D and cylindrical coordinates.

Verifies that the extraction efficiency of a point dipole in a dielectric
layer above a lossless metallic ground plane computed in two different
coordinate systems agree.
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/extraction_eff_ldos.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/extraction_eff_ldos/"
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
        _p = _p.with_name("extraction_eff_ldos_meep-" + _p.stem)
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


import matplotlib.pyplot as plt
import meep as mp
import numpy as np

resolution = 80  # pixels/μm
dpml = 0.5  # thickness of PML
dair = 1.0  # thickness of air padding
L = 6.0  # length of non-PML region
n = 2.4  # refractive index of surrounding medium
wvl = 1.0  # wavelength (in vacuum)

fcen = 1 / wvl  # center frequency of source/monitor

# runtime termination criteria
tol = 1e-8


def extraction_eff_cyl(dmat: float, h: float) -> float:
    """Computes the extraction efficiency in cylindrical coordinates.

    Args:
      dmat: thickness of dielectric layer.
      h: height of dipole above ground plane as fraction of dmat.

    Returns:
      The extraction efficiency of the dipole within the dielecric layer.
    """
    sr = L + dpml
    sz = dmat + dair + dpml
    cell_size = mp.Vector3(sr, 0, sz)

    boundary_layers = [
        mp.PML(dpml, direction=mp.R),
        mp.PML(dpml, direction=mp.Z, side=mp.High),
    ]

    src_cmpt = mp.Er

    # Because (1) Er is not defined at r=0 on the Yee grid, and (2) there
    # seems to be a bug in the interpolation of an Er point source at r=0,
    # the source is placed at r=~Δr (just outside the first voxel).
    # This incurs a small error which decreases linearly with resolution.
    # Ref: https://github.com/NanoComp/meep/issues/2704
    src_pt = mp.Vector3(1.5 / resolution, 0, -0.5 * sz + h * dmat)

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            component=src_cmpt,
            center=src_pt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=n),
            center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * dmat),
            size=mp.Vector3(mp.inf, mp.inf, dmat),
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        dimensions=mp.CYLINDRICAL,
        m=-1,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
    )

    flux_air = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0.5 * L, 0, 0.5 * sz - dpml),
            size=mp.Vector3(L, 0, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(L, 0, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(0, 0, dair),
        ),
    )

    sim.run(
        mp.dft_ldos(fcen, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(20, src_cmpt, src_pt, tol),
    )

    out_flux = mp.get_fluxes(flux_air)[0]
    if src_pt.x == 0:
        dV = np.pi / (resolution**3)
    else:
        dV = 2 * np.pi * src_pt.x / (resolution**2)
    total_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * dV
    ext_eff = out_flux / total_flux
    print(f"extraction efficiency (cyl):, " f"{dmat:.4f}, {h:.4f}, {ext_eff:.6f}")

    return ext_eff


def extraction_eff_3D(dmat: float, h: float) -> float:
    """Computes the extraction efficiency in 3D Cartesian coordinates.

    Args:
      dmat: thickness of dielectric layer.
      h: height of dipole above ground plane as fraction of dmat.

    Returns:
      The extraction efficiency of the dipole within the dielecric layer.
    """
    sxy = L + 2 * dpml
    sz = dmat + dair + dpml
    cell_size = mp.Vector3(sxy, sxy, sz)

    symmetries = [
        mp.Mirror(direction=mp.X, phase=-1),
        mp.Mirror(direction=mp.Y),
    ]

    boundary_layers = [
        mp.PML(dpml, direction=mp.X),
        mp.PML(dpml, direction=mp.Y),
        mp.PML(dpml, direction=mp.Z, side=mp.High),
    ]

    src_cmpt = mp.Ex
    src_pt = mp.Vector3(0, 0, -0.5 * sz + h * dmat)
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            component=src_cmpt,
            center=src_pt,
        )
    ]

    geometry = [
        mp.Block(
            material=mp.Medium(index=n),
            center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * dmat),
            size=mp.Vector3(mp.inf, mp.inf, dmat),
        )
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        sources=sources,
        geometry=geometry,
        symmetries=symmetries,
    )

    flux_air = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(
            center=mp.Vector3(0, 0, 0.5 * sz - dpml),
            size=mp.Vector3(L, L, 0),
        ),
        mp.FluxRegion(
            center=mp.Vector3(0.5 * L, 0, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(0, L, dair),
        ),
        mp.FluxRegion(
            center=mp.Vector3(-0.5 * L, 0, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(0, L, dair),
            weight=-1.0,
        ),
        mp.FluxRegion(
            center=mp.Vector3(0, 0.5 * L, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(L, 0, dair),
        ),
        mp.FluxRegion(
            center=mp.Vector3(0, -0.5 * L, 0.5 * sz - dpml - 0.5 * dair),
            size=mp.Vector3(L, 0, dair),
            weight=-1.0,
        ),
    )

    sim.run(
        mp.dft_ldos(fcen, 0, 1),
        until_after_sources=mp.stop_when_fields_decayed(20, src_cmpt, src_pt, tol),
    )

    out_flux = mp.get_fluxes(flux_air)[0]
    dV = 1 / (resolution**3)
    total_flux = -np.real(sim.ldos_Fdata[0] * np.conj(sim.ldos_Jdata[0])) * dV
    ext_eff = out_flux / total_flux
    print(f"extraction efficiency (3D):, {dmat:.4f}, {h:.4f}, {ext_eff:.6f}")

    return ext_eff


if __name__ == "__main__":
    layer_thickness = 0.7 * wvl / n
    dipole_height = np.linspace(0.1, 0.9, 21)

    exteff_cyl = np.zeros(len(dipole_height))
    exteff_3D = np.zeros(len(dipole_height))
    for j in range(len(dipole_height)):
        exteff_cyl[j] = extraction_eff_cyl(layer_thickness, dipole_height[j])
        exteff_3D[j] = extraction_eff_3D(layer_thickness, dipole_height[j])

    plt.plot(dipole_height, exteff_cyl, "bo-", label="cylindrical")
    plt.plot(dipole_height, exteff_3D, "ro-", label="3D Cartesian")
    plt.xlabel("height of dipole above ground plane (fraction of layer thickness)")
    plt.ylabel("extraction efficiency")
    plt.legend()

    if mp.am_master():
        plt.savefig("extraction_eff_vs_dipole_height.png", dpi=150, bbox_inches="tight")
