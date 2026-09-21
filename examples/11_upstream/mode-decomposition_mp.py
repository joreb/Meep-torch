# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mode-decomposition.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mode-decomposition/"
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
        _p = _p.with_name("mode-decomposition_mp-" + _p.stem)
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

import matplotlib.pyplot as plt

import meep_torch.mp as mp

resolution = 25  # pixels/μm

w1 = 1.0  # width of waveguide 1
w2 = 2.0  # width of waveguide 2
Lw = 10.0  # length of waveguides 1 and 2

# lengths of waveguide taper
Lts = [2**m for m in range(4)]

dair = 3.0  # length of air region
dpml_x = 6.0  # length of PML in x direction
dpml_y = 2.0  # length of PML in y direction

sy = dpml_y + dair + w2 + dair + dpml_y

Si = mp.Medium(epsilon=12.0)

boundary_layers = [mp.PML(dpml_x, direction=mp.X), mp.PML(dpml_y, direction=mp.Y)]

lcen = 6.67  # mode wavelength
fcen = 1 / lcen  # mode frequency

symmetries = [mp.Mirror(mp.Y)]

R_coeffs = []
R_flux = []

for Lt in Lts:
    sx = dpml_x + Lw + Lt + Lw + dpml_x
    cell_size = mp.Vector3(sx, sy, 0)

    src_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.2 * Lw)
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
            center=src_pt,
            size=mp.Vector3(y=sy - 2 * dpml_y),
            eig_match_freq=True,
            eig_parity=mp.ODD_Z + mp.EVEN_Y,
        )
    ]

    # straight waveguide
    vertices = [
        mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
        mp.Vector3(0.5 * sx + 1, 0.5 * w1),
        mp.Vector3(0.5 * sx + 1, -0.5 * w1),
        mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
        sources=sources,
        symmetries=symmetries,
    )

    mon_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.7 * Lw)
    flux = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
    )

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))

    res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
    incident_coeffs = res.alpha
    incident_flux = mp.get_fluxes(flux)
    incident_flux_data = sim.get_flux_data(flux)

    sim.reset_meep()

    # linear taper
    vertices = [
        mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
        mp.Vector3(-0.5 * Lt, 0.5 * w1),
        mp.Vector3(0.5 * Lt, 0.5 * w2),
        mp.Vector3(0.5 * sx + 1, 0.5 * w2),
        mp.Vector3(0.5 * sx + 1, -0.5 * w2),
        mp.Vector3(0.5 * Lt, -0.5 * w2),
        mp.Vector3(-0.5 * Lt, -0.5 * w1),
        mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=boundary_layers,
        geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
        sources=sources,
        symmetries=symmetries,
    )

    flux = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
    )
    sim.load_minus_flux_data(flux, incident_flux_data)

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))

    res2 = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
    taper_coeffs = res2.alpha
    taper_flux = mp.get_fluxes(flux)

    R_coeffs.append(
        abs(taper_coeffs[0, 0, 1]) ** 2 / abs(incident_coeffs[0, 0, 0]) ** 2
    )
    R_flux.append(-taper_flux[0] / incident_flux[0])
    print(f"refl:, {Lt}, {R_coeffs[-1]:.8f}, {R_flux[-1]:.8f}")


if mp.am_master():
    plt.figure()
    plt.loglog(Lts, R_coeffs, "bo-", label="mode decomposition")
    plt.loglog(Lts, R_flux, "ro-", label="Poynting flux")
    plt.loglog(
        Lts,
        [0.005 / Lt**2 for Lt in Lts],
        "k-",
        label=r"quadratic reference (1/Lt$^2$)",
    )
    plt.legend(loc="upper right")
    plt.xlabel("taper length Lt (μm)")
    plt.ylabel("reflectance")
    plt.show()
