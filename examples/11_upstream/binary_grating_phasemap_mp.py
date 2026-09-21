# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/binary_grating_phasemap.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/binary_grating_phasemap/"
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
        _p = _p.with_name("binary_grating_phasemap_mp-" + _p.stem)
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

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

import meep_torch.mp as mp

resolution = 60  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width
nfreq = 21  # number of frequency bins

k_point = mp.Vector3(0, 0, 0)

glass = mp.Medium(index=1.5)


def grating(gp, gh, gdc, oddz):
    sx = dpml + dsub + gh + dpad + dpml
    sy = gp

    cell_size = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

    src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub, 0, 0)
    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez if oddz else mp.Hz,
            center=src_pt,
            size=mp.Vector3(0, sy, 0),
        )
    ]

    symmetries = [mp.Mirror(mp.Y, phase=+1 if oddz else -1)]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        k_point=k_point,
        default_material=glass,
        sources=sources,
        symmetries=symmetries,
    )

    mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
    flux_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=100)

    input_flux = mp.get_fluxes(flux_mon)

    sim.reset_meep()

    geometry = [
        mp.Block(
            material=glass,
            size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
            center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub), 0, 0),
        ),
        mp.Block(
            material=glass,
            size=mp.Vector3(gh, gdc * gp, mp.inf),
            center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh, 0, 0),
        ),
    ]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        k_point=k_point,
        sources=sources,
        symmetries=symmetries,
    )

    mode_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
    )

    sim.run(until_after_sources=300)

    freqs = mp.get_eigenmode_freqs(mode_mon)
    res = sim.get_eigenmode_coefficients(
        mode_mon, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y if oddz else mp.EVEN_Z + mp.ODD_Y
    )
    coeffs = res.alpha

    mode_wvl = [1 / freqs[nf] for nf in range(nfreq)]
    mode_tran = [abs(coeffs[0, nf, 0]) ** 2 / input_flux[nf] for nf in range(nfreq)]
    mode_phase = [np.angle(coeffs[0, nf, 0]) for nf in range(nfreq)]

    return mode_wvl, mode_tran, mode_phase


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-gp", type=float, default=0.35, help="grating periodicity (default: 0.35 μm)"
    )
    parser.add_argument(
        "-gh", type=float, default=0.6, help="grating height (default: 0.6 μm)"
    )
    parser.add_argument(
        "-oddz", action="store_true", default=False, help="oddz? (default: False)"
    )
    args = parser.parse_args()

    gdc = np.arange(0.1, 1.0, 0.1)
    mode_tran = np.empty((gdc.size, nfreq))
    mode_phase = np.empty((gdc.size, nfreq))
    for n in range(gdc.size):
        mode_wvl, mode_tran[n, :], mode_phase[n, :] = grating(
            args.gp, args.gh, gdc[n], args.oddz
        )

    plt.figure(dpi=150)

    plt.subplot(1, 2, 1)
    plt.pcolormesh(
        mode_wvl,
        gdc,
        mode_tran,
        cmap="hot_r",
        shading="gouraud",
        vmin=0,
        vmax=mode_tran.max(),
    )
    plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
    plt.xlabel("wavelength (μm)")
    plt.xticks(list(np.arange(wvl_min, wvl_max + 0.1, 0.1)))
    plt.ylabel("grating duty cycle")
    plt.yticks(list(np.arange(gdc[0], gdc[-1] + 0.1, 0.1)))
    plt.title("transmittance")
    cbar = plt.colorbar()
    cbar.set_ticks(list(np.arange(0, 1.2, 0.2)))
    cbar.set_ticklabels([f"{t:.1f}" for t in np.arange(0, 1.2, 0.2)])

    plt.subplot(1, 2, 2)
    plt.pcolormesh(
        mode_wvl,
        gdc,
        mode_phase,
        cmap="RdBu",
        shading="gouraud",
        vmin=mode_phase.min(),
        vmax=mode_phase.max(),
    )
    plt.axis([wvl_min, wvl_max, gdc[0], gdc[-1]])
    plt.xlabel("wavelength (μm)")
    plt.xticks(list(np.arange(wvl_min, wvl_max + 0.1, 0.1)))
    plt.ylabel("grating duty cycle")
    plt.yticks(list(np.arange(gdc[0], gdc[-1] + 0.1, 0.1)))
    plt.title("phase (radians)")
    cbar = plt.colorbar()
    cbar.set_ticks(list(range(-3, 4)))
    cbar.set_ticklabels([f"{t:.1f}" for t in range(-3, 4)])

    plt.tight_layout()
    plt.show()
