"""
1d simulation of a plane wave propagating through a Kerr medium
and generating the third-harmonic frequency component.

ref: https://meep.readthedocs.io/en/latest/Python_Tutorials/Third_Harmonic_Generation/
"""
# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/3rd-harm-1d.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/3rd-harm-1d/"
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
        _p = _p.with_name("3rd-harm-1d_meep-" + _p.stem)
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


from typing import Tuple, List, Union
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import meep as mp


def third_harmonic_generation(
    k: float, amp: float = 1.0, nfreq: int = 10, flux_spectrum: bool = True
) -> Union[Tuple[List[float], List[float]], Tuple[float, float]]:

    """Computes the transmission spectrum of a plane wave propagating
       through a Kerr medium.

    Args:
      k: strength of Kerr susceptibility.
      amp: amplitude of the incident planewave.
      nfreq: number of frequencies in flux spectrum.
      flux_spectrum: compute the flux spectrum over broad bandwidth (True) or
                     just the two harmonic frequencies ω and 3ω (False).

    Returns:
      The frequencies and transmitted flux over the flux spectrum or
      the transmitted flux at the harmonic frequencies ω and 3ω.
    """

    sz = 100  # size of cell in z direction
    fcen = 1 / 3.0  # center frequency of source
    df = fcen / 20.0  # frequency width of source
    dpml = 1.0  # PML thickness

    # We'll use an explicitly 1d simulation.  Setting dimensions=1 will actually
    # result in faster execution than just using two no-size dimensions.  However,
    # in this case Meep requires us to use E in the x direction (and H in y),
    # and our one no-size dimension must be z.
    dimensions = 1
    cell = mp.Vector3(0, 0, sz)
    pml_layers = [mp.PML(dpml)]
    resolution = 25

    # to put the same material in all space, we can just set the default material
    # and pass it to the Simulation constructor
    default_material = mp.Medium(index=1, chi3=k)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, -0.5 * sz + dpml),
            amplitude=amp,
        )
    ]

    # frequency range for flux calculation
    fmin = fcen / 2.0
    fmax = fcen * 4

    sim = mp.Simulation(
        cell_size=cell,
        sources=sources,
        boundary_layers=pml_layers,
        default_material=default_material,
        resolution=resolution,
        dimensions=dimensions,
    )

    mon_pt = mp.Vector3(0, 0, 0.5 * sz - dpml - 0.5)

    if flux_spectrum:
        trans = sim.add_flux(
            0.5 * (fmin + fmax),
            fmax - fmin,
            nfreq,
            mp.FluxRegion(mon_pt),
        )
    else:
        trans1 = sim.add_flux(fcen, 0, 1, mp.FluxRegion(mon_pt))
        trans3 = sim.add_flux(3 * fcen, 0, 1, mp.FluxRegion(mon_pt))

    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mon_pt, 1e-6))

    if flux_spectrum:
        freqs = mp.get_flux_freqs(trans)
        trans_flux = mp.get_fluxes(trans)
        return freqs, trans_flux
    else:
        print(
            f"harmonics:, {k}, {amp}, {mp.get_fluxes(trans1)[0]}, "
            f"{mp.get_fluxes(trans3)[0]}"
        )
        return mp.get_fluxes(trans1)[0], mp.get_fluxes(trans3)[0]


if __name__ == "__main__":
    # Part 1: plot transmitted power spectrum for several values of χ(3).
    nfreq = 400
    logk = range(-3, 1)
    tflux = np.zeros((nfreq, len(logk)))
    for i, lk in enumerate(logk):
        freqs, tflux[:, i] = third_harmonic_generation(
            10**lk, nfreq=nfreq, flux_spectrum=True
        )

    fig, ax = plt.subplots()
    ax.semilogy(freqs, tflux[:, 0], "bo-", label=r"$\chi^{(3)}$=0.001")
    ax.semilogy(freqs, tflux[:, 1], "ro-", label=r"$\chi^{(3)}$=0.01")
    ax.semilogy(freqs, tflux[:, 2], "go-", label=r"$\chi^{(3)}$=0.1")
    ax.semilogy(freqs, tflux[:, 3], "co-", label=r"$\chi^{(3)}$=1")
    ax.set_xlabel("frequency")
    ax.set_ylabel("transmitted power (a.u.)")
    ax.set_xlim(0.2, 1.2)
    ax.set_ylim(1e-15, 1e2)
    ax.legend()
    ax.grid(True)
    fig.savefig(
        "transmitted_power_vs_frequency_vary_chi3.png", dpi=150, bbox_inches="tight"
    )

    # Part 2: plot transmittance vs. χ(3) for frequencies ω and 3ω.
    logk = np.arange(-6.0, 0.2, 0.2)
    first_order = np.zeros(len(logk))
    third_order = np.zeros(len(logk))
    for i, lk in enumerate(logk):
        first_order[i], third_order[i] = third_harmonic_generation(
            10**lk, flux_spectrum=False
        )

    input_flux = first_order[0]
    fig, ax = plt.subplots()
    ax.loglog(10**logk, first_order / input_flux, "ro-", label=r"$\omega$")
    ax.loglog(10**logk, third_order / input_flux, "bo-", label=r"$3\omega$")
    ax.loglog(10**logk, (10**logk) ** 2, "k-", label="quadratic line")
    ax.set_xlabel(r"$\chi^{(3)}$")
    ax.set_ylabel("transmission / incident power")
    ax.legend()
    ax.grid(True, "both")
    fig.savefig("transmittance_vs_chi3.png", dpi=150, bbox_inches="tight")
