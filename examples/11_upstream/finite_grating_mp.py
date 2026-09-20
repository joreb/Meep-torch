# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/finite_grating.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/finite_grating/"
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
        _p = _p.with_name("finite_grating_mp-" + _p.stem)
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
import numpy as np

import meep_torch.mp as mp

# True:  plot the scattered fields in the extended air region adjacent to the grating
# False: plot the diffraction spectra based on a 1d cross section of the scattered fields
field_profile = True

resolution = 50  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dpad = 1.0  # flat-surface padding
gp = 1.0  # grating periodicity
gh = 0.5  # grating height
gdc = 0.5  # grating duty cycle
num_cells = 5  # number of grating unit cells

# air region thickness adjacent to grating
dair = 10 if field_profile else dpad

wvl = 0.5  # center wavelength
fcen = 1 / wvl  # center frequency

k_point = mp.Vector3()

glass = mp.Medium(index=1.5)

pml_layers = [mp.PML(thickness=dpml)]

symmetries = [mp.Mirror(mp.Y)]

sx = dpml + dsub + gh + dair + dpml
sy = dpml + dpad + num_cells * gp + dpad + dpml
cell_size = mp.Vector3(sx, sy)

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=0.2 * fcen, is_integrated=True),
        component=mp.Ez,
        center=src_pt,
        size=mp.Vector3(y=sy),
    )
]

geometry = [
    mp.Block(
        material=glass,
        size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
        center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub)),
    )
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

mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dair)
near_fields = sim.add_dft_fields(
    [mp.Ez],
    fcen,
    0,
    1,
    center=mon_pt,
    size=mp.Vector3(dair if field_profile else 0, sy - 2 * dpml),
)

sim.run(until_after_sources=100)

flat_dft = sim.get_dft_array(near_fields, mp.Ez, 0)

sim.reset_meep()

geometry.extend(
    mp.Block(
        material=glass,
        size=mp.Vector3(gh, gdc * gp, mp.inf),
        center=mp.Vector3(
            -0.5 * sx + dpml + dsub + 0.5 * gh, -0.5 * sy + dpml + dpad + (j + 0.5) * gp
        ),
    )
    for j in range(num_cells)
)
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

near_fields = sim.add_dft_fields(
    [mp.Ez],
    fcen,
    0,
    1,
    center=mon_pt,
    size=mp.Vector3(dair if field_profile else 0, sy - 2 * dpml),
)

sim.run(until_after_sources=100)

grating_dft = sim.get_dft_array(near_fields, mp.Ez, 0)

scattered_field = grating_dft - flat_dft
scattered_amplitude = np.abs(scattered_field) ** 2

[x, y, z, w] = sim.get_array_metadata(dft_cell=near_fields)

if field_profile:
    if mp.am_master():
        plt.figure(dpi=150)
        plt.pcolormesh(
            x,
            y,
            np.rot90(scattered_amplitude),
            cmap="inferno",
            shading="gouraud",
            vmin=0,
            vmax=scattered_amplitude.max(),
        )
        plt.gca().set_aspect("equal")
        plt.xlabel("x (μm)")
        plt.ylabel("y (μm)")

        # ensure that the height of the colobar matches that of the plot
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        plt.tight_layout()
        plt.show()
else:
    ky = np.fft.fftshift(np.fft.fftfreq(len(scattered_field), 1 / resolution))
    FT_scattered_field = np.fft.fftshift(np.fft.fft(scattered_field))
    if mp.am_master():
        plt.figure(dpi=150)
        plt.subplots_adjust(hspace=0.3)

        plt.subplot(2, 1, 1)
        plt.plot(y, scattered_amplitude, "bo-")
        plt.xlabel("y (μm)")
        plt.ylabel("field amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(ky, np.abs(FT_scattered_field) ** 2, "ro-")
        plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.xlabel(r"wavevector k$_y$, 2π (μm)$^{-1}$")
        plt.ylabel("Fourier transform")
        plt.gca().set_xlim([-3, 3])

        plt.tight_layout(pad=1.0)
        plt.show()
