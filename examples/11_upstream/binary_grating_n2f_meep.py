# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/binary_grating_n2f.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/binary_grating_n2f/"
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
        _p = _p.with_name("binary_grating_n2f_meep-" + _p.stem)
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
from numpy import linalg as LA

import meep as mp

resolution = 25  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML
gp = 10.0  # grating period
gh = 0.5  # grating height
gdc = 0.5  # grating duty cycle

nperiods = 10  # number of unit cells in finite periodic grating

ff_distance = 1e8  # far-field distance from near-field monitor
ff_angle = 20  # far-field cone angle
ff_npts = 500  # number of far-field points

ff_length = ff_distance * math.tan(math.radians(ff_angle))
ff_res = ff_npts / ff_length

sx = dpml + dsub + gh + dpad + dpml
cell_size = mp.Vector3(sx)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

symmetries = [mp.Mirror(mp.Y)]

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub)
sources = [
    mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=src_pt)
]

k_point = mp.Vector3()

glass = mp.Medium(index=1.5)

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    k_point=k_point,
    default_material=glass,
    sources=sources,
)

nfreq = 21
n2f_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad)
n2f_obj = sim.add_near2far(fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt))

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9))

ff_source = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

sim.reset_meep()

### unit cell with periodic boundaries

sy = gp
cell_size = mp.Vector3(sx, sy)

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
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
    ),
    mp.Block(
        material=glass,
        size=mp.Vector3(gh, gdc * gp, mp.inf),
        center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh),
    ),
]

sim = mp.Simulation(
    resolution=resolution,
    split_chunks_evenly=True,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

n2f_obj = sim.add_near2far(
    fcen,
    df,
    nfreq,
    mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy)),
    nperiods=nperiods,
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9))

ff_unitcell = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

sim.reset_meep()

### finite periodic grating with flat surface termination extending into PML

num_cells = 2 * nperiods + 1
sy = dpml + num_cells * gp + dpml
cell_size = mp.Vector3(sx, sy)

pml_layers = [mp.PML(thickness=dpml)]

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
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

for j in range(num_cells):
    geometry.append(
        mp.Block(
            material=glass,
            size=mp.Vector3(gh, gdc * gp, mp.inf),
            center=mp.Vector3(
                -0.5 * sx + dpml + dsub + 0.5 * gh, -0.5 * sy + dpml + (j + 0.5) * gp
            ),
        )
    )

sim = mp.Simulation(
    resolution=resolution,
    split_chunks_evenly=True,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    k_point=k_point,
    sources=sources,
    symmetries=symmetries,
)

n2f_obj = sim.add_near2far(
    fcen, df, nfreq, mp.Near2FarRegion(center=n2f_pt, size=mp.Vector3(y=sy - 2 * dpml))
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, n2f_pt, 1e-9))

ff_supercell = sim.get_farfields(
    n2f_obj,
    ff_res,
    center=mp.Vector3(ff_distance, 0.5 * ff_length),
    size=mp.Vector3(y=ff_length),
)

norm_err = LA.norm(ff_unitcell["Ez"] - ff_supercell["Ez"]) / nperiods
print(f"error:, {nperiods}, {norm_err}")

freqs = mp.get_near2far_freqs(n2f_obj)
wvl = np.divide(1, freqs)
ff_lengths = np.linspace(0, ff_length, ff_npts)
angles = [math.degrees(math.atan(f)) for f in ff_lengths / ff_distance]

wvl_slice = 0.5
idx_slice = np.where(np.asarray(freqs) == 1 / wvl_slice)[0][0]

rel_enh = np.absolute(ff_unitcell["Ez"]) ** 2 / np.absolute(ff_source["Ez"]) ** 2

plt.figure(dpi=150)

plt.subplot(1, 2, 1)
plt.pcolormesh(wvl, angles, rel_enh, cmap="Blues", shading="flat")
plt.axis([wvl_min, wvl_max, 0, ff_angle])
plt.xlabel("wavelength (μm)")
plt.ylabel("angle (degrees)")
plt.grid(linewidth=0.5, linestyle="--")
plt.xticks([t for t in np.arange(wvl_min, wvl_max + 0.1, 0.1)])
plt.yticks([t for t in range(0, ff_angle + 1, 10)])
plt.title("far-field spectra")

plt.subplot(1, 2, 2)
plt.plot(angles, rel_enh[:, idx_slice], "bo-")
plt.xlim(0, ff_angle)
plt.ylim(0)
plt.xticks([t for t in range(0, ff_angle + 1, 10)])
plt.xlabel("angle (degrees)")
plt.ylabel("relative enhancement")
plt.grid(axis="x", linewidth=0.5, linestyle="--")
plt.title(f"f.-f. spectra @  λ = {wvl_slice:.1} μm")

plt.tight_layout(pad=0.5)
plt.show()
