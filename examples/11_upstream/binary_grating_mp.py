# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/binary_grating.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/binary_grating/"
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
        _p = _p.with_name("binary_grating_mp-" + _p.stem)
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

resolution = 60  # pixels/μm

dpml = 1.0  # PML thickness
dsub = 3.0  # substrate thickness
dpad = 3.0  # padding between grating and PML
gp = 10.0  # grating period
gh = 0.5  # grating height
gdc = 0.5  # grating duty cycle

sx = dpml + dsub + gh + dpad + dpml
sy = gp

cell_size = mp.Vector3(sx, sy, 0)
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

wvl_min = 0.4  # min wavelength
wvl_max = 0.6  # max wavelength
fmin = 1 / wvl_max  # min frequency
fmax = 1 / wvl_min  # max frequency
fcen = 0.5 * (fmin + fmax)  # center frequency
df = fmax - fmin  # frequency width

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub, 0, 0)
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=src_pt,
        size=mp.Vector3(0, sy, 0),
    )
]

k_point = mp.Vector3(0, 0, 0)

glass = mp.Medium(index=1.5)

symmetries = [mp.Mirror(mp.Y)]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    k_point=k_point,
    default_material=glass,
    sources=sources,
    symmetries=symmetries,
)

nfreq = 21
mon_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
flux_mon = sim.add_flux(
    fcen, df, nfreq, mp.FluxRegion(center=mon_pt, size=mp.Vector3(0, sy, 0))
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))

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

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))

freqs = mp.get_eigenmode_freqs(mode_mon)

nmode = 10
res = sim.get_eigenmode_coefficients(
    mode_mon, range(1, nmode + 1), eig_parity=mp.ODD_Z + mp.EVEN_Y
)
coeffs = res.alpha
kdom = res.kdom

mode_wvl = []
mode_angle = []
mode_tran = []

for nm in range(nmode):
    for nf in range(nfreq):
        mode_wvl.append(1 / freqs[nf])
        mode_angle.append(math.degrees(math.acos(kdom[nm * nfreq + nf].x / freqs[nf])))
        tran = abs(coeffs[nm, nf, 0]) ** 2 / input_flux[nf]
        mode_tran.append(0.5 * tran if nm != 0 else tran)
        print(
            "grating{}:, {:.5f}, {:.2f}, {:.8f}".format(
                nm, mode_wvl[-1], mode_angle[-1], mode_tran[-1]
            )
        )

tran_max = round(max(mode_tran), 1)

plt.figure()
plt.pcolormesh(
    np.reshape(mode_wvl, (nmode, nfreq)),
    np.reshape(mode_angle, (nmode, nfreq)),
    np.reshape(mode_tran, (nmode, nfreq)),
    cmap="Blues",
    shading="nearest",
    vmin=0,
    vmax=tran_max,
)
plt.axis([min(mode_wvl), max(mode_wvl), min(mode_angle), max(mode_angle)])
plt.xlabel("wavelength (μm)")
plt.ylabel("diffraction angle (degrees)")
plt.xticks(list(np.arange(0.4, 0.7, 0.1)))
plt.yticks(list(range(0, 35, 5)))
plt.title("transmittance of diffraction orders")
cbar = plt.colorbar()
cbar.set_ticks(list(np.arange(0, tran_max + 0.1, 0.1)))
cbar.set_ticklabels([f"{t:.1f}" for t in np.arange(0, tran_max + 0.1, 0.1)])
plt.show()
