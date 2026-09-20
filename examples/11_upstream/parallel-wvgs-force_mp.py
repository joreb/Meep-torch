# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/parallel-wvgs-force.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/parallel-wvgs-force/"
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
        _p = _p.with_name("parallel-wvgs-force_mp-" + _p.stem)
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

import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

resolution = 40  # pixels/μm

Si = mp.Medium(index=3.45)

dpml = 1.0
pml_layers = [mp.PML(dpml)]

sx = 5
sy = 3
cell = mp.Vector3(sx + 2 * dpml, sy + 2 * dpml, 0)

a = 1.0  # waveguide width/height

k_point = mp.Vector3(z=0.5)


def parallel_waveguide(s, xodd):
    geometry = [
        mp.Block(
            center=mp.Vector3(-0.5 * (s + a)),
            size=mp.Vector3(a, a, mp.inf),
            material=Si,
        ),
        mp.Block(
            center=mp.Vector3(0.5 * (s + a)), size=mp.Vector3(a, a, mp.inf), material=Si
        ),
    ]

    symmetries = [mp.Mirror(mp.X, phase=-1 if xodd else 1), mp.Mirror(mp.Y, phase=-1)]

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        symmetries=symmetries,
        k_point=k_point,
    )

    sim.init_sim()
    EigenmodeData = sim.get_eigenmode(
        0.22,
        mp.Z,
        mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy)),
        2 if xodd else 1,
        k_point,
        match_frequency=False,
        parity=mp.ODD_Y,
    )

    fcen = EigenmodeData.freq
    print(f'freq:, {"xodd" if xodd else "xeven"}, {s}, {fcen}')

    sim.reset_meep()

    eig_sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
            size=mp.Vector3(sx, sy),
            center=mp.Vector3(),
            eig_band=2 if xodd else 1,
            eig_kpoint=k_point,
            eig_match_freq=False,
            eig_parity=mp.ODD_Y,
        )
    ]

    sim.change_sources(eig_sources)

    flux_reg = mp.FluxRegion(
        direction=mp.Z, center=mp.Vector3(), size=mp.Vector3(sx, sy)
    )
    wvg_flux = sim.add_flux(fcen, 0, 1, flux_reg)

    force_reg1 = mp.ForceRegion(
        mp.Vector3(0.49 * s), direction=mp.X, weight=1, size=mp.Vector3(y=sy)
    )
    force_reg2 = mp.ForceRegion(
        mp.Vector3(0.5 * s + 1.01 * a), direction=mp.X, weight=-1, size=mp.Vector3(y=sy)
    )
    wvg_force = sim.add_force(fcen, 0, 1, force_reg1, force_reg2)

    sim.run(until_after_sources=1500)

    flux = mp.get_fluxes(wvg_flux)[0]
    force = mp.get_forces(wvg_force)[0]
    print(
        f'data:, {"xodd" if xodd else "xeven"}, {s}, {flux}, {force}, {-force / flux}'
    )

    sim.reset_meep()
    return flux, force


s = np.arange(0.05, 1.05, 0.05)
fluxes_odd = np.zeros(s.size)
forces_odd = np.zeros(s.size)
fluxes_even = np.zeros(s.size)
forces_even = np.zeros(s.size)

for k in range(len(s)):
    fluxes_odd[k], forces_odd[k] = parallel_waveguide(s[k], True)
    fluxes_even[k], forces_even[k] = parallel_waveguide(s[k], False)

plt.figure(dpi=150)
plt.plot(s, -forces_odd / fluxes_odd, "rs", label="anti-symmetric")
plt.plot(s, -forces_even / fluxes_even, "bo", label="symmetric")
plt.grid(True)
plt.xlabel("waveguide separation s/a")
plt.ylabel("optical force (F/L)(ac/P)")
plt.legend(loc="upper right")
plt.show()
