# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/grating2d_triangular_lattice.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/grating2d_triangular_lattice/"
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
        _p = _p.with_name("grating2d_triangular_lattice_mp-" + _p.stem)
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

# Computes the diffraction orders of a 2D binary grating with
# triangular lattice using a rectangular supercell and verifies
# that only the diffraction orders of the actual unit cell
# produce non-zero power (up to discretization error)
import math

import numpy as np

import meep_torch.mp as mp

resolution = 100  # pixels/μm

ng = 1.5
glass = mp.Medium(index=ng)

wvl = 0.5  # wavelength
fcen = 1 / wvl

# rectangular supercell
sx = 1.0
sy = np.sqrt(3)

dpml = 1.0  # PML thickness
dsub = 2.0  # substrate thickness
dair = 2.0  # air padding
hcyl = 0.5  # cylinder height
rcyl = 0.1  # cylinder radius

sz = dpml + dsub + hcyl + dair + dpml

cell_size = mp.Vector3(sx, sy, sz)

boundary_layers = [mp.PML(thickness=dpml, direction=mp.Z)]

# periodic boundary conditions
k_point = mp.Vector3()

src_pt = mp.Vector3(0, 0, -0.5 * sz + dpml)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=0.1 * fcen),
        size=mp.Vector3(sx, sy, 0),
        center=src_pt,
        component=mp.Ex,
    )
]

substrate = [
    mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, dpml + dsub),
        center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * (dpml + dsub)),
        material=glass,
    )
]

cyl_grating = [
    mp.Cylinder(
        center=mp.Vector3(0, 0, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(-0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(-0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
    mp.Cylinder(
        center=mp.Vector3(0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl),
        radius=rcyl,
        height=hcyl,
        material=glass,
    ),
]

geometry = substrate + cyl_grating

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    sources=sources,
    geometry=geometry,
    boundary_layers=boundary_layers,
    k_point=k_point,
)

tran_pt = mp.Vector3(0, 0, 0.5 * sz - dpml)
tran_flux = sim.add_mode_monitor(
    fcen, 0, 1, mp.ModeRegion(center=tran_pt, size=mp.Vector3(sx, sy, 0))
)

sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Ex, src_pt, 1e-6))

# diffraction order of unit cell (triangular lattice)
mx = 0
my = 1

# check: for diffraction orders of supercell for which
#        nx = mx and ny = -mx + 2*my and thus
#        only even orders should produce nonzero power
nx = mx
for ny in range(4):
    kz2 = fcen**2 - (nx / sx) ** 2 - (ny / sy) ** 2
    if kz2 > 0:
        res = sim.get_eigenmode_coefficients(
            tran_flux, mp.DiffractedPlanewave((nx, ny, 0), mp.Vector3(0, 1, 0), 1, 0)
        )
        t_coeffs = res.alpha
        tran = abs(t_coeffs[0, 0, 0]) ** 2

        print(f"order:, {nx}, {ny}, {tran:.5f}")
