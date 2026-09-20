# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/cavity-farfield.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE = os.environ.get("MEEP_TORCH_DEVICE")`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid        # upstream predates numpy 2.0, which removed it
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/cavity-farfield/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import matplotlib
import numpy as np

import meep_torch.mp as mp

matplotlib.use("agg")
import matplotlib.pyplot as plt

resolution = 20  # pixels/μm

fcen = 0.25  # pulse center frequency
df = 0.2  # pulse width (in frequency)

eps = 13  # dielectric constant of waveguide
w = 1.2  # width of waveguide
r = 0.36  # radius of holes
d = 1.4  # defect spacing (ordinary spacing = 1)
N = 3  # number of holes on either side of defect

dpad = 32  # padding between last hole and PML edge
dpml = 0.5 / (fcen - 0.5 * df)  # PML thickness (> half the largest wavelength)
sx = 2 * (dpad + dpml + N) + d - 1  # size of cell in x direction

d1 = 0.2  # y-distance from waveguide edge to near2far surface
d2 = 2.0  # y-distance from near2far surface to far-field line
sy = w + 2 * (d1 + d2 + dpml)  # size of cell in y direction (perpendicular to wvg.)

cell = mp.Vector3(sx, sy, 0)

geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, w, mp.inf),
        material=mp.Medium(epsilon=eps),
    )
]

for i in range(N):
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / -2 - i)))

pml_layers = [mp.PML(dpml)]

sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df), component=mp.Hz, center=mp.Vector3()
    )
]

symmetries = [mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=-1)]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    symmetries=symmetries,
    boundary_layers=pml_layers,
    resolution=resolution,
)

nearfield = sim.add_near2far(
    fcen,
    0,
    1,
    mp.Near2FarRegion(mp.Vector3(0, 0.5 * w + d1), size=mp.Vector3(sx - 2 * dpml)),
    mp.Near2FarRegion(
        mp.Vector3(-0.5 * sx + dpml, 0.5 * w + 0.5 * d1),
        size=mp.Vector3(0, d1),
        weight=-1.0,
    ),
    mp.Near2FarRegion(
        mp.Vector3(0.5 * sx - dpml, 0.5 * w + 0.5 * d1), size=mp.Vector3(0, d1)
    ),
)

mon = sim.add_dft_fields(
    [mp.Hz],
    fcen,
    0,
    1,
    center=mp.Vector3(0, 0.5 * w + d1 + d2),
    size=mp.Vector3(sx - 2 * (dpad + dpml), 0),
)

sim.run(until_after_sources=mp.stop_when_dft_decayed())

sim.plot2D()
if mp.am_master():
    plt.savefig(
        f"cavity_farfield_plot2D_dpad{dpad}_{d1}_{d2}.png", bbox_inches="tight", dpi=150
    )

Hz_mon = sim.get_dft_array(mon, mp.Hz, 0)

(x, y, z, w) = sim.get_array_metadata(dft_cell=mon)

ff = []
for xc in x:
    ff_pt = sim.get_farfield(nearfield, mp.Vector3(xc, y[0]))
    ff.append(ff_pt[5])
ff = np.array(ff)

if mp.am_master():
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.real(Hz_mon), "bo-", label="DFT")
    plt.plot(np.real(ff), "ro-", label="N2F")
    plt.legend()
    plt.xlabel("$x$ (μm)")
    plt.ylabel("real(Hz)")

    plt.subplot(1, 3, 2)
    plt.plot(np.imag(Hz_mon), "bo-", label="DFT")
    plt.plot(np.imag(ff), "ro-", label="N2F")
    plt.legend()
    plt.xlabel("$x$ (μm)")
    plt.ylabel("imag(Hz)")

    plt.subplot(1, 3, 3)
    plt.plot(np.abs(Hz_mon), "bo-", label="DFT")
    plt.plot(np.abs(ff), "ro-", label="N2F")
    plt.legend()
    plt.xlabel("$x$ (μm)")
    plt.ylabel("|Hz|")

    plt.suptitle(
        f"comparison of near2far and actual DFT fields\n dpad={dpad}, d1={d1}, d2={d2}"
    )
    plt.subplots_adjust(wspace=0.6)
    plt.savefig(
        f"test_Hz_dft_vs_n2f_res{resolution}_dpad{dpad}_d1{d1}_d2{d2}.png",
        bbox_inches="tight",
        dpi=150,
    )
