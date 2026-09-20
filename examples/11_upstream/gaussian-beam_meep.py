# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/gaussian-beam.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/gaussian-beam/"
OUT.mkdir(exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

## launch a Gaussian beam
import math

import matplotlib

import meep as mp

matplotlib.use("agg")
import matplotlib.pyplot as plt

s = 14
resolution = 50
dpml = 2

cell_size = mp.Vector3(s, s)

boundary_layers = [mp.PML(thickness=dpml)]

beam_x0 = mp.Vector3(0, 3.0)  # beam focus (relative to source center)
rot_angle = 0  # CCW rotation angle about z axis (0: +y axis)
beam_kdir = mp.Vector3(0, 1, 0).rotate(
    mp.Vector3(0, 0, 1), math.radians(rot_angle)
)  # beam propagation direction
beam_w0 = 0.8  # beam waist radius
beam_E0 = mp.Vector3(0, 0, 1)
fcen = 1
sources = [
    mp.GaussianBeamSource(
        src=mp.ContinuousSource(fcen),
        center=mp.Vector3(0, -0.5 * s + dpml + 1.0),
        size=mp.Vector3(s),
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    sources=sources,
)

sim.run(until=20)

sim.plot2D(
    fields=mp.Ez,
    output_plane=mp.Volume(
        center=mp.Vector3(), size=mp.Vector3(s - 2 * dpml, s - 2 * dpml)
    ),
)

plt.savefig(f"Ez_angle{rot_angle}.png", bbox_inches="tight", pad_inches=0)
