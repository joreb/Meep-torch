# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/phase_in_material.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/phase_in_material/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import meep_torch.mp as mp

cell_size = mp.Vector3(6, 6, 0)

geometry1 = [
    mp.Cylinder(center=mp.Vector3(), radius=1.0, material=mp.Medium(index=3.5))
]

sim1 = mp.Simulation(cell_size=cell_size, geometry=geometry1, resolution=20)

sim1.init_sim()

geometry2 = [
    mp.Cylinder(center=mp.Vector3(1, 1), radius=1.0, material=mp.Medium(index=3.5))
]

sim2 = mp.Simulation(cell_size=cell_size, geometry=geometry2, resolution=20)

sim2.init_sim()

sim1.fields.phase_in_material(sim2.structure, 10.0)

sim1.run(
    mp.at_beginning(mp.output_epsilon), mp.at_every(0.5, mp.output_epsilon), until=10
)
