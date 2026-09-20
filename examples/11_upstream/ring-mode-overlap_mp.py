# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/ring-mode-overlap.py by scripts/port_upstream_example.py.
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
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/ring-mode-overlap/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

# Calculating 2d ring-resonator modes, from the Meep tutorial.
import meep_torch.mp as mp

n = 3.4  # index of waveguide
w = 1  # width of waveguide
r = 1  # inner radius of ring

pad = 4  # padding between waveguide and edge of PML
dpml = 2  # thickness of PML

sxy = 2 * (r + w + pad + dpml)  # cell size
cell = mp.Vector3(sxy, sxy)

# Create a ring waveguide by two overlapping cylinders - later objects
# take precedence over earlier objects, so we put the outer cylinder first.
# and the inner (air) cylinder second.
geometry = [
    mp.Cylinder(radius=r + w, height=mp.inf, material=mp.Medium(index=n)),
    mp.Cylinder(radius=r, height=mp.inf, material=mp.air),
]

pml_layers = [mp.PML(dpml)]
resolution = 20

# If we don't want to excite a specific mode symmetry, we can just
# put a single point source at some arbitrary place, pointing in some
# arbitrary direction. We will only look for Ez-polarized modes.

fcen = 0.118  # pulse center frequency
df = 0.010  # pulse width (in frequency)
sources = [
    mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ez,
        center=mp.Vector3(r + 0.1),
    )
]

# exploit the mirror symmetry in structure+source:
symmetries = [mp.Mirror(mp.Y)]

sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    geometry=geometry,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
)

h1 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h1), until_after_sources=300)

fields2 = sim.fields
sim.reset_meep()

fcen = 0.236
h2 = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)
sim.run(mp.after_sources(h2), until_after_sources=300)


def overlap_integral(r, ez1, ez2):
    return ez1.conjugate() * ez2


res = sim.integrate2_field_function(fields2, [mp.Ez], [mp.Ez], overlap_integral)
print(f"overlap integral of mode at w and 2w: {abs(res)}")
