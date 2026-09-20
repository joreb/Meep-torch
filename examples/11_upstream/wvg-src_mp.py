# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/wvg-src.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/wvg-src/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import meep_torch.mp as mp

# Example file illustrating an eigenmode source, generating a waveguide mode
# (requires recent MPB version to be installed before Meep is compiled)

cell = mp.Vector3(16, 8)

# an asymmetrical dielectric waveguide:
geometry = [
    mp.Block(
        center=mp.Vector3(),
        size=mp.Vector3(mp.inf, 1, mp.inf),
        material=mp.Medium(epsilon=12),
    ),
    mp.Block(
        center=mp.Vector3(y=0.3),
        size=mp.Vector3(mp.inf, 0.1, mp.inf),
        material=mp.Medium(),
    ),
]

# create a transparent source that excites a right-going waveguide mode
sources = [
    mp.EigenModeSource(
        src=mp.ContinuousSource(0.15),
        size=mp.Vector3(y=6),
        center=mp.Vector3(x=-5),
        component=mp.Dielectric,
        eig_parity=mp.ODD_Z,
    )
]

pml_layers = [mp.PML(1.0)]

force_complex_fields = True  # so we can get time-average flux

resolution = 10

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    force_complex_fields=force_complex_fields,
    resolution=resolution,
)

sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.at_end(mp.output_png(mp.Ez, "-a yarg -A $EPS -S3 -Zc dkbluered", rm_h5=False)),
    until=200,
)

flux1 = sim.flux_in_box(
    mp.X, mp.Volume(center=mp.Vector3(-6.0), size=mp.Vector3(1.8, 6))
)
flux2 = sim.flux_in_box(
    mp.X, mp.Volume(center=mp.Vector3(6.0), size=mp.Vector3(1.8, 6))
)

# averaged over y region of width 1.8
print(f"left-going flux = {flux1 / -1.8}")

# averaged over y region of width 1.8
print(f"right-going flux = {flux2 / 1.8}")
