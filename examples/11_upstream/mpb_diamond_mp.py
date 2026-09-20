# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_diamond.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mpb_diamond/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import math

import meep_torch.mp as mp
from meep_torch import mpb

# Dielectric spheres in a diamond (fcc) lattice.  This file is used in
# the "Data Analysis Tutorial" section of the MPB manual.

sqrt_half = math.sqrt(0.5)
geometry_lattice = mp.Lattice(
    basis_size=mp.Vector3(sqrt_half, sqrt_half, sqrt_half),
    basis1=mp.Vector3(0, 1, 1),
    basis2=mp.Vector3(1, 0, 1),
    basis3=mp.Vector3(1, 1),
)

# Corners of the irreducible Brillouin zone for the fcc lattice,
# in a canonical order:
vlist = [
    mp.Vector3(0, 0.5, 0.5),  # X
    mp.Vector3(0, 0.625, 0.375),  # U
    mp.Vector3(0, 0.5, 0),  # L
    mp.Vector3(0, 0, 0),  # Gamma
    mp.Vector3(0, 0.5, 0.5),  # X
    mp.Vector3(0.25, 0.75, 0.5),  # W
    mp.Vector3(0.375, 0.75, 0.375),  # K
]

k_points = mp.interpolate(4, vlist)

# define a couple of parameters (which we can set from the command_line)
eps = 11.56  # the dielectric constant of the spheres
r = 0.25  # the radius of the spheres

diel = mp.Medium(epsilon=eps)

# A diamond lattice has two "atoms" per unit cell:
geometry = [
    mp.Sphere(r, center=mp.Vector3(0.125, 0.125, 0.125), material=diel),
    mp.Sphere(r, center=mp.Vector3(-0.125, -0.125, -0.125), material=diel),
]

# (A simple fcc lattice would have only one sphere/object at the origin.)

resolution = 16  # use a 16x16x16 grid
mesh_size = 5
num_bands = 5

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_bands,
    mesh_size=mesh_size,
)
ms.device = "cuda"
ms.filename_prefix = str(OUT / "mpb_diamond")


def main():
    # run calculation, outputting electric_field energy density at the U point:
    ms.run(mpb.output_at_kpoint(mp.Vector3(0, 0.625, 0.375), mpb.output_dpwr))


if __name__ == "__main__":
    main()
