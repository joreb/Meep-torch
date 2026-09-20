# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_sq_rods.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mpb_sq_rods/"
OUT.mkdir(exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import time

import meep as mp
from meep import mpb

# Compute band structure for a square lattice of dielectric rods
# in air.

# Define various parameters with define_param so that they are
# settable from the command_line (with mpb <param>=<value>):
r = 0.2  # radius of the rods
eps = 11.56  # dielectric constant
k_interp = 4  # number of k points to interpolate

GaAs = mp.Medium(epsilon=eps)

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))  # 2d cell

geometry = [mp.Cylinder(r, material=GaAs)]

Gamma = mp.Vector3()
X = mp.Vector3(0.5, 0)
M = mp.Vector3(0.5, 0.5)
k_points = mp.interpolate(k_interp, [Gamma, X, M, Gamma])

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
)
ms.filename_prefix = str(OUT / "mpb_sq_rods")


def main():
    # Compute the TE and TM bands and report the total elapsed time:
    t0 = time.time()
    ms.run_te()
    ms.run_tm()
    print(f"total time for both TE and TM bands: {time.time() - t0:.2f} seconds")

    ms.display_eigensolver_stats()


if __name__ == "__main__":
    main()
