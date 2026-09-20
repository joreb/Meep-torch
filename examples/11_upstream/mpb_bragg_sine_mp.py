# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_bragg_sine.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mpb_bragg_sine/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import math

import meep_torch.mp as mp
from meep_torch import mpb

# Compute the band structure for a Bragg mirror consisting of a
# sinusoidally-varying dielectric index.

# The index will vary sinusoidally between index-min and index-max:
index_min = 1
index_max = 3


# Define a function of position p (in the lattice basis) that returns
# the material at that position.  In this case, we use the function:
#        index-min + 0.5 * (index-max - index-min)
#                        * (1 + cos(2*pi*x))
# This is periodic, and also has inversion symmetry.
def eps_func(p):
    return mp.Medium(
        index=index_min
        + 0.5 * (index_max - index_min) * (1 + math.cos(2 * math.pi * p.x))
    )


geometry_lattice = mp.Lattice(size=mp.Vector3(1))  # 1d cell

# We'll just make it the default material, so that it goes everywhere.
default_material = eps_func

k_points = mp.interpolate(9, [mp.Vector3(), mp.Vector3(x=0.5)])

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry_lattice=geometry_lattice,
    resolution=resolution,
    default_material=default_material,
)
ms.device = "cuda"
ms.filename_prefix = str(OUT / "mpb_bragg_sine")


def main():
    # the TM and TE bands are degenerate, so we only need TM:
    ms.run_tm()


if __name__ == "__main__":
    main()
