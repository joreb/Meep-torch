# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_bragg.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mpb_bragg/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import meep_torch.mp as mp
from meep_torch.mp import mpb

# Compute the bands at the X point for a quarter-wave stack Bragg
# mirror (this is the point that defines the band gap edges).

# the high and low indices:
n_lo = 1.0
n_hi = 3.0

w_hi = n_lo / (n_hi + n_lo)  # a quarter_wave stack

geometry_lattice = mp.Lattice(size=mp.Vector3(1))  # 1d cell
default_material = mp.Medium(index=n_lo)
geometry = mp.Cylinder(
    material=mp.Medium(index=n_hi),
    center=mp.Vector3(),
    axis=mp.Vector3(1),
    radius=mp.inf,
    height=w_hi,
)

kx = 0.5
k_points = [mp.Vector3(kx)]

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry_lattice=geometry_lattice,
    geometry=[geometry],
    resolution=resolution,
    default_material=default_material,
)
ms.device = DEVICE   # generated: 3D planewave backend only
ms.filename_prefix = str(OUT / "mpb_bragg")


def main():
    ms.run_tm(
        mpb.output_hfield_y
    )  # note that TM and TE bands are degenerate, so we only need TM


if __name__ == "__main__":
    main()
