# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_data_analysis.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/mpb_data_analysis/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch.mp import mpb

examples_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, examples_dir)


def tri_rods():
    # Import the ModeSolver defined in the mpb_tri_rods.py example
    from mpb_tri_rods import ms as tr_ms

    efields = []

    # Band function to collect the efields
    def get_efields(tr_ms, band):
        efields.append(tr_ms.get_efield(band))

    tr_ms.run_tm(
        mpb.output_at_kpoint(
            mp.Vector3(1 / -3, 1 / 3), mpb.fix_efield_phase, get_efields
        )
    )

    # Create an MPBData instance to transform the efields
    md = mpb.MPBData(rectify=True, resolution=32, periods=3)

    converted = []
    for f in efields:
        # Get just the z component of the efields
        f = f[..., 0, 2]
        converted.append(md.convert(f))

    tr_ms.run_te()

    eps = tr_ms.get_epsilon()
    plt.imshow(eps.T, interpolation="spline36", cmap="binary")
    plt.axis("off")
    plt.show()

    md = mpb.MPBData(rectify=True, resolution=32, periods=3)
    rectangular_data = md.convert(eps)
    plt.imshow(rectangular_data.T, interpolation="spline36", cmap="binary")
    plt.axis("off")
    plt.show()

    for i, f in enumerate(converted):
        plt.subplot(331 + i)
        plt.contour(rectangular_data.T, cmap="binary")
        plt.imshow(np.real(f).T, interpolation="spline36", cmap="RdBu", alpha=0.9)
        plt.axis("off")

    plt.show()


def diamond():
    # Import the ModeSolver from the mpb_diamond.py example
    from mpb_diamond import ms as d_ms

    dpwr = []

    def get_dpwr(ms, band):
        dpwr.append(ms.get_dpwr(band))

    d_ms.run(mpb.output_at_kpoint(mp.Vector3(0, 0.625, 0.375), get_dpwr))

    md = mpb.MPBData(rectify=True, periods=2, resolution=32)
    converted_dpwr = [md.convert(d) for d in dpwr]

    # TODO: Plot


if __name__ == "__main__":
    tri_rods()
    diamond()
