# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/cyl-ellipsoid.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the generated ModeSolver lines, so that a difference between
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
OUT = pathlib.Path(__file__).parent / "output/cyl-ellipsoid/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import meep as mp


def main():

    c = mp.Cylinder(radius=3, material=mp.Medium(index=3.5))
    e = mp.Ellipsoid(size=mp.Vector3(1, 2, mp.inf))

    src_cmpt = mp.Hz
    sources = mp.Source(
        src=mp.GaussianSource(1, fwidth=0.1), component=src_cmpt, center=mp.Vector3()
    )

    if src_cmpt == mp.Ez:
        symmetries = [mp.Mirror(mp.X), mp.Mirror(mp.Y)]

    if src_cmpt == mp.Hz:
        symmetries = [mp.Mirror(mp.X, -1), mp.Mirror(mp.Y, -1)]

    sim = mp.Simulation(
        cell_size=mp.Vector3(10, 10),
        geometry=[c, e],
        boundary_layers=[mp.PML(1.0)],
        sources=[sources],
        symmetries=symmetries,
        resolution=100,
    )

    def print_stuff(sim_obj):
        v = mp.Vector3(4.13, 3.75, 0)
        p = sim.get_field_point(src_cmpt, v)
        print(f"t, Ez: {sim.round_time()} {p.real}+{p.imag}i")

    sim.run(
        mp.at_beginning(mp.output_epsilon),
        mp.at_every(0.25, print_stuff),
        mp.at_end(print_stuff),
        mp.at_end(mp.output_efield_z),
        until=23,
    )

    print(f"stopped at meep time = {sim.round_time()}")


if __name__ == "__main__":
    main()
