# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/cherenkov-radiation.py by scripts/port_upstream_example.py.
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

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/cherenkov-radiation/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

## moving point charge with superluminal phase velocity in dielectric media emitting Cherenkov radiation
import meep as mp

sx = 60
sy = 60
cell_size = mp.Vector3(sx, sy, 0)

dpml = 1.0
pml_layers = [mp.PML(thickness=dpml)]

v = 0.7  # velocity of point charge

symmetries = [mp.Mirror(direction=mp.Y)]

sim = mp.Simulation(
    resolution=10,
    cell_size=cell_size,
    default_material=mp.Medium(index=1.5),
    symmetries=symmetries,
    boundary_layers=pml_layers,
)


def move_source(sim):
    sim.change_sources(
        [
            mp.Source(
                mp.ContinuousSource(frequency=1e-10),
                component=mp.Ex,
                center=mp.Vector3(-0.5 * sx + dpml + v * sim.meep_time()),
            )
        ]
    )


sim.run(
    move_source,
    mp.at_every(2, mp.output_png(mp.Hz, "-vZc dkbluered -M 1")),
    until=sx / v,
)
