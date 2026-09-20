# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_line_defect.py by scripts/port_upstream_example.py.
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
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid        # upstream predates numpy 2.0, which removed it
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/mpb_line_defect/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import math

import meep as mp
from meep import mpb

# A line_defect waveguide in a 2d triangular lattice of dielectric
# rods (c.f. tri_rods.ctl), formed by a row of missing rods along the
# "x" direction.  (Here, "x" and "y" refer to the first and second
# basis directions.)  This structure supports a single guided band
# within the band gap, much like the analogous waveguide in a square
# lattice of rods (see "Photonic Crystals" by Joannopoulos et al.).

supercell_y = 7  # the (odd) number of lateral supercell periods

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, supercell_y),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

eps = 12  # the dielectric constant of the rods
r = 0.2  # the rod radius in the bulk crystal

geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=eps))]

# duplicate the bulk crystal rods over the supercell:
geometry = mp.geometric_objects_lattice_duplicates(geometry_lattice, geometry)

# add a rod of air, to erase a row of rods and form a waveguide:
geometry += [mp.Cylinder(r, material=mp.air)]

Gamma = mp.Vector3()
K_prime = mp.lattice_to_reciprocal(
    mp.Vector3(0.5), geometry_lattice
)  # edge of Brillouin zone.
k_points = mp.interpolate(4, [Gamma, K_prime])

# the bigger the supercell, the more bands you need to compute to get
# to the defect modes (the lowest band is "folded" supercell_y times):
extra_bands = 5  # number of extra bands to compute above the gap
num_bands = supercell_y + extra_bands

resolution = 32

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
)
ms.filename_prefix = str(OUT / "mpb_line_defect")


def main():
    # Compute the TM modes, outputting the Ez field in the *middle* of the
    # band.  (In general, the guided mode in such an air defect may have
    # exited the gap by the time it reaches the edge of the Brillouin
    # zone at K_prime.)
    ms.run_tm(
        mpb.output_at_kpoint(k_points[len(k_points) // 2]),
        ms.fix_efield_phase,
        mpb.output_efield_z,
    )


if __name__ == "__main__":
    main()
