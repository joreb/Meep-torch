# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracle: the tutorial's anisotropic complete-2D-gap section.

Rods of epsilon_diag (1, 1, 12) in a background of (12, 12, 1), solved with
run() -- no parity -- so the bands are the UNION of TM and TE and a gap in
that union is a COMPLETE 2D gap. Exercises three things at once: a
diagonal epsilon tensor, 2D no-parity, and the union band numbering.

Run: mamba run -n meep-oracle python tests/examples/data/generate_aniso_mpb.py
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)
RES, NB = 32, 8
# The tutorial reaches this section with the TRIANGULAR lattice still set
# (from the "Bands of a Triangular Lattice" block) and with k_points that
# are the interpolated SQUARE path -- upstream assigns the triangular
# Gamma/M/K list and then immediately overwrites it with
# `mp.interpolate(4, k_points)`, where k_points is still the square one.
# That quirk is upstream's; reproduce the state, not the intent.
TRI = mp.Lattice(size=mp.Vector3(1, 1),
                 basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                 basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))
K = mp.interpolate(4, [mp.Vector3(), mp.Vector3(0.5),
                       mp.Vector3(0.5, 0.5), mp.Vector3()])

ms = mpb.ModeSolver(
    num_bands=NB, resolution=RES,
    geometry_lattice=TRI,
    geometry=[mp.Cylinder(0.3,
                          material=mp.Medium(epsilon_diag=mp.Vector3(1, 1, 12)))],
    default_material=mp.Medium(epsilon_diag=mp.Vector3(12, 12, 1)),
    k_points=K)
ms.run()

out = {"resolution": RES, "num_bands": NB, "r": 0.3,
       "lattice": "triangular",
       "rod_eps_diag": [1, 1, 12], "bg_eps_diag": [12, 12, 1],
       "k": [[float(k.x), float(k.y)] for k in K],
       "all_freqs": [[float(f) for f in row] for row in ms.all_freqs],
       "gap_list": [[float(x) for x in g] for g in ms.gap_list]}
with open(os.path.join(HERE, "aniso_mpb.json"), "w") as f:
    json.dump(out, f, indent=1)
print("gaps:", out["gap_list"])
print("band1 range", min(r[0] for r in out["all_freqs"]),
      max(r[0] for r in out["all_freqs"]))
