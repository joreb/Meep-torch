# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracle: triangular rods, TE (zeven) bands.

tri_rods_mpb.json carries TM only. TE on a triangular lattice is the case
that exercises the oblique operator's metric CROSS term hardest, because
the in-plane E field sees the 1/eps weights on both edge families and the
mixed u-v derivative that a rectangular cell does not have.

Run: mamba run -n meep-oracle python tests/examples/data/generate_tri_te_mpb.py
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)
R, EPS, RES, NB = 0.2, 12, 32, 4
TRI = mp.Lattice(size=mp.Vector3(1, 1),
                 basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                 basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))
KPTS = {"gamma": mp.Vector3(),
        "M": mp.Vector3(y=0.5),
        "K": mp.Vector3(-1 / 3, 1 / 3)}

ms = mpb.ModeSolver(num_bands=NB, resolution=RES, geometry_lattice=TRI,
                    geometry=[mp.Cylinder(R, material=mp.Medium(epsilon=EPS))],
                    k_points=list(KPTS.values()))
ms.run_te()
out = {"r": R, "eps": EPS, "resolution": RES, "num_bands": NB,
       "parity": "te"}
for i, name in enumerate(KPTS):
    out[name] = [float(f) for f in ms.all_freqs[i]]
with open(os.path.join(HERE, "tri_te_mpb.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
