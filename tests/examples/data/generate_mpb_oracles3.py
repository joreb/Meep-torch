# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracles, round 3: sq_rods, bragg quarter-wave, tri holes (TM).

Run: mamba run -n meep-oracle python tests/examples/data/generate_mpb_oracles3.py
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)

ms = mpb.ModeSolver(num_bands=3, resolution=32,
                    geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1)),
                    geometry=[mp.Cylinder(0.2, material=mp.Medium(epsilon=11.56))],
                    k_points=[mp.Vector3(kx, 0) for kx in (0.2, 0.4)])
ms.run_tm()
json.dump({"r": 0.2, "eps": 11.56, "kx": [0.2, 0.4],
           "bands": [list(map(float, r)) for r in ms.all_freqs]},
          open(os.path.join(HERE, "sq_rods_mpb.json"), "w"))
print("sq_rods:", [list(map(float, r)) for r in ms.all_freqs])

ms = mpb.ModeSolver(num_bands=3, resolution=64,
                    geometry_lattice=mp.Lattice(size=mp.Vector3(1)),
                    default_material=mp.Medium(index=1.0),
                    geometry=[mp.Block(size=mp.Vector3(0.25, mp.inf, mp.inf),
                                       material=mp.Medium(index=3.0))],
                    k_points=[mp.Vector3(0.3), mp.Vector3(0.5)])
ms.run_tm()
json.dump({"n_lo": 1.0, "n_hi": 3.0, "w": 0.25, "kx": [0.3, 0.5],
           "bands": [list(map(float, r)) for r in ms.all_freqs]},
          open(os.path.join(HERE, "bragg_mpb.json"), "w"))
print("bragg:", [list(map(float, r)) for r in ms.all_freqs])

TRI = mp.Lattice(size=mp.Vector3(1, 1),
                 basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                 basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))
ms = mpb.ModeSolver(num_bands=3, resolution=32, geometry_lattice=TRI,
                    default_material=mp.Medium(epsilon=12),
                    geometry=[mp.Cylinder(0.45, material=mp.air)],
                    k_points=[mp.Vector3(y=0.5)])
ms.run_tm()
json.dump({"r": 0.45, "eps": 12, "M_tm": list(map(float, ms.all_freqs[0]))},
          open(os.path.join(HERE, "tri_holes_mpb.json"), "w"))
print("tri_holes M TM:", list(map(float, ms.all_freqs[0])))
