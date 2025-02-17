# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracles: square-lattice rod TM bands + 1D sine-index bands.

Run: mamba run -n meep-oracle python tests/examples/data/generate_mpb_oracles.py
-> tests/examples/data/sq_lattice_mpb.json, bragg_sine_mpb.json
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)

# --- square lattice of rods (a = 1, r = 0.2, eps = 8.9), TM, Gamma-X ---
ks = [mp.Vector3(kx, 0) for kx in (0.1, 0.2, 0.3, 0.4, 0.5)]
ms = mpb.ModeSolver(
    num_bands=3, k_points=ks, resolution=32,
    geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1)),
    geometry=[mp.Cylinder(0.2, material=mp.Medium(epsilon=8.9))])
ms.run_tm()
sq = {"r": 0.2, "eps": 8.9, "kx": [k.x for k in ks],
      "bands": [list(map(float, row)) for row in ms.all_freqs]}
with open(os.path.join(HERE, "sq_lattice_mpb.json"), "w") as f:
    json.dump(sq, f, indent=1)
print("sq lattice bands:", sq["bands"])

# --- 1D sine-index Bragg (index 1..3, cos profile), first bands at k ---
idx_min, idx_max = 1, 3


def eps_func(p):
    return mp.Medium(index=idx_min + 0.5 * (idx_max - idx_min)
                     * (1 + math.cos(2 * math.pi * p.x)))


ks1 = [mp.Vector3(kx) for kx in (0.1, 0.3, 0.5)]
ms = mpb.ModeSolver(num_bands=3, k_points=ks1, resolution=64,
                    geometry_lattice=mp.Lattice(size=mp.Vector3(1)),
                    default_material=eps_func)
ms.run_tm()
br = {"index_min": idx_min, "index_max": idx_max, "kx": [k.x for k in ks1],
      "bands": [list(map(float, row)) for row in ms.all_freqs]}
with open(os.path.join(HERE, "bragg_sine_mpb.json"), "w") as f:
    json.dump(br, f, indent=1)
print("bragg sine bands:", br["bands"])
