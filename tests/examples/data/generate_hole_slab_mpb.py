# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracle: triangular hole-slab z-even bands at K (guided).

Run: mamba run -n meep-oracle python tests/examples/data/generate_hole_slab_mpb.py
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)
ms = mpb.ModeSolver(
    num_bands=3, resolution=mp.Vector3(16, 16, 16),
    geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1, 4),
                                basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                                basis2=mp.Vector3(math.sqrt(3) / 2, -0.5)),
    geometry=[mp.Block(material=mp.Medium(epsilon=12),
                       size=mp.Vector3(mp.inf, mp.inf, 0.5)),
              mp.Cylinder(0.3, material=mp.air, height=mp.inf)],
    k_points=[mp.Vector3(-1 / 3, 1 / 3)])
ms.run_zeven()
out = {"h": 0.5, "r": 0.3, "eps": 12, "supercell_h": 4,
       "K_zeven": list(map(float, ms.all_freqs[0]))}
json.dump(out, open(os.path.join(HERE, "hole_slab_mpb.json"), "w"), indent=1)
print("hole_slab K z-even:", out["K_zeven"])
