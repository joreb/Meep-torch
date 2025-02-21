# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracles, round 2: tri rods, honeycomb, strip, line defect.

Run: mamba run -n meep-oracle python tests/examples/data/generate_mpb_oracles2.py
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)
TRI = mp.Lattice(size=mp.Vector3(1, 1),
                 basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                 basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))

# --- tri rods (r=0.2, eps=12), TM at Gamma, M, K ---
ms = mpb.ModeSolver(num_bands=4, resolution=32, geometry_lattice=TRI,
                    geometry=[mp.Cylinder(0.2, material=mp.Medium(epsilon=12))],
                    k_points=[mp.Vector3(), mp.Vector3(y=0.5),
                              mp.Vector3(-1 / 3, 1 / 3)])
ms.run_tm()
tri = {"r": 0.2, "eps": 12,
       "gamma": list(map(float, ms.all_freqs[0])),
       "M": list(map(float, ms.all_freqs[1])),
       "K": list(map(float, ms.all_freqs[2]))}
json.dump(tri, open(os.path.join(HERE, "tri_rods_mpb.json"), "w"), indent=1)
print("tri:", tri)

# --- honeycomb rods (r=0.14, eps=12), TM at Gamma ---
ms = mpb.ModeSolver(num_bands=4, resolution=32, geometry_lattice=TRI,
                    geometry=[
                        mp.Cylinder(0.14, center=mp.Vector3(1 / 6, 1 / 6),
                                    material=mp.Medium(epsilon=12)),
                        mp.Cylinder(0.14, center=mp.Vector3(-1 / 6, -1 / 6),
                                    material=mp.Medium(epsilon=12))],
                    k_points=[mp.Vector3()])
ms.run_tm()
honey = {"r": 0.14, "eps": 12, "gamma": list(map(float, ms.all_freqs[0]))}
json.dump(honey, open(os.path.join(HERE, "honey_rods_mpb.json"), "w"), indent=1)
print("honey:", honey)

# --- Si strip on oxide (w=0.3, h=0.25), fundamental band ---
Si, SiO2 = mp.Medium(index=3.45), mp.Medium(index=1.45)
ms = mpb.ModeSolver(
    num_bands=2, resolution=32,
    geometry_lattice=mp.Lattice(size=mp.Vector3(0, 2, 2)),
    geometry=[mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5 * (2 - 0.25)),
                       center=mp.Vector3(z=0.25 * (2 + 0.25)), material=SiO2),
              mp.Block(size=mp.Vector3(mp.inf, 0.3, 0.25), material=Si)],
    k_points=[mp.Vector3(2.0), mp.Vector3(3.0)])
ms.run()
strip = {"w": 0.3, "h": 0.25, "n_si": 3.45, "n_ox": 1.45,
         "beta": [2.0, 3.0],
         "bands": [list(map(float, row)) for row in ms.all_freqs]}
json.dump(strip, open(os.path.join(HERE, "strip_mpb.json"), "w"), indent=1)
print("strip:", strip)

# --- line defect (tri rods, 7-row supercell, one rod removed), TM ---
lat = mp.Lattice(size=mp.Vector3(1, 7),
                 basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                 basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))
geom = mp.geometric_objects_lattice_duplicates(
    lat, [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))])
geom += [mp.Cylinder(0.2, material=mp.air)]
ms = mpb.ModeSolver(num_bands=12, resolution=32, geometry_lattice=lat,
                    geometry=geom, k_points=[mp.Vector3(0.4)])
ms.run_tm()
ld = {"r": 0.2, "eps": 12, "kx": 0.4, "supercell_y": 7,
      "bands": list(map(float, ms.all_freqs[0]))}
json.dump(ld, open(os.path.join(HERE, "line_defect_mpb.json"), "w"), indent=1)
print("line_defect:", ld)
