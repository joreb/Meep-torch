# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracle: the mpb_tutorial 5x5 point defect, TM.

The section of the tutorial that exercises three things at once -- a
defect built by APPENDING an air rod over a lattice duplicate (so it only
forms if later geometry objects take precedence), the energy density
integrated inside an object, and MPB's targeted eigensolver.

Run: mamba run -n meep-oracle python tests/examples/data/generate_defect_mpb.py
"""
import json
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)
RES, NUM_BANDS = 16, 50
TARGET = (0.2812 + 0.4174) / 2

lat = mp.Lattice(size=mp.Vector3(5, 5))
geom = mp.geometric_objects_lattice_duplicates(
    lat, [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))])
geom.append(mp.Cylinder(0.2, material=mp.air))

ms = mpb.ModeSolver(num_bands=NUM_BANDS, resolution=RES, geometry_lattice=lat,
                    geometry=geom, k_points=[mp.Vector3(0.5, 0.5)])
ms.run_tm()
freqs = list(ms.get_freqs())
ms.get_dfield(25)
ms.compute_field_energy()
energy = ms.compute_energy_in_objects([mp.Cylinder(1.0, material=mp.air)])

# perfect lattice, same engine and resolution, for the TM gap edges
perfect = mpb.ModeSolver(num_bands=2, resolution=RES,
                         geometry_lattice=mp.Lattice(size=mp.Vector3(1, 1)),
                         geometry=[mp.Cylinder(0.2,
                                               material=mp.Medium(epsilon=12))],
                         k_points=[mp.Vector3(), mp.Vector3(0.5),
                                   mp.Vector3(0.5, 0.5)])
perfect.run_tm()
gap_lo = max(f[0] for f in perfect.all_freqs)
gap_hi = min(f[1] for f in perfect.all_freqs)

tms = mpb.ModeSolver(num_bands=1, resolution=RES, geometry_lattice=lat,
                     geometry=geom, k_points=[mp.Vector3(0.5, 0.5)],
                     target_freq=TARGET, tolerance=1e-8)
tms.run_tm()

out = {
    "resolution": RES, "supercell": 5, "r": 0.2, "eps": 12,
    "k": [0.5, 0.5], "num_bands": NUM_BANDS, "target_freq": TARGET,
    "defect_band": 25,
    "defect_freq": freqs[24],
    "energy_in_cylinder_r1": energy,
    "targeted_freq": tms.get_freqs()[0],
    "gap_lo": gap_lo, "gap_hi": gap_hi,
}
with open(os.path.join(HERE, "defect_mpb.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
