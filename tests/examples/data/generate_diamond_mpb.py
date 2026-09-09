# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb oracle: diamond (fcc) lattice dielectric-sphere bands.

mpb_diamond.py uses the fcc *primitive* cell (2 spheres, basis vectors
(0,1,1),(1,0,1),(1,1,0) at basis_size sqrt(0.5), conventional cubic
constant a_conv = sqrt(2), sphere r = 0.25). For the meep-torch rect-
supercell trick we instead use the *conventional cubic cell* of side 1
(a_conv rescaled to 1), which contains the 8 diamond atoms:

    fcc sublattice:        (0,0,0) (0,1/2,1/2) (1/2,0,1/2) (1/2,1/2,0)
    + (1/4,1/4,1/4) shift: (1/4,1/4,1/4) (1/4,3/4,3/4) (3/4,1/4,3/4) (3/4,3/4,1/4)

r rescaled to 0.25/sqrt(2) so r/a_conv is identical to the mpb example
(eps=11.56 unchanged). The cubic supercell BZ is 1/4 the fcc BZ, so the
fcc bands fold into it; the oracle uses the SAME supercell + k as the
meep-torch side, so the gate is self-consistent (both folded identically).

k point: simple-cubic X = (0, 1/2, 1/2) -> k = (0, pi, pi) for cell=1.

Run: mamba run -n meep-oracle python tests/examples/data/generate_diamond_mpb.py
"""
import json
import math
import os

import meep as mp
from meep import mpb

HERE = os.path.dirname(__file__)

A_CONV = 1.0                       # conventional cubic cell side (rescaled)
R = 0.25 / math.sqrt(2)            # sphere radius (r/a_conv == mpb_diamond 0.25/sqrt2)
EPS = 11.56

# 8 diamond atoms in the conventional cubic cell (fcc + (1/4,1/4,1/4) offset).
FCC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
OFF = (0.25, 0.25, 0.25)
geometry = []
for cx, cy, cz in FCC:
    geometry.append(mp.Sphere(R, center=mp.Vector3(cx, cy, cz),
                              material=mp.Medium(epsilon=EPS)))
    geometry.append(mp.Sphere(R,
                              center=mp.Vector3(cx + OFF[0], cy + OFF[1],
                                                cz + OFF[2]),
                              material=mp.Medium(epsilon=EPS)))

ms = mpb.ModeSolver(
    num_bands=4,
    resolution=mp.Vector3(16, 16, 16),
    geometry_lattice=mp.Lattice(size=mp.Vector3(A_CONV, A_CONV, A_CONV)),
    geometry=geometry,
    k_points=[mp.Vector3(0, 0.5, 0.5)],   # SC X -> k=(0, pi, pi)
)
ms.run()

bands = [list(map(float, row)) for row in ms.all_freqs]
out = {"a_conv": A_CONV, "r": R, "eps": EPS, "k": [0.0, 0.5, 0.5],
       "bands": bands}
json.dump(out, open(os.path.join(HERE, "diamond_mpb.json"), "w"), indent=1)
print("diamond SC-X bands:", bands[0])

# --- fcc PRIMITIVE cell, added 2026-09-09 ---------------------------------
#
# The conventional-cell oracle above exists because the meep-torch side
# could only rasterize a cartesian box. The 3D planewave backend solves the
# oblique primitive cell directly, so it wants an oracle on the SAME cell
# rather than a folded one -- a supercell gate cannot tell a correct
# oblique operator from one that happens to fold the same way.
#
# This is mpb_diamond.py's own cell: 2 spheres, basis (0,1,1)/(1,0,1)/(1,1,0)
# at basis_size sqrt(0.5), r = 0.25, at the high-symmetry points the
# example visits.
def _primitive():
    import math

    ms_p = mpb.ModeSolver(
        num_bands=5,
        resolution=16,
        mesh_size=5,
        geometry_lattice=mp.Lattice(
            basis_size=mp.Vector3(math.sqrt(0.5), math.sqrt(0.5),
                                  math.sqrt(0.5)),
            basis1=mp.Vector3(0, 1, 1),
            basis2=mp.Vector3(1, 0, 1),
            basis3=mp.Vector3(1, 1, 0)),
        geometry=[
            mp.Sphere(0.25, center=mp.Vector3(0.125, 0.125, 0.125),
                      material=mp.Medium(epsilon=EPS)),
            mp.Sphere(0.25, center=mp.Vector3(-0.125, -0.125, -0.125),
                      material=mp.Medium(epsilon=EPS))],
        k_points=[mp.Vector3(0, 0.5, 0.5),        # X
                  mp.Vector3(0, 0.625, 0.375),    # U
                  mp.Vector3(0, 0.5, 0),          # L
                  mp.Vector3(0, 0, 0),            # Gamma
                  mp.Vector3(0, 0.5, 0.25)],      # W
    )
    ms_p.run()
    return {"r": 0.25, "eps": EPS, "resolution": 16, "mesh_size": 5,
            "basis_size": math.sqrt(0.5),
            "k": [[0, 0.5, 0.5], [0, 0.625, 0.375], [0, 0.5, 0],
                  [0, 0, 0], [0, 0.5, 0.25]],
            "bands": [list(map(float, row)) for row in ms_p.all_freqs]}


if __name__ == "__main__":
    json.dump(_primitive(),
              open(os.path.join(HERE, "diamond_primitive_mpb.json"), "w"),
              indent=1)
    print("wrote diamond_primitive_mpb.json")
