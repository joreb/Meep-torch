# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: off-axis Er dipole (m = 1) radiation pattern
via cylindrical near2far.

Run: mamba run -n meep-oracle python tests/examples/data/generate_dipole_m1_n2f.py
-> tests/examples/data/dipole_m1_n2f.json
"""
import json
import math
import os

import meep as mp
import numpy as np

RES = 25
FCEN, DF = 1.0, 0.2
SR, SZ, DPML = 3.0, 6.0, 1.0
R_FAR = 1000.0
N_THETA = 50

sim = mp.Simulation(
    cell_size=mp.Vector3(SR, 0, SZ), dimensions=mp.CYLINDRICAL, m=1,
    resolution=RES, boundary_layers=[mp.PML(DPML)],
    sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=mp.Er,
                       center=mp.Vector3(0.5, 0, 0))])
n2f = sim.add_near2far(FCEN, 0, 1,
    mp.Near2FarRegion(center=mp.Vector3(1.5, 0, 0), size=mp.Vector3(0, 0, 3.0)),
    mp.Near2FarRegion(center=mp.Vector3(0.75, 0, 1.5), size=mp.Vector3(1.5, 0, 0)),
    mp.Near2FarRegion(center=mp.Vector3(0.75, 0, -1.5), size=mp.Vector3(1.5, 0, 0),
                      weight=-1))
sim.run(until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Er, mp.Vector3(0.5, 0, 0.4), 1e-8))

thetas = np.linspace(math.radians(5), math.radians(175), N_THETA)
flux = []
for th in thetas:
    ff = sim.get_farfield(n2f, mp.Vector3(R_FAR * math.sin(th), 0,
                                          R_FAR * math.cos(th)))
    e = np.array(ff[:3])
    h = np.array(ff[3:])
    s = np.real(np.cross(np.conj(e), h))
    rhat = np.array([math.sin(th), 0, math.cos(th)])
    flux.append(float(np.dot(s, rhat)))
flux = np.array(flux)
out = {"resolution": RES, "fcen": FCEN, "r_src": 0.5, "m": 1,
       "theta_deg": np.degrees(thetas).tolist(),
       "pattern": (flux / flux.max()).tolist()}
path = os.path.join(os.path.dirname(__file__), "dipole_m1_n2f.json")
with open(path, "w") as f:
    json.dump(out, f)
print("wrote", path)
