# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: faraday-rotation Ex/Ey profiles (tutorial setup).

Run: mamba run -n meep-oracle python tests/examples/data/generate_faraday.py
-> tests/examples/data/faraday.json (Ex(z), Ey(z) at t=100, res 50).
"""
import json
import os

import meep as mp
import numpy as np

epsn, f0, gamma, sn, b0 = 1.5, 1.0, 1e-6, 0.1, 0.15
fsrc, src_z, L, tmax = 0.8, -8.5, 20.0, 100

susc = [mp.GyrotropicLorentzianSusceptibility(
    frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0, 0, b0))]
mat = mp.Medium(epsilon=epsn, mu=1, E_susceptibilities=susc)

sim = mp.Simulation(
    cell_size=mp.Vector3(0, 0, L), resolution=50, default_material=mat,
    boundary_layers=[mp.PML(thickness=1.0, direction=mp.Z)],
    sources=[mp.Source(mp.ContinuousSource(frequency=fsrc), component=mp.Ex,
                       center=mp.Vector3(0, 0, src_z))])
sim.run(until=tmax)

out = {"epsn": epsn, "f0": f0, "gamma": gamma, "sn": sn, "b0": b0,
       "fsrc": fsrc, "src_z": src_z, "L": L, "tmax": tmax, "resolution": 50,
       "ex": np.real(sim.get_efield_x()).tolist(),
       "ey": np.real(sim.get_efield_y()).tolist()}
path = os.path.join(os.path.dirname(__file__), "faraday.json")
with open(path, "w") as f:
    json.dump(out, f)
print("wrote", path, len(out["ex"]), "points")
