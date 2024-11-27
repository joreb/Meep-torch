# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: radiation pressure on a metal mirror (2D, periodic y).

Plane wave onto a mp.metal mirror; add_force box around it. JSON stores the
dimensionless pressure ratio F / ((1+R) S_inc Ly) per freq — the same
quantity the meep-torch gate computes, so engine normalizations drop out.

Run: mamba run -n meep-oracle python tests/examples/data/generate_radiation_force.py
-> tests/examples/data/radiation_force.json
"""
import json
import os

import meep as mp
import numpy as np

RES = 20
FCEN, DF, NF = 1.0, 0.2, 3
LY = 0.4


def run(with_mirror):
    geometry = []
    if with_mirror:
        geometry = [mp.Block(size=mp.Vector3(1.0, mp.inf, mp.inf),
                             center=mp.Vector3(2.5, 0), material=mp.metal)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(30, LY, 0), resolution=RES, geometry=geometry,
        k_point=mp.Vector3(),                     # periodic y
        boundary_layers=[mp.PML(1.5, direction=mp.X)],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=mp.Ez,
                           center=mp.Vector3(-9, 0), size=mp.Vector3(0, LY))])
    frc = sim.add_force(FCEN, DF, NF,
        mp.ForceRegion(center=mp.Vector3(1.0, 0), size=mp.Vector3(0, LY),
                       direction=mp.X, weight=-1),
        mp.ForceRegion(center=mp.Vector3(4.0, 0), size=mp.Vector3(0, LY),
                       direction=mp.X, weight=+1))
    flx = sim.add_flux(FCEN, DF, NF,
                       mp.FluxRegion(center=mp.Vector3(1.0, 0), size=mp.Vector3(0, LY)))
    probe = mp.Vector3(1.0, 0)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, probe, 1e-6))
    return (np.array(mp.get_force_freqs(frc)), np.array(mp.get_forces(frc)),
            np.array(mp.get_fluxes(flx)))


freqs, F_m, flux_m = run(True)
_, F_0, flux_0 = run(False)
# flux with mirror = S_inc - S_refl = S_inc (1 - R); without = S_inc
R = 1.0 - flux_m / flux_0
s_inc = flux_0 / LY
ratio = F_m / ((1 + R) * s_inc * LY)
out = {"resolution": RES, "freqs": freqs.tolist(), "R": R.tolist(),
       "force": F_m.tolist(), "flux_inc": flux_0.tolist(),
       "pressure_ratio": ratio.tolist()}
path = os.path.join(os.path.dirname(__file__), "radiation_force.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path, "ratio:", ratio)
