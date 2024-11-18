# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: 1D transmission through a D-conductivity slab.

L=2 slab of eps=1, D_conductivity=1 in vacuum; power transmission spectrum
over the pulse band via two runs (with/without slab). Run via
`mamba run -n meep-oracle python tests/examples/data/generate_conductive_slab.py`
-> writes tests/examples/data/conductive_slab.json.
"""
import json
import os

import meep as mp
import numpy as np

RES = 40
FCEN, DF = 1.0, 0.2
NFREQ = 5
L = 2.0


def run(with_slab):
    cell = mp.Vector3(z=30)
    geometry = []
    if with_slab:
        geometry = [mp.Block(size=mp.Vector3(z=L), center=mp.Vector3(),
                             material=mp.Medium(epsilon=1, D_conductivity=1.0))]
    sim = mp.Simulation(cell_size=cell, resolution=RES, dimensions=1,
                        geometry=geometry,
                        boundary_layers=[mp.PML(1.5)],
                        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                                           component=mp.Ex,
                                           center=mp.Vector3(z=-10))])
    fr = sim.add_flux(FCEN, DF, NFREQ,
                      mp.FluxRegion(center=mp.Vector3(z=6)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(z=6), 1e-8))
    return np.array(mp.get_flux_freqs(fr)), np.array(mp.get_fluxes(fr))


freqs, p_slab = run(True)
_, p_ref = run(False)
out = {"resolution": RES, "fcen": FCEN, "df": DF, "L": L, "sigma_d": 1.0,
       "freqs": freqs.tolist(), "T_power": (p_slab / p_ref).tolist()}
path = os.path.join(os.path.dirname(__file__), "conductive_slab.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path, out["T_power"])
