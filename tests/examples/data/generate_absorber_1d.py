# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference for absorber-1d: Absorber vs PML ringdown residual.

SUBSTITUTION vs meep/python/examples/absorber-1d.py: the metal is a
conductive medium (eps=1, D_conductivity=1) instead of meep.materials.Al --
Al's Drude pole (w0=0) has no meep-torch ADE representation, and the point
of the example (Absorber works where PML is used) survives with any lossy
background. Same material on both engines keeps the gate apples-to-apples.

Run: mamba run -n meep-oracle python tests/examples/data/generate_absorber_1d.py
-> tests/examples/data/absorber_1d.json (peak + trailing residual per mode).
"""
import json
import os

import meep as mp
import numpy as np

RES = 40
FCEN, DF = 1.0 / 0.803, 0.1
SIGMA_D = 1.0


def run(use_pml):
    bl = [mp.PML(1.0)] if use_pml else [mp.Absorber(1.0)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(z=10), resolution=RES, dimensions=1,
        default_material=mp.Medium(epsilon=1, D_conductivity=SIGMA_D),
        boundary_layers=bl,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                           center=mp.Vector3(), component=mp.Ex)])
    series = []
    # GaussianSource width 1/df = 10 -> cutoff ~ t = 100; measure after t = 160
    sim.run(mp.at_every(0.5, lambda s: series.append(
        abs(s.get_field_point(mp.Ex, mp.Vector3()).real))), until=240)
    a = np.array(series)
    peak = float(a.max())
    residual = float(a[2 * len(a) // 3:].max())      # after t = 160
    return peak, residual


p_abs, r_abs = run(False)
p_pml, r_pml = run(True)
out = {"resolution": RES, "fcen": FCEN, "df": DF, "sigma_d": SIGMA_D,
       "absorber": {"peak": p_abs, "residual": r_abs},
       "pml": {"peak": p_pml, "residual": r_pml}}
path = os.path.join(os.path.dirname(__file__), "absorber_1d.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path, out)
