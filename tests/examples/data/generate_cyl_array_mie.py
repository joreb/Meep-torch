# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle: scattered-flux 'Q' of a periodic cylinder array (pitch 12),
same two-run box method as the meep-torch ports (apples-to-apples -- the
wrapped-copy interference is part of the measured quantity).

Run: mamba run -n meep-oracle python tests/examples/data/generate_cyl_array_mie.py
"""
import json
import os

import meep as mp
import numpy as np

RES, N_CYL, R_CYL, PITCH = 20, 2.0, 0.7, 12.0
FREQS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def q_at(f):
    def build(with_cyl):
        geometry = ([mp.Cylinder(R_CYL, material=mp.Medium(epsilon=N_CYL ** 2),
                                 center=mp.Vector3(1.0, 0))] if with_cyl else [])
        sim = mp.Simulation(
            cell_size=mp.Vector3(10, PITCH, 0), resolution=RES,
            geometry=geometry, k_point=mp.Vector3(),
            eps_averaging=True,
            boundary_layers=[mp.PML(1.0, direction=mp.X)],
            sources=[mp.Source(mp.GaussianSource(f, fwidth=0.2), component=mp.Ez,
                               center=mp.Vector3(-3.0, 0), size=mp.Vector3(0, PITCH))])
        box = sim.add_flux(f, 0, 1,
            mp.FluxRegion(center=mp.Vector3(1.0 + 1.5, 0), size=mp.Vector3(0, 3), weight=+1),
            mp.FluxRegion(center=mp.Vector3(1.0 - 1.5, 0), size=mp.Vector3(0, 3), weight=-1),
            mp.FluxRegion(center=mp.Vector3(1.0, 1.5), size=mp.Vector3(3, 0), weight=+1),
            mp.FluxRegion(center=mp.Vector3(1.0, -1.5), size=mp.Vector3(3, 0), weight=-1))
        inc = sim.add_flux(f, 0, 1,
            mp.FluxRegion(center=mp.Vector3(1.0 - 1.5, 0), size=mp.Vector3(0, 3)))
        return sim, box, inc

    def go(sim):
        sim.run(until_after_sources=mp.stop_when_fields_decayed(
            25, mp.Ez, mp.Vector3(1.0, 0), 1e-7))

    sim0, box0, inc0 = build(False)
    go(sim0)
    s_inc = abs(np.array(mp.get_fluxes(inc0))[0]) / 3.0
    saved = sim0.get_flux_data(box0)
    sim1, box1, _ = build(True)
    sim1.load_minus_flux_data(box1, saved)     # BEFORE the run
    go(sim1)
    f_sc = abs(np.array(mp.get_fluxes(box1))[0])
    return f_sc / s_inc / (2 * R_CYL)


out = {"resolution": RES, "n": N_CYL, "r": R_CYL, "pitch": PITCH,
       "freqs": FREQS, "Q": [q_at(f) for f in FREQS]}
json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                 "cyl_array_mie.json"), "w"), indent=1)
print("Q:", [round(q, 3) for q in out["Q"]])
