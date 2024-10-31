# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference for binary_grating_oblique: reflected + transmitted
diffraction orders (angle, power). Run via `mamba run -n meep-oracle python
tests/examples/data/generate_binary_grating_oblique.py` -> writes
tests/examples/data/binary_grating_oblique.json. Mirrors meep/python/examples/
binary_grating_oblique.py (gp=10, gh=0.5, gdc=0.5, ng=1.5, wvl=0.5, theta=10.7deg).
"""
import json
import math
import os

import numpy as np

import meep as mp

resolution = 50
dpml, dsub, dpad = 1.0, 3.0, 3.0
gp, gh, gdc = 10.0, 0.5, 0.5
sx = dpml + dsub + gh + dpad + dpml
sy = gp
wvl = 0.5
fcen = 1 / wvl
df = 0.05 * fcen
ng = 1.5
glass = mp.Medium(index=ng)
theta_in = math.radians(10.7)
k = mp.Vector3(fcen * ng).rotate(mp.Vector3(z=1), theta_in)
eig_parity = mp.ODD_Z


def pw_amp(kv, x0):
    return lambda x: cmath.exp(1j * 2 * math.pi * kv.dot(x + x0))


import cmath  # noqa: E402

src_pt = mp.Vector3(-0.5 * sx + dpml + 0.3 * dsub, 0, 0)
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                    center=src_pt, size=mp.Vector3(0, sy, 0), amp_func=pw_amp(k, src_pt))]
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

sim = mp.Simulation(resolution=resolution, cell_size=mp.Vector3(sx, sy, 0),
                    boundary_layers=pml_layers, k_point=k, default_material=glass,
                    sources=sources)
refl_pt = mp.Vector3(-0.5 * sx + dpml + 0.5 * dsub, 0, 0)
refl_flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, sy, 0)))
sim.run(until_after_sources=100)
input_flux = mp.get_fluxes(refl_flux)
input_flux_data = sim.get_flux_data(refl_flux)
sim.reset_meep()

geometry = [
    mp.Block(material=glass, size=mp.Vector3(dpml + dsub, mp.inf, mp.inf),
             center=mp.Vector3(-0.5 * sx + 0.5 * (dpml + dsub), 0, 0)),
    mp.Block(material=glass, size=mp.Vector3(gh, gdc * gp, mp.inf),
             center=mp.Vector3(-0.5 * sx + dpml + dsub + 0.5 * gh, 0, 0)),
]
sim = mp.Simulation(resolution=resolution, cell_size=mp.Vector3(sx, sy, 0),
                    boundary_layers=pml_layers, geometry=geometry, k_point=k,
                    sources=sources)
refl_flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=refl_pt, size=mp.Vector3(0, sy, 0)))
sim.load_minus_flux_data(refl_flux, input_flux_data)
tran_pt = mp.Vector3(0.5 * sx - dpml - 0.5 * dpad, 0, 0)
tran_flux = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=tran_pt, size=mp.Vector3(0, sy, 0)))
sim.run(until_after_sources=200)

nm_r = int(np.floor((fcen * ng - k.y) * gp) - np.ceil((-fcen * ng - k.y) * gp))
res = sim.get_eigenmode_coefficients(refl_flux, range(1, nm_r + 1), eig_parity=eig_parity)
refl = []
for nm in range(nm_r):
    r_kdom = res.kdom[nm]
    Rmode = abs(res.alpha[nm, 0, 1]) ** 2 / input_flux[0]
    r_angle = math.degrees(np.sign(r_kdom.y) * math.acos(r_kdom.x / (ng * fcen)))
    refl.append({"order": nm, "angle": r_angle, "R": Rmode})

nm_t = int(np.floor((fcen - k.y) * gp) - np.ceil((-fcen - k.y) * gp))
res = sim.get_eigenmode_coefficients(tran_flux, range(1, nm_t + 1), eig_parity=eig_parity)
tran = []
for nm in range(nm_t):
    t_kdom = res.kdom[nm]
    Tmode = abs(res.alpha[nm, 0, 0]) ** 2 / input_flux[0]
    t_angle = math.degrees(np.sign(t_kdom.y) * math.acos(t_kdom.x / fcen))
    tran.append({"order": nm, "angle": t_angle, "T": Tmode})

out = {"theta_in_deg": 10.7, "ng": ng, "gp": gp, "gh": gh, "gdc": gdc, "wvl": wvl,
       "refl": refl, "tran": tran}
path = os.path.join(os.path.dirname(__file__), "binary_grating_oblique.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path)
print("refl orders:", [(r["angle"], round(r["R"], 6)) for r in refl])
print("tran orders:", [(t["angle"], round(t["T"], 6)) for t in tran])