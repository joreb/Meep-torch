# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: metal-cavity-ldos (w=0.2 aperture cavity).

Mirrors meep/python/examples/metal-cavity-ldos.py for one aperture width:
harminv f/Q of the cavity mode and the dft_ldos spectrum around it.
No symmetries (matches the meep-torch full-cell port).

Run: mamba run -n meep-oracle python tests/examples/data/generate_metal_cavity_ldos.py
-> tests/examples/data/metal_cavity_ldos.json
"""
import contextlib
import io
import json
import math
import os
import re

import meep as mp
import numpy as np

W = 0.2
RESOLUTION = 25          # reduced from 50 (runtime); noted in JSON
NFREQ = 25


def make_sim():
    sxy = 2 + 2 * 1      # cavity region + PML
    a, t = 1.0, 0.1
    geometry = [
        mp.Block(mp.Vector3(a + 2 * t, a + 2 * t, mp.inf), material=mp.metal),
        mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
        mp.Block(center=mp.Vector3(a / 2), size=mp.Vector3(2 * t, W, mp.inf),
                 material=mp.air),
    ]
    fcen = math.sqrt(0.5)
    return mp.Simulation(
        cell_size=mp.Vector3(sxy, sxy), geometry=geometry,
        eps_averaging=False,     # binary staircase, matches meep-torch
        boundary_layers=[mp.PML(1.0)], resolution=RESOLUTION,
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2),
                           component=mp.Ez, center=mp.Vector3())]), fcen


sim, fcen = make_sim()
h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, 0.2)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sim.run(mp.after_sources(h), until_after_sources=500)
modes = []
for line in buf.getvalue().splitlines():
    m = re.match(r"harminv\d+:,\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)", line)
    if m:
        modes.append({"freq": float(m.group(1)), "Q": float(m.group(3)),
                      "amp": float(m.group(4))})
modes.sort(key=lambda m: -m["amp"])
f_mode, q_mode = modes[0]["freq"], modes[0]["Q"]

sim, _ = make_sim()
T = 2 * q_mode / f_mode
sim.run(mp.dft_ldos(f_mode, 0.05, NFREQ), until_after_sources=T)
freqs = np.linspace(f_mode - 0.025, f_mode + 0.025, NFREQ)

out = {"w": W, "resolution": RESOLUTION, "harminv_f": f_mode, "harminv_Q": q_mode,
       "ldos_freqs": freqs.tolist(), "ldos": np.array(sim.ldos_data).tolist()}
path = os.path.join(os.path.dirname(__file__), "metal_cavity_ldos.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path, "f=", f_mode, "Q=", q_mode)
