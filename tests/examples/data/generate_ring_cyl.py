# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: ring-cyl resonances (cylindrical, m = 3 and 4).

Run: mamba run -n meep-oracle python tests/examples/data/generate_ring_cyl.py
-> tests/examples/data/ring_cyl.json
"""
import contextlib
import io
import json
import os
import re

import meep as mp

N_IDX, W, R, PAD, DPML = 3.4, 1.0, 1.0, 4.0, 32.0
RES = 20
FCEN, DF = 0.15, 0.1


def modes_for(m):
    sr = R + W + PAD + DPML
    sim = mp.Simulation(
        cell_size=mp.Vector3(sr, 0, 0), dimensions=mp.CYLINDRICAL, m=m,
        resolution=RES, boundary_layers=[mp.PML(DPML)],
        geometry=[mp.Block(center=mp.Vector3(R + W / 2),
                           size=mp.Vector3(W, mp.inf, mp.inf),
                           material=mp.Medium(index=N_IDX))],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=mp.Ez,
                           center=mp.Vector3(R + 0.1))])
    h = mp.Harminv(mp.Ez, mp.Vector3(R + 0.1), FCEN, DF)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.run(mp.after_sources(h), until_after_sources=200)
    out = []
    for line in buf.getvalue().splitlines():
        g = re.match(r"harminv\d+:,\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)", line)
        if g:
            out.append({"freq": float(g.group(1)), "Q": float(g.group(3)),
                        "amp": float(g.group(4))})
    out.sort(key=lambda x: -x["amp"])
    return out


data = {"n": N_IDX, "w": W, "r": R, "pad": PAD, "dpml": DPML,
        "resolution": RES, "fcen": FCEN, "df": DF,
        "m3": modes_for(3), "m4": modes_for(4)}
path = os.path.join(os.path.dirname(__file__), "ring_cyl.json")
with open(path, "w") as f:
    json.dump(data, f, indent=2)
print("wrote", path)
print("m3 dominant:", data["m3"][0] if data["m3"] else None)
print("m4 dominant:", data["m4"][0] if data["m4"] else None)
