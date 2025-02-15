# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: holey-wvg-bands k-sweep (Hz-pol, reduced).

Run: mamba run -n meep-oracle python tests/examples/data/generate_holey_bands.py
-> tests/examples/data/holey_bands.json
"""
import contextlib
import io
import json
import os
import re

import meep as mp

RES = 10
EPS, W, R = 13.0, 1.2, 0.36
SY, DPML = 12.0, 1.0
FCEN, DF = 0.25, 1.5
KX = [0.1, 0.2, 0.3, 0.4, 0.5]


def bands_at(kx):
    sim = mp.Simulation(
        cell_size=mp.Vector3(1, SY), resolution=RES,
        geometry=[mp.Block(size=mp.Vector3(mp.inf, W, mp.inf),
                           material=mp.Medium(epsilon=EPS)),
                  mp.Cylinder(radius=R)],
        boundary_layers=[mp.PML(DPML, direction=mp.Y)],
        k_point=mp.Vector3(kx),
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=mp.Hz,
                           center=mp.Vector3(0.1234, 0.2345))])
    h = mp.Harminv(mp.Hz, mp.Vector3(0.1234, 0.2345), FCEN, DF)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.run(mp.after_sources(h), until_after_sources=300)
    out = []
    for line in buf.getvalue().splitlines():
        g = re.match(r"harminv\d+:,\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)", line)
        if g:
            f, q, a = float(g.group(1)), float(g.group(3)), float(g.group(4))
            if f > 0 and q > 50:
                out.append({"freq": f, "Q": q, "amp": a})
    out.sort(key=lambda x: x["freq"])
    return out


data = {"resolution": RES, "eps": EPS, "w": W, "r": R, "sy": SY,
        "dpml": DPML, "fcen": FCEN, "df": DF,
        "bands": {str(k): bands_at(k) for k in KX}}
path = os.path.join(os.path.dirname(__file__), "holey_bands.json")
with open(path, "w") as f:
    json.dump(data, f, indent=1)
print("wrote", path)
for k in KX:
    fs = [round(m["freq"], 4) for m in data["bands"][str(k)][:3]]
    print(f"kx={k}: {fs}")
