# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle harminv reference for the holey-wvg-cavity defect mode (TM, Ez).

Mirrors meep/python/examples/holey-wvg-cavity.py resonant_modes but in Ez (TM,
no symmetries) to match the meep-torch TM port. Run via `mamba run -n meep-oracle
python tests/examples/data/generate_holey_wvg_cavity_harminv.py` -> writes
tests/examples/data/holey_wvg_cavity_harminv.json (dominant mode f, Q).
"""
import json
import os

import meep as mp

resolution = 20
eps = 13
w = 1.2
r = 0.36
d = 1.4
N = 3
sy = 6
pad = 2
dpml = 1
sx = 2 * (pad + dpml + N) + d - 1
fcen, df = 0.25, 0.2

cell = mp.Vector3(sx, sy, 0)
blk = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=mp.Medium(epsilon=eps))
geometry = [blk]
for i in range(N):
    geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
    geometry.append(mp.Cylinder(r, center=mp.Vector3(-(d / 2 + i))))

sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=[],
                    boundary_layers=[mp.PML(dpml)], resolution=resolution)
sim.sources.append(mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez,
                             center=mp.Vector3()))
hv = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)

import io  # noqa: E402
import contextlib  # noqa: E402

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sim.run(mp.after_sources(hv), until_after_sources=400)

# parse meep's printed "harminv0:, freq, imag.freq, Q, |amp|, amplitude, error" lines
import re  # noqa: E402

modes = []
for line in buf.getvalue().splitlines():
    m = re.match(r"harminv\d+:,\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)",
                 line)
    if m:
        modes.append({"freq": float(m.group(1)), "Q": float(m.group(3)),
                       "amp": float(m.group(4))})
if not modes:
    raise RuntimeError("no harminv lines parsed")
modes.sort(key=lambda m: -m["amp"])
out = {"N": N, "sy": sy, "eps": eps, "w": w, "r": r, "d": d, "fcen": fcen, "df": df,
       "resolution": resolution, "modes": modes}
path = os.path.join(os.path.dirname(__file__), "holey_wvg_cavity_harminv.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path)
print("dominant:", modes[0])