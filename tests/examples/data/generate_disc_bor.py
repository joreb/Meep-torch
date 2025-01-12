# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle reference: disc resonator in 2D cylindrical (r, z), m = 3.

Er-polarized (thin-disc TE-like modes; an Ez drive finds nothing).

Disc radius 1.2, thickness 0.29, n = 2.4 (disc_radiation_pattern
geometry, reduced resolution 25); Ez harminv at a point inside the disc.

Run: mamba run -n meep-oracle python tests/examples/data/generate_disc_bor.py
-> tests/examples/data/disc_bor.json
"""
import contextlib
import io
import json
import os
import re

import meep as mp

RES = 25
M = 3
N_DISC = 2.4
R_DISC = 1.2
H_DISC = 0.29
FCEN, DF = 1.6, 0.5
PAD, DPML = 1.0, 0.5

sr = R_DISC + PAD + DPML
sz = H_DISC + 2 * (PAD + DPML)
sim = mp.Simulation(
    cell_size=mp.Vector3(sr, 0, sz), dimensions=mp.CYLINDRICAL, m=M,
    resolution=RES, boundary_layers=[mp.PML(DPML)],
    geometry=[mp.Block(center=mp.Vector3(0.5 * R_DISC, 0, 0),
                       size=mp.Vector3(R_DISC, mp.inf, H_DISC),
                       material=mp.Medium(index=N_DISC))],
    sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=mp.Er,
                       center=mp.Vector3(0.8 * R_DISC, 0, 0))])
h = mp.Harminv(mp.Er, mp.Vector3(0.8 * R_DISC, 0, 0), FCEN, DF)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sim.run(mp.after_sources(h), until_after_sources=300)
modes = []
for line in buf.getvalue().splitlines():
    g = re.match(r"harminv\d+:,\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+),\s*([-\d.eE+]+)", line)
    if g:
        modes.append({"freq": float(g.group(1)), "Q": float(g.group(3)),
                      "amp": float(g.group(4))})
modes.sort(key=lambda x: -x["amp"])
out = {"resolution": RES, "m": M, "n": N_DISC, "r_disc": R_DISC,
       "h_disc": H_DISC, "fcen": FCEN, "df": DF, "pad": PAD, "dpml": DPML,
       "modes": modes}
path = os.path.join(os.path.dirname(__file__), "disc_bor.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path)
print("dominant:", modes[0] if modes else None)
