# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Regenerate tests/ports/data/meep_reference.json.

Run: mamba run -n meep-oracle python tests/ports/generate_meep_reference.py
Requires pymeep (conda-forge, env meep-oracle). Not a test-time dependency.

The physical geometry mirrors the meep-torch Task 6/8 test geometries
(straight eps=12 waveguide width 1um; linear taper width 1->2um over 4um).
Source pulse fwidth=0.1 (meep-matched: w=1/fwidth, same convention as our
GaussianPulse). The mode-monitor frequency span is 0.04 so the 3 DFT
frequencies are exactly [0.13, 0.15, 0.17] (decoupled from the source fwidth).
"""
import json
import pathlib

import meep as mp

FCEN = 0.15
SOURCE_FWIDTH = 0.1
MON_FWIDTH = 0.04  # -> DFT freqs [0.13, 0.15, 0.17]
FREQS = [0.13, 0.15, 0.17]
NF = len(FREQS)
RES = 10  # cells/um == dx 0.1
PML = 1.0  # um (10 cells at RES=10) -- matches our npml=10


def _sim(geom_fn, sx, sy):
    cell = mp.Vector3(sx, sy, 0)
    sim = mp.Simulation(
        cell_size=cell, resolution=RES, boundary_layers=[mp.PML(PML)],
        geometry=geom_fn(),
        sources=[mp.EigenModeSource(
            mp.GaussianSource(FCEN, fwidth=SOURCE_FWIDTH),
            center=mp.Vector3(-sx / 2 + 4.0, 0), size=mp.Vector3(0, sy - 2),
            eig_parity=mp.ODD_Z,  # TM (Ez) in meep 2D
        )],
    )
    mon_in = sim.add_mode_monitor(
        FCEN, MON_FWIDTH, NF,
        mp.ModeRegion(center=mp.Vector3(-sx / 2 + 5.0, 0), size=mp.Vector3(0, sy - 2)))
    mon_out = sim.add_mode_monitor(
        FCEN, MON_FWIDTH, NF,
        mp.ModeRegion(center=mp.Vector3(sx / 2 - 4.0, 0), size=mp.Vector3(0, sy - 2)))
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(sx / 2 - 4.0, 0), 1e-8))
    # alpha shape (num_bands, nf, 2): [:, :, 0] forward, [:, :, 1] backward
    ci = sim.get_eigenmode_coefficients(mon_in, [1], eig_parity=mp.ODD_Z).alpha
    co = sim.get_eigenmode_coefficients(mon_out, [1], eig_parity=mp.ODD_Z).alpha
    s21 = [abs(co[0, i, 0] / ci[0, i, 0]) for i in range(NF)]
    s11 = [abs(ci[0, i, 1] / ci[0, i, 0]) for i in range(NF)]
    return {"freqs": FREQS, "S21_mag": s21, "S11_mag": s11}


def straight_geom():
    return [mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                     material=mp.Medium(epsilon=12))]


def taper_geom():
    # width 1 -> 2 um linear taper over 4 um centered at origin (matches Task 6 test)
    return [
        mp.Block(mp.Vector3(10, 1, mp.inf), center=mp.Vector3(-7, 0),
                 material=mp.Medium(epsilon=12)),
        mp.Prism([mp.Vector3(-2, -0.5), mp.Vector3(2, -1), mp.Vector3(2, 1),
                  mp.Vector3(-2, 0.5)], height=mp.inf,
                 material=mp.Medium(epsilon=12)),
        mp.Block(mp.Vector3(10, 2, mp.inf), center=mp.Vector3(7, 0),
                 material=mp.Medium(epsilon=12)),
    ]


if __name__ == "__main__":
    out = {
        "straight": _sim(straight_geom, 20, 8),
        "taper": _sim(taper_geom, 24, 10),
        "provenance": {
            "meep_version": mp.__version__,
            "command": "mamba run -n meep-oracle python tests/ports/generate_meep_reference.py",
        },
    }
    p = pathlib.Path(__file__).parent / "data" / "meep_reference.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
    print(json.dumps(out, indent=1))