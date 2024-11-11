# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W4 gate (c): antenna-radiation dipole patterns vs meep-oracle JSON."""
import json
import math
import pathlib

import numpy as np

import meep_torch.mp as mp

DATA = pathlib.Path(__file__).parent / "data" / "antenna_radiation_n2f.json"
FREQ = 1.0
R_FAR = 1000.0


def _oracle():
    with open(DATA) as f:
        return json.load(f)


def _pattern_tm():
    cell = 4.0
    sim = mp.Simulation(cell_size=mp.Vector3(6, 6, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FREQ, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    n2f = sim.add_near2far(FREQ, 0, 1,
        mp.Near2FarRegion(center=mp.Vector3(0, +cell / 2), size=mp.Vector3(cell, 0), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -cell / 2), size=mp.Vector3(cell, 0), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(+cell / 2, 0), size=mp.Vector3(0, cell), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-cell / 2, 0), size=mp.Vector3(0, cell), weight=-1))
    sim.run(until=80)
    th = np.linspace(0, 2 * math.pi, 100, endpoint=False)
    ff = np.array([sim.get_farfield(n2f, mp.Vector3(R_FAR * math.cos(t), R_FAR * math.sin(t)))
                   for t in th])
    e, h = ff[:, :3], ff[:, 3:]
    sx = np.real(np.conj(e[:, 1]) * h[:, 2] - np.conj(e[:, 2]) * h[:, 1])
    sy = np.real(np.conj(e[:, 2]) * h[:, 0] - np.conj(e[:, 0]) * h[:, 2])
    return np.sqrt(sx ** 2 + sy ** 2)


def test_ez_pattern_vs_oracle():
    ref = np.array(_oracle()["flux_ez"])
    got = _pattern_tm()
    ref = ref / ref.max()
    got = got / got.max()
    rms = float(np.sqrt(np.mean((got - ref) ** 2)))
    assert rms <= 0.05, f"normalized rms vs oracle {rms:.4f}"
