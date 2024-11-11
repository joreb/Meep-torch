# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-oracle near2far reference: antenna-radiation dipole patterns (2D).

Mirrors meep/python/examples/antenna-radiation.py at reduced resolution 25:
Ez and Ey point dipoles in vacuum, radial Poynting flux over 100 polar angles
at r = 1000 lambda via add_near2far/get_farfield. Run via
`mamba run -n meep-oracle python tests/examples/data/generate_antenna_radiation_n2f.py`
-> writes tests/examples/data/antenna_radiation_n2f.json.
"""
import json
import math
import os

import meep as mp
import numpy as np

RESOLUTION = 25
PML_UM = 1.0
WAVELENGTH_UM = 1.0
NUM_POLAR = 100
FARFIELD_RADIUS = 1000 * WAVELENGTH_UM

frequency = 1 / WAVELENGTH_UM
polar_rad = np.linspace(0, 2 * math.pi, NUM_POLAR, endpoint=False)


def pattern(dipole_component):
    cell_um = 4.0
    sxy = PML_UM + cell_um + PML_UM
    sim = mp.Simulation(
        cell_size=mp.Vector3(sxy, sxy, 0),
        resolution=RESOLUTION,
        boundary_layers=[mp.PML(PML_UM)],
        sources=[mp.Source(mp.GaussianSource(frequency, fwidth=0.2 * frequency),
                           center=mp.Vector3(), component=dipole_component)],
    )
    n2f = sim.add_near2far(
        frequency, 0, 1,
        mp.Near2FarRegion(center=mp.Vector3(0, +cell_um / 2), size=mp.Vector3(cell_um, 0), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -cell_um / 2), size=mp.Vector3(cell_um, 0), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(+cell_um / 2, 0), size=mp.Vector3(0, cell_um), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-cell_um / 2, 0), size=mp.Vector3(0, cell_um), weight=-1),
    )
    sim.run(until_after_sources=mp.stop_when_fields_decayed(50, dipole_component,
                                                            mp.Vector3(), 1e-8))
    e = np.zeros((NUM_POLAR, 3), dtype=np.complex128)
    h = np.zeros((NUM_POLAR, 3), dtype=np.complex128)
    for i, th in enumerate(polar_rad):
        ff = sim.get_farfield(n2f, mp.Vector3(FARFIELD_RADIUS * math.cos(th),
                                              FARFIELD_RADIUS * math.sin(th)))
        e[i, :] = [ff[j] for j in range(3)]
        h[i, :] = [ff[j + 3] for j in range(3)]
    sx = np.real(np.conj(e[:, 1]) * h[:, 2] - np.conj(e[:, 2]) * h[:, 1])
    sy = np.real(np.conj(e[:, 2]) * h[:, 0] - np.conj(e[:, 0]) * h[:, 2])
    return np.sqrt(sx ** 2 + sy ** 2)


out = {
    "resolution": RESOLUTION, "farfield_radius": FARFIELD_RADIUS,
    "num_polar": NUM_POLAR, "frequency": frequency,
    "flux_ez": pattern(mp.Ez).tolist(),
    "flux_ey": pattern(mp.Ey).tolist(),
}
path = os.path.join(os.path.dirname(__file__), "antenna_radiation_n2f.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", path)
