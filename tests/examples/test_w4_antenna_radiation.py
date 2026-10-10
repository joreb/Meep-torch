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


def _pattern_te(component=None):
    cell = 4.0
    sim = mp.Simulation(cell_size=mp.Vector3(6, 6, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FREQ, fwidth=0.2),
                           component=component or mp.Ey,
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
    return np.sqrt(sx ** 2 + sy ** 2), th, sim


def test_ey_pattern_cos2_and_oracle():
    got, th, _ = _pattern_te()
    got = got / got.max()
    # gate (b): y dipole radiates cos^2 (nulls along +-y)
    rms_cos2 = float(np.sqrt(np.mean((got - np.cos(th) ** 2) ** 2)))
    assert rms_cos2 <= 0.03, f"normalized rms vs cos^2 {rms_cos2:.4f}"
    # gate (c), Ey row
    ref = np.array(_oracle()["flux_ey"])
    ref = ref / ref.max()
    rms = float(np.sqrt(np.mean((got - ref) ** 2)))
    assert rms <= 0.05, f"normalized rms vs oracle {rms:.4f}"


def test_ex_dipole_radiates_sin2_and_leaves_ez_empty():
    """An x dipole radiates sin^2 -- nulls along +-x, not +-y.

    THE WHOLE PATTERN USED TO BE THE WRONG ONE, to FDTD accuracy. `pol_for`
    listed `Ex` in `_EZ_COMPONENTS` as a 1D alias ("Ex used in 1D -> TM Ez"),
    which silently captured 2D as well: an `mp.Source(component=mp.Ex)` ran a
    TM simulation and injected Ez. On the antenna-radiation tutorial that came
    out as cos^2 -- matching the Ey dipole's curve to rms 0.0015 while missing
    its own by 0.7104. A gate reading only "is the pattern smooth and does it
    peak somewhere" cannot see that; this one compares against the curve the
    orientation PICKS.

    The Ez assertion is the cheap half and the sharper one: in 2D an x-directed
    electric current couples to (Ex, Ey, Hz) only, so Ez is identically zero --
    meep returns exact 0.0, and the old TM routing filled it. Nothing here
    needs pymeep: both halves are analytic.
    """
    got, th, sim = _pattern_te(mp.Ex)
    got = got / got.max()
    rms_sin2 = float(np.sqrt(np.mean((got - np.sin(th) ** 2) ** 2)))
    rms_cos2 = float(np.sqrt(np.mean((got - np.cos(th) ** 2) ** 2)))
    assert rms_sin2 <= 0.03, f"normalized rms vs sin^2 {rms_sin2:.4f}"
    # and it must not be the OTHER dipole's pattern, which is what it was
    assert rms_cos2 > 0.3, f"pattern is cos^2, i.e. a y dipole: {rms_cos2:.4f}"
    ez = np.asarray(sim.get_array(component=mp.Ez))
    assert np.abs(ez).max() == 0.0, (
        f"Ez is not empty for an x-directed current: max {np.abs(ez).max():.3e}"
        " -- the source is being routed to the TM polarization")
