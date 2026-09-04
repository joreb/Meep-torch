# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp/cyl_shim.py: the meep Simulation surface driving core/stepper_bor.

Driven through a plain ``mp.Simulation`` -- the dispatch lives in
simulation.py, so there is no subclass here to keep in step with it.

Physics gate = the upstream ``dipole_in_vacuum_cyl_on_axis.py z`` case:
an on-axis Ez point dipole in vacuum (m = 0) must radiate sin^2(theta),
computed the way the example does -- radial Poynting flux built from the
six components ``get_farfield`` returns.
"""
import math

import numpy as np
import pytest

import meep_torch.mp as mp


RES = 25
FCEN = 1.0
PML = 1.0
SR, SZ = 2.0, 4.0
NPTS = 30
POLAR = np.linspace(0.0, 0.5 * math.pi, NPTS)
R_FAR = 1e6


def _run(component, m, amplitude_pair=False):
    src_r = 1.5 / RES if amplitude_pair else 0.0
    sources = [mp.Source(src=mp.GaussianSource(FCEN, fwidth=0.1 * FCEN),
                         component=component,
                         center=mp.Vector3(src_r, 0, 0))]
    if amplitude_pair:
        sources.append(mp.Source(
            src=mp.GaussianSource(FCEN, fwidth=0.1 * FCEN), component=mp.Ep,
            center=mp.Vector3(src_r, 0, 0), amplitude=1j if m == 1 else -1j))
    sim = mp.Simulation(resolution=RES,
                        cell_size=mp.Vector3(SR + PML, 0, SZ + 2 * PML),
                        dimensions=mp.CYLINDRICAL, m=m,
                        boundary_layers=[mp.PML(thickness=PML)],
                        sources=sources, force_complex_fields=True)
    n2f = sim.add_near2far(
        FCEN, 0, 1,
        mp.FluxRegion(center=mp.Vector3(0.5 * SR, 0, 0.5 * SZ),
                      size=mp.Vector3(SR, 0, 0)),
        mp.FluxRegion(center=mp.Vector3(SR, 0, 0), size=mp.Vector3(0, 0, SZ)),
        mp.FluxRegion(center=mp.Vector3(0.5 * SR, 0, -0.5 * SZ),
                      size=mp.Vector3(SR, 0, 0)))
    sim.run(until_after_sources=40.0)
    e = np.zeros((NPTS, 3), dtype=complex)
    h = np.zeros((NPTS, 3), dtype=complex)
    for n, th in enumerate(POLAR):
        ff = sim.get_farfield(n2f, mp.Vector3(R_FAR * math.sin(th), 0,
                                              R_FAR * math.cos(th)), 1e-6)
        e[n], h[n] = ff[:3], ff[3:]
    return e, h


def _pattern(e, h):
    """Radial Poynting flux, exactly the upstream example's formula."""
    fx = np.real(e[:, 1] * np.conj(h[:, 2]) - e[:, 2] * np.conj(h[:, 1]))
    fz = np.real(e[:, 0] * np.conj(h[:, 1]) - e[:, 1] * np.conj(h[:, 0]))
    p = np.hypot(fx, fz)
    return p / p.max()


def test_on_axis_ez_dipole_sin2():
    """m = 0, on-axis Ez dipole in vacuum -> sin^2(theta)."""
    e, h = _run(mp.Ez, 0)
    got = _pattern(e, h)
    want = np.sin(POLAR) ** 2
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    # measured 0.0054 at resolution 25
    assert err <= 0.03, f"relative error vs sin^2 = {err:.4f}\n{got}"


def test_on_axis_er_dipole_cos2():
    """|m| = 1 circular pair, on-axis Er dipole -> cos^2(theta). Exercises
    the axis treatment of the m != 0 BOR update through the shim."""
    ep, hp = _run(mp.Er, +1, amplitude_pair=True)
    em, hm = _run(mp.Er, -1, amplitude_pair=True)
    got = _pattern(0.5 * (ep + em), 0.5 * (hp + hm))
    want = np.cos(POLAR) ** 2
    err = np.linalg.norm(got - want) / np.linalg.norm(want)
    # measured 0.0426 at resolution 25 (the on-axis Er source sits
    # 1.5/res off the axis, meep NanoComp/meep#2704)
    assert err <= 0.08, f"relative error vs cos^2 = {err:.4f}\n{got}"


def test_named_limitations():
    """Out-of-scope combinations raise with a named reason, not silently."""
    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(2, 0, 2),
                        dimensions=mp.CYLINDRICAL, m=1,
                        geometry=[mp.Cylinder(radius=0.2,
                                              material=mp.Medium(index=2.0))],
                        sources=[mp.Source(src=mp.GaussianSource(1.0, fwidth=0.2),
                                           component=mp.Ez,
                                           center=mp.Vector3(0.5, 0, 0))])
    with pytest.raises(NotImplementedError, match="mp.Block"):
        sim.run(until=1.0)


def test_m_without_cylindrical_names_the_fix():
    """`m` on a cartesian cell used to raise "the cylindrical engine is not
    ported" -- false since W7. It now names the argument that was missing."""
    with pytest.raises(NotImplementedError, match="dimensions=mp.CYLINDRICAL"):
        mp.Simulation(resolution=10, cell_size=mp.Vector3(2, 2), m=1)


def test_force_complex_fields_still_raises_outside_cylindrical():
    """The cylindrical exemption must not switch the guard off everywhere:
    BOR fields are complex by construction, cartesian ones are not."""
    with pytest.raises(NotImplementedError, match="force_complex_fields"):
        mp.Simulation(resolution=10, cell_size=mp.Vector3(2, 2),
                      force_complex_fields=True)
