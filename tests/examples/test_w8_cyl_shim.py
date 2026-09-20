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


def test_force_complex_fields_is_2d_and_cylindrical_only():
    """2D cartesian carries a companion imaginary field set now -- it is
    what solve_cw solves for -- so force_complex_fields is accepted there
    as well as in cylindrical, where BOR fields are complex by
    construction. 1D and 3D still raise: those steppers have no companion
    set.

    This asserted the opposite until 2026-09-08. It was written when the
    guard was blanket, and the solve_cw port (e2eb7ab) superseded it; the
    failure went unnoticed for a day because the work was gated on
    `pytest tests/mp_parity` and this test lives in tests/examples.
    """
    assert mp.Simulation(resolution=10, cell_size=mp.Vector3(2, 2),
                         force_complex_fields=True).force_complex_fields
    with pytest.raises(NotImplementedError, match="force_complex_fields"):
        mp.Simulation(resolution=10, cell_size=mp.Vector3(2, 2, 2),
                      dimensions=3, force_complex_fields=True)


# --- the radial (zero z extent) path: core/stepper_cyl through the shim ------

def test_radial_cylindrical_run_selects_the_radial_stepper():
    """A cylindrical cell with no z extent routes to core/stepper_cyl.

    Found by the change-impact graph, which flagged CylGrid/CylRun/step as
    untested: the radial path was validated by running ring-cyl and
    perturbation_theory against the oracle, but nothing in the suite pinned it,
    so a regression would only have shown up in a tutorial run.
    """
    import numpy as np

    from meep_torch.mp.cyl_shim import CylRunRadial

    sim = mp.Simulation(resolution=20, cell_size=mp.Vector3(3.0, 0, 0),
                        dimensions=mp.CYLINDRICAL, m=3,
                        boundary_layers=[mp.PML(1.0)],
                        geometry=[mp.Block(center=mp.Vector3(1.5),
                                           size=mp.Vector3(1, mp.inf, mp.inf),
                                           material=mp.Medium(index=3.4))],
                        sources=[mp.Source(mp.GaussianSource(0.15, fwidth=0.1),
                                           component=mp.Ez,
                                           center=mp.Vector3(1.1))])
    h = mp.Harminv(mp.Ez, mp.Vector3(1.1), 0.15, 0.1)
    sim.run(mp.after_sources(h), until_after_sources=50)

    assert type(sim._cyl) is CylRunRadial
    assert sim._cyl.g.radial and sim._cyl.g.nz == 1

    # eps comes from the ordinary rasterizer, not a hand-built profile: the
    # ring index 3.4 must appear as 3.4^2.
    eps = np.asarray(sim.get_epsilon())
    assert eps.shape == (60,)
    assert eps.max() == pytest.approx(3.4 ** 2, rel=1e-9)
    assert eps.min() == pytest.approx(1.0, rel=1e-9)

    # the meep step-function protocol runs on this path (after_sources fired)
    assert len(h.data) > 100

    # the complementary components are refused, not faked: a radial run has no
    # Er/Ep/Hz, and returning zeros for them would be a plausible wrong answer
    for comp in ("Er", "Ep", "Hz"):
        with pytest.raises(NotImplementedError):
            sim.get_field_point(comp, mp.Vector3(1.1))


def test_radial_cylindrical_source_carries_the_derived_measure():
    """The radial point source carries 1/dr, from core.sources.source_measure.

    meep treats a zero-thickness source in r as a Dirac delta(r), so the
    amplitude picks up one factor of the resolution -- R is the only grid
    direction it is a delta in once a zero z extent makes Z a nosize direction.
    """
    from meep_torch.core.sources import source_measure

    sim = mp.Simulation(resolution=20, cell_size=mp.Vector3(3.0, 0, 0),
                        dimensions=mp.CYLINDRICAL, m=3,
                        boundary_layers=[mp.PML(1.0)],
                        sources=[mp.Source(mp.GaussianSource(0.15, fwidth=0.1),
                                           component=mp.Ez,
                                           center=mp.Vector3(1.1))])
    sim.run(until=1)
    assert sim._cyl.measure == pytest.approx(source_measure((1,), sim._cyl.g.dr, ndim=1))
    assert sim._cyl.measure == pytest.approx(1.0 / sim._cyl.g.dr)


def test_continuous_source_is_not_silenced_in_a_cylindrical_run():
    """A CW source must actually inject: _source_cutoff returns 0 for "never
    ends", and reading that as a turn-off time zeroed every CW cylindrical run.
    """
    import numpy as np

    sim = mp.Simulation(resolution=20, cell_size=mp.Vector3(3.0, 0, 0),
                        dimensions=mp.CYLINDRICAL, m=3,
                        boundary_layers=[mp.PML(1.0)],
                        sources=[mp.Source(mp.ContinuousSource(0.15),
                                           component=mp.Ez,
                                           center=mp.Vector3(1.1))])
    assert sim._source_cutoff() == 0.0          # the "never ends" sentinel
    sim.run(until=8)
    assert np.abs(np.asarray(sim.get_epsilon())).max() > 0      # built
    assert abs(complex(sim.get_field_point(mp.Ez, mp.Vector3(1.1)))) > 0.0
