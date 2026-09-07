# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""force_complex_fields + solve_cw (meep src/cw_fields.cpp).

The discriminating gate is the physics one: the frequency-domain solve must
reproduce the SAME cell's time-domain CW steady state. A self-consistency
check (error shrinking with tolerance, which is what the solve-cw tutorial
prints) only says the Krylov solver converged to something -- it would pass
just as happily on a wrong operator.
"""
import numpy as np
import pytest

import meep_torch.mp as mp

RES = 10
FCEN = 0.15
SXY = 8.0
DPML = 1.0


def _sim(force_complex=True, symmetries=None, src_x=-2.0):
    return mp.Simulation(
        resolution=RES, cell_size=mp.Vector3(SXY, SXY),
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.ContinuousSource(FCEN), component=mp.Ez,
                           center=mp.Vector3(src_x, 0))],
        symmetries=symmetries, boundary_layers=[mp.PML(DPML)],
        force_complex_fields=force_complex)


def test_solve_cw_matches_time_domain_steady_state():
    sim = _sim()
    sim.init_sim()
    assert sim.solve_cw(1e-10, 5000, 2), "BiCGSTAB(2) did not converge"
    ez_cw = np.asarray(sim.get_array(component=mp.Ez))
    assert ez_cw.dtype == np.complex128

    # same cell, driven in time until the CW field settles; two snapshots a
    # quarter period apart give the complex amplitude (a + i c = E e^{-i w t})
    td = _sim(force_complex=False)
    td.run(until=400)
    a = np.asarray(td.get_array(component=mp.Ez))
    td.run(until=1 / FCEN / 4)
    c = np.asarray(td.get_array(component=mp.Ez))
    ez_td = a + 1j * c

    # solve_cw drives with the source's amplitude at ITS current time, which
    # for a fresh sim sits at the bottom of the turn-on ramp -- meep does the
    # same (cw_fields.cpp reads calc_sources at time()). That is a constant
    # complex scale on a linear solve, so divide it out before comparing.
    i = np.unravel_index(np.abs(ez_td).argmax(), ez_td.shape)
    k = ez_td[i] / ez_cw[i]
    rel = np.linalg.norm(ez_cw * k - ez_td) / np.linalg.norm(ez_td)
    print(f"solve_cw vs time-domain steady state: rel-L2 {rel:.3e}, "
          f"phase {np.angle(k):+.4f} rad")
    assert rel < 5e-2, f"solve_cw field disagrees with the CW run: {rel:.3e}"
    # ContinuousSource.camplitude is the analytic continuation of this
    # engine's own sin() carrier, so the two agree in phase, not just in
    # magnitude. A quarter turn here means the convention drifted.
    assert abs(np.angle(k)) < 0.1, f"CW phase off by {np.angle(k):+.3f} rad"


def test_solve_cw_under_mirrors_respects_the_imposed_parity():
    """A folded solve, de-mirrored, is EXACTLY odd/even by construction --
    separately in the real and imaginary parts. This is what breaks first if
    the complex assembly de-mirrors only one part, or with one parity."""
    sim = _sim(symmetries=[mp.Mirror(mp.X, phase=-1), mp.Mirror(mp.Y, phase=+1)],
               src_x=0.0)
    sim.init_sim()
    assert sim.grid.nx == RES * 4 and sim.grid.ny == RES * 4, "cell not quartered"
    assert sim.solve_cw(1e-10, 5000, 2)
    ez = np.asarray(sim.get_array(component=mp.Ez))
    s = np.abs(ez).max()
    assert s > 0
    for part in (ez.real, ez.imag):
        np.testing.assert_allclose(part, -part[::-1, :], atol=1e-12 * s)
        np.testing.assert_allclose(part, part[:, ::-1], atol=1e-12 * s)


def test_complex_fields_run_reproduces_the_real_run():
    """force_complex_fields must not change a plain run: engine currents are
    real, so the companion set stays at zero and the real half is untouched."""
    a = _sim(force_complex=False)
    a.run(until=40)
    b = _sim(force_complex=True)
    b.run(until=40)
    ea = np.asarray(a.get_array(component=mp.Ez))
    eb = np.asarray(b.get_array(component=mp.Ez))
    assert eb.dtype == np.complex128
    assert np.abs(eb.imag).max() == 0.0
    np.testing.assert_allclose(eb.real, ea, rtol=0, atol=0)


def test_solve_cw_guards():
    real = _sim(force_complex=False)
    real.init_sim()
    with pytest.raises(RuntimeError, match="force_complex_fields"):
        real.solve_cw()
    with pytest.raises(NotImplementedError):
        mp.Simulation(resolution=RES, cell_size=mp.Vector3(4, 4, 4),
                      dimensions=3, force_complex_fields=True,
                      sources=[mp.Source(mp.ContinuousSource(FCEN),
                                         component=mp.Ez, center=mp.Vector3())])
