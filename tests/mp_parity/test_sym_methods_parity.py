# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Symmetry family + Simulation method parity.

Physics gate: a TM straight waveguide (test_y_mirror physics, run through the
shim) gives the same |S21| = sqrt(P_out/P_in) on the full cell and on the
Mirror(Y, phase=+1) half cell (engine y_mirror=+1), within 1%.
"""
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import meep_torch.mp as mp

RES = 20
FCEN = 1 / 1.55
N_CO, N_CL = 1.4880, 1.4577


def _wg_sim(symmetries=None):
    return mp.Simulation(
        cell_size=mp.Vector3(15, 6),
        resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        default_material=mp.Medium(epsilon=N_CL ** 2),
        geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, 2.0),
                           material=mp.Medium(epsilon=N_CO ** 2))],
        sources=[mp.Source(mp.GaussianSource(frequency=FCEN, fwidth=0.2 * FCEN),
                           component=mp.Ez, center=mp.Vector3(-5.5, 0),
                           size=mp.Vector3(0, 2.0))],
        symmetries=symmetries,
    )


def _s21(symmetries=None):
    sim = _wg_sim(symmetries)
    f_in = sim.add_flux(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(-5.0, 0),
                                                  size=mp.Vector3(0, 5.6)))
    f_out = sim.add_flux(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(5.5, 0),
                                                   size=mp.Vector3(0, 5.6)))
    sim.run(until=60)
    p_in = mp.get_fluxes(f_in)[0]
    p_out = mp.get_fluxes(f_out)[0]
    return math.sqrt(p_out / p_in), p_in, sim


# --- Symmetry class family (meep signatures) ---

def test_symmetry_family():
    s = mp.Symmetry(direction=mp.Y, phase=-1)
    assert s.direction == mp.Y and s.phase == complex(-1)
    assert mp.Symmetry(direction=mp.X).phase == 1.0 + 0j
    for cls in (mp.Mirror, mp.Rotate2, mp.Rotate4, mp.Identity):
        assert issubclass(cls, mp.Symmetry)
        assert cls(direction=mp.Z, phase=1j).phase == 1j
    m = mp.Mirror(mp.Y)          # positional direction, meep style
    assert m.direction == mp.Y and m.phase == 1.0 + 0j
    assert (mp.X, mp.Y, mp.Z) == (0, 1, 2)


def test_unsupported_symmetries_raise_at_simulation():
    for sym in (mp.Rotate4(mp.Z), mp.Mirror(mp.Z), mp.Rotate2(mp.X),
                mp.Mirror(mp.Y, phase=1j), mp.Rotate2(mp.Z, phase=1j)):
        with pytest.raises(NotImplementedError):
            _wg_sim([sym])
    # a C2 fold does not compose with a mirror cut in the half-domain engine
    with pytest.raises(NotImplementedError):
        _wg_sim([mp.Rotate2(mp.Z), mp.Mirror(mp.Y)])
    _wg_sim([mp.Identity()])     # no-op, accepted
    # Mirror(X/Y, phase=+-1) are engine-supported now (test_symmetry_parity)
    _wg_sim([mp.Mirror(mp.X)])
    _wg_sim([mp.Mirror(mp.Y, phase=-1)])
    _wg_sim([mp.Rotate2(mp.Z)])              # C2 halves in y (test_symmetry_parity)
    _wg_sim([mp.Rotate2(mp.Z, phase=-1)])


# --- Mirror(Y) half-domain physics gate ---

def test_mirror_y_half_domain_matches_full():
    s_full, pin_full, sim_full = _s21()
    s_half, pin_half, sim_half = _s21([mp.Mirror(mp.Y)])
    rel = abs(s_half - s_full) / s_full
    assert rel < 0.01, f"|S21| full={s_full:.5f} half={s_half:.5f} rel={rel:.4f}"
    # mirror-plane-crossing flux is doubled back to the full-cell value
    assert abs(pin_half - pin_full) / abs(pin_full) < 0.05
    # de-mirrored outputs are full-cell shaped and even in y
    eps = sim_half.get_epsilon()
    assert eps.shape == (300, 120)
    np.testing.assert_allclose(eps, eps[:, ::-1])
    np.testing.assert_allclose(eps, sim_full.get_epsilon(), rtol=0, atol=1e-12)
    ez = sim_half.get_array(component=mp.Ez)
    assert ez.shape == (300, 120)
    np.testing.assert_allclose(ez, ez[:, ::-1])


def test_mirror_y_blocks_eigenmode_path():
    sim = mp.Simulation(
        cell_size=mp.Vector3(15, 6), resolution=RES,
        boundary_layers=[mp.PML(1.0)],
        default_material=mp.Medium(epsilon=N_CL ** 2),
        sources=[mp.EigenModeSource(
            mp.GaussianSource(frequency=FCEN, fwidth=0.2 * FCEN),
            center=mp.Vector3(-5.5, 0), size=mp.Vector3(0, 2.0),
            direction=1, eig_band=1)],
        symmetries=[mp.Mirror(mp.Y)],
    )
    with pytest.raises(NotImplementedError):
        sim.run(until=10)


# --- Simulation methods ---

def test_get_epsilon_roundtrip():
    sim = _wg_sim()
    eps = sim.get_epsilon()
    assert eps.shape == (300, 120)
    np.testing.assert_allclose(eps, sim.get_array(component=mp.Dielectric))
    assert abs(sim.get_epsilon_point(mp.Vector3(0, 0)) - N_CO ** 2) < 1e-9
    assert abs(sim.get_epsilon_point(mp.Vector3(0, 2.5)) - N_CL ** 2) < 1e-9


def test_get_array_metadata():
    sim = _wg_sim()
    x, y, z, w = sim.get_array_metadata()
    assert len(x) == 300 and len(y) == 120 and len(z) == 1
    assert w.shape == (300, 120)
    assert abs(np.sum(w) - 15 * 6) < 1e-9          # cubature integrates the cell
    assert abs(x[0] - (-7.5 + 0.5 / RES)) < 1e-12  # meep-centered coordinates
    x2, y2, z2, w2 = sim.get_array_metadata(center=mp.Vector3(0, 0),
                                            size=mp.Vector3(2, 1))
    assert len(x2) == 2 * RES and len(y2) == RES and w2.shape == (40, 20)
    p, wp = sim.get_array_metadata(vol=mp.Volume(center=mp.Vector3(),
                                                 size=mp.Vector3(2, 1)),
                                   return_pw=True)
    assert len(p) == 40 * 20 and wp.shape == (40, 20)


def test_reset_and_restart():
    sim = _wg_sim()
    sim.add_flux(FCEN, 0, 1, mp.FluxRegion(center=mp.Vector3(5.5, 0),
                                           size=mp.Vector3(0, 5.6)))
    sim.run(until=5)
    assert sim.round_time() > 0 and sim._state is not None
    sim.restart_fields()
    assert sim.round_time() == 0 and float(abs(sim._state.Ez).max()) == 0.0
    assert len(sim._monitors) == 1                 # dft monitors survive restart
    sim.reset_meep()
    assert sim._state is None and sim._monitors == [] and sim._last_acc == {}
    sim.run(until=1)                               # usable again after reset
    assert sim.round_time() > 0


def test_trivial_queries_and_plot2d():
    sim = _wg_sim()
    assert sim.using_real_fields() is True
    assert sim.has_mu() is False
    mu_sim = mp.Simulation(cell_size=mp.Vector3(4, 4), resolution=10,
                           default_material=mp.Medium(epsilon=2.0, mu=1.5),
                           sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                              component=mp.Ez,
                                              center=mp.Vector3())])
    assert mu_sim.has_mu() is True
    assert abs(mu_sim.get_mu_point(mp.Vector3()) - 1.5) < 1e-9
    assert mu_sim.get_mu().shape == (40, 40)
    sim.init_sim()
    assert sim._state is not None and sim.meep_time() == 0.0
    ax = sim.plot2D()
    assert ax is not None
    matplotlib.pyplot.close("all")
