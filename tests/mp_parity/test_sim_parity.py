# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Simulation-surface parity: Dft return objects, Volume regions, run() step
functions (at_every/after_sources/...), stop conditions, until_after_sources
source-cutoff semantics, and the Harminv step-function shim."""
import math

import numpy as np

import meep_torch.mp as mp

FCEN, DF = 0.5, 0.2


def _sim(sources=None, cell=8):
    if sources is None:
        sources = [mp.Source(mp.GaussianSource(frequency=FCEN, fwidth=DF),
                             component=mp.Ez, center=mp.Vector3(-2, 0),
                             size=mp.Vector3(0, 2, 0))]
    return mp.Simulation(cell_size=mp.Vector3(cell, cell, 0), resolution=10,
                         sources=sources, boundary_layers=[mp.PML(1.0)])


def _cutoff(sim):
    return sim._sources[0].src.profile().t_end


def test_dft_flux_object_roundtrip():
    sim = _sim()
    f = sim.add_flux(FCEN, DF, 5, mp.FluxRegion(center=mp.Vector3(2, 0),
                                                size=mp.Vector3(0, 4)))
    assert isinstance(f, mp.DftFlux)
    sim.run(until=30)
    assert mp.get_fluxes(f) == sim.get_fluxes(f.name)       # object == legacy string
    assert mp.get_flux_freqs(f) == sim.get_flux_freqs(f.name)
    np.testing.assert_allclose(mp.get_flux_freqs(f), np.linspace(0.4, 0.6, 5))


def test_dft_fields_and_module_freqs():
    sim = _sim()
    d = sim.add_dft_fields([mp.Ez], FCEN, DF, 3)
    n2f = sim.add_near2far(FCEN, 0, 1,
                           mp.Near2FarRegion(center=mp.Vector3(2, 0),
                                             size=mp.Vector3(0, 4)))
    ldos = sim.dft_ldos(FCEN, DF, 3)
    assert isinstance(d, mp.DftFields) and isinstance(n2f, mp.DftNear2Far)
    sim.run(until=30)
    a_obj = sim.get_dft_array(d, mp.Ez, 0)
    a_str = sim.get_dft_array(d.name, mp.Ez, 0)
    np.testing.assert_array_equal(a_obj, a_str)
    np.testing.assert_allclose(mp.get_near2far_freqs(n2f), [FCEN])
    np.testing.assert_allclose(mp.get_ldos_freqs(ldos), np.linspace(0.4, 0.6, 3))
    ff = sim.get_farfield(n2f, mp.Vector3(20, 0))
    assert np.shape(ff) == (6,)


def test_volume_regions():
    v = mp.Volume(center=mp.Vector3(2, 0), size=mp.Vector3(0, 4))
    fr = mp.FluxRegion(volume=v)
    assert fr.center == v.center and fr.size == v.size
    nr = mp.Near2FarRegion(volume=v)
    assert nr.center == v.center and nr.size == v.size
    # volume region wires the same monitor as center/size
    sim = _sim()
    f1 = sim.add_flux(FCEN, DF, 3, mp.FluxRegion(volume=v))
    f2 = sim.add_flux(FCEN, DF, 3, mp.FluxRegion(center=v.center, size=v.size))
    sim.run(until=30)
    assert mp.get_fluxes(f1) == mp.get_fluxes(f2)


def test_at_every_and_bare_step_function():
    sim = _sim()
    times, calls = [], []
    sim.run(mp.at_every(1.0, lambda s: times.append(s.round_time())),
            lambda s: calls.append(s.round_time()), until=10)
    assert len(times) == 11, times                # t = 0, 1, ..., 10
    np.testing.assert_allclose(times, np.arange(11.0))
    assert len(calls) == round(10 / sim.grid.dt)  # bare callable: every step
    assert abs(sim.round_time() - 10.0) < 1e-9


def test_begin_end_at_time_before_time():
    sim = _sim()
    order = []
    sim.run(mp.at_beginning(lambda s: order.append(("begin", s.round_time()))),
            mp.at_end(lambda s: order.append(("end", s.round_time()))),
            mp.at_time(5.0, lambda s: order.append(("at", s.round_time()))),
            until=10)
    assert order[0] == ("begin", 0.0)
    assert order[-1] == ("end", 10.0)
    assert any(k == "at" and abs(t - 5.0) < 1e-9 for k, t in order)
    sim2 = _sim()
    early = []
    sim2.run(mp.before_time(1.0, lambda s: early.append(s.round_time())), until=5)
    assert early and max(early) <= 1.0 + 1e-9
    # with_prefix wraps into a meep f(sim, todo) step function
    fn = lambda s: None
    wrapped = mp.with_prefix("out-", fn)
    assert callable(wrapped) and wrapped is not fn


def test_after_during_sources_timing():
    sim = _sim()
    cut = _cutoff(sim)
    after, during = [], []
    sim.run(mp.after_sources(lambda s: after.append(s.round_time())),
            mp.during_sources(lambda s: during.append(s.round_time())),
            until=cut + 5)
    assert after and after[0] >= cut - 1e-9
    assert after[0] - cut < 1.0                    # starts right at the cutoff
    assert during and max(during) <= cut + 1e-9


def test_until_after_sources_adds_cutoff():
    sim = _sim()
    cut = _cutoff(sim)
    sim.run(until_after_sources=10)
    assert abs(sim.round_time() - (cut + 10)) <= sim.grid.dt + 1e-9


def test_stop_when_fields_decayed():
    sim = _sim()
    cut = _cutoff(sim)
    pt = mp.Vector3(2, 0)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(5, mp.Ez, pt, 1e-6))
    assert sim.round_time() >= cut                 # ran past the source cutoff
    assert abs(sim.get_field_point(mp.Ez, pt)) < 1e-3   # and the pulse left


def test_stop_after_walltime():
    sim = _sim()
    sim.run(until=mp.stop_after_walltime(-1.0))    # already expired -> stops at once
    assert sim.round_time() <= 10 * sim.grid.dt


def test_harminv_driven_point_source():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(frequency=FCEN, fwidth=DF),
                                  component=mp.Ez, center=mp.Vector3(0, 0))],
               cell=6)
    h = mp.Harminv(mp.Ez, mp.Vector3(0.5, 0.5), FCEN, 2 * DF)
    sim.run(h, until_after_sources=20)
    assert h.modes, "Harminv found no modes"
    m = h.modes[0]                                 # strongest amplitude
    assert abs(m.freq - FCEN) < 0.05, f"freq={m.freq}"
    for attr in ("freq", "decay", "Q", "amp", "err"):   # meep Harminv mode fields
        assert hasattr(m, attr)


def test_add_flux_multiple_regions():
    sim = _sim()
    r1 = mp.FluxRegion(center=mp.Vector3(2, 1), size=mp.Vector3(0, 2))
    r2 = mp.FluxRegion(center=mp.Vector3(2, -1), size=mp.Vector3(0, 2))
    f12 = sim.add_flux(FCEN, DF, 3, r1, r2)          # meep variadic form: summed
    f1 = sim.add_flux(FCEN, DF, 3, r1)
    f2 = sim.add_flux(FCEN, DF, 3, r2)
    sim.run(until=30)
    np.testing.assert_allclose(mp.get_fluxes(f12),
                               np.add(mp.get_fluxes(f1), mp.get_fluxes(f2)))
    assert mp.get_flux_freqs(f12) == mp.get_flux_freqs(f1)
