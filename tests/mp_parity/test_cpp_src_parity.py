# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""C++-binding-surface parity (sources/susceptibilities/DFT group):
src_time adapters agree with the core engine pulses, dft views round-trip
through a live shim run, and every class matches its meep_api.json spec."""
import inspect
import json
import math
import pathlib

import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.core.sources import GaussianPulse

API = json.load(open(pathlib.Path(__file__).parents[2] / "scripts" / "meep_api.json"))

MINE = ["src_time", "gaussian_src_time", "continuous_src_time",
        "custom_src_time", "custom_py_src_time", "gaussianbeam",
        "diffractedplanewave", "susceptibility", "lorentzian_susceptibility",
        "noisy_lorentzian_susceptibility", "gyrotropic_susceptibility",
        "multilevel_susceptibility", "dft_fields", "dft_flux", "dft_force",
        "dft_energy", "dft_near2far", "sourcedata", "near_src_data",
        "eigenmode_data", "pol", "polarization_state"]


# ------------------------------------------------------------- structural ---

@pytest.mark.parametrize("name", MINE)
def test_spec_surface(name):
    """Class exists, direct bases cover the spec, all spec methods present,
    and named init params accepted (mirrors check_meep_parity grading)."""
    spec = API["classes"][name]
    cls = getattr(mp, name)
    assert inspect.isclass(cls)
    have_b = {b.__name__ for b in cls.__bases__}
    assert set(spec["bases"]) <= have_b
    have_m = {m for m in dir(cls) if not m.startswith("_")}
    assert set(spec["methods"]) <= have_m, set(spec["methods"]) - have_m
    want_p = set(spec["init_params"] or []) - {"args", "kwargs"}
    have_p = {p.name for p in
              inspect.signature(cls.__init__).parameters.values()}
    assert want_p <= have_p


# ------------------------------------------------------- src_time adapters ---

def test_gaussian_src_time_wraps_engine_pulse():
    g = mp.gaussian_src_time(1.0, 0.2)          # width 5, s=5
    pulse = GaussianPulse(1.0, 0.2)
    assert g.frequency() == 1.0
    assert g.get_fwidth() == pytest.approx(0.2)
    assert g.last_time() == pytest.approx(pulse.t_end)
    assert g.last_time_max(100.0) == 100.0
    for t in (5.0, 20.0, 25.0, 30.0):           # inside window: engine profile
        assert g.dipole(t) == pytest.approx(pulse.amplitude(t))
    assert g.dipole(-1.0) == 0.0 and g.dipole(g.last_time() + 1e-6) == 0.0
    assert g.dipole(25.0) == pytest.approx(1.0)  # cos peak at t0 = 25
    # current is the finite-difference derivative of dipole
    dt = 1e-4
    assert g.current(10.0, dt) == pytest.approx(
        (g.dipole(10.0 + dt) - g.dipole(10.0)) / dt)
    # update caches the no-arg dipole()
    g.update(10.0, dt)
    assert g.dipole() == g.dipole(10.0)


def test_gaussian_src_time_window_form_and_equality():
    g4 = mp.gaussian_src_time(1.0, 5.0, 0.0, 50.0)   # (f, width, start, end)
    assert g4.last_time() == pytest.approx(50.0)
    assert g4.dipole(25.0) == pytest.approx(1.0)     # peak at mid-window
    g = mp.gaussian_src_time(1.0, 0.2)
    assert g.is_equal(g4) and g4.is_equal(g.clone())  # same (f, w, peak, cut)
    g.set_frequency(2.0)
    assert not g.is_equal(g4) and g.frequency() == 2.0
    g.set_fwidth(0.5)
    assert g.get_fwidth() == pytest.approx(0.5)
    # analytic FT peaks at the carrier and is down by exp(-1/2) one sigma off
    f0 = mp.gaussian_src_time(1.0, 0.2)
    peak = abs(f0.fourier_transform(1.0))
    off = abs(f0.fourier_transform(1.0 + 1.0 / (2 * math.pi * 5.0)))
    assert off / peak == pytest.approx(math.exp(-0.5), rel=1e-6)


def test_continuous_src_time():
    c = mp.continuous_src_time(0.5, width=2.0, start_time=1.0, end_time=40.0)
    assert c.frequency() == 0.5 and c.get_fwidth() == 0.0
    assert c.last_time() == 40.0
    assert c.dipole(0.5) == 0.0 and c.dipole(41.0) == 0.0    # outside window
    assert c.dipole(1.0) == pytest.approx(0.0)               # ramp starts at 0
    t = 20.0    # fully ramped: pure sin carrier, engine convention
    assert c.dipole(t) == pytest.approx(math.sin(2 * math.pi * 0.5 * (t - 1.0)))
    assert c.is_equal(c.clone())


def test_custom_src_time_and_py_variant():
    for cls in (mp.custom_src_time, mp.custom_py_src_time):
        c = cls(lambda t: 2.0 * t, start_time=0.0, end_time=10.0,
                center_frequency=0.7, fwidth=0.3)
        assert c.dipole(1.5) == pytest.approx(3.0)
        assert c.dipole(-1.0) == 0.0 and c.dipole(11.0) == 0.0
        assert c.frequency() == 0.7 and c.get_fwidth() == pytest.approx(0.3)
        assert c.last_time() == 10.0
        c.set_fwidth(0.6)
        assert c.get_fwidth() == pytest.approx(0.6)
    assert isinstance(mp.custom_py_src_time(lambda t: t), mp.custom_src_time)


def test_src_time_raises():
    g = mp.gaussian_src_time(1.0, 0.2)
    with pytest.raises(NotImplementedError):
        g.add_to()
    with pytest.raises(NotImplementedError):
        mp.src_time().last_time()


# ---------------------------------------------------------- susceptibility ---

def test_lorentzian_chi1():
    s = mp.lorentzian_susceptibility(omega_0=1.0, gamma=0.1)
    f = 0.5
    want = 1.0 / (1.0 - f * f - 1j * f * 0.1)
    assert s.chi1(f, sigma=1.0) == pytest.approx(want)
    assert s.chi1(f, sigma=2.5) == pytest.approx(2.5 * want)
    d = mp.lorentzian_susceptibility(1.0, 0.1, no_omega_0_denominator=True)
    assert d.chi1(f) == pytest.approx(1.0 / (-f * f - 1j * f * 0.1))
    assert mp.susceptibility().chi1(f) == 0.0j


def test_susceptibility_surface():
    s = mp.noisy_lorentzian_susceptibility(0.1, 1.0, 0.2)
    assert isinstance(s, mp.lorentzian_susceptibility)
    assert s.dump_params()["noise_amp"] == 0.1
    assert s.get_num_params() == 4
    s.set_id(7)
    assert s.get_id() == 7 and s.clone().get_id() == 7
    assert s.has_nonlinearities() is False and s.needs_W_prev() is False
    with pytest.raises(NotImplementedError):
        s.update_P()
    with pytest.raises(NotImplementedError):
        s.new_internal_data()
    assert mp.gyrotropic_susceptibility(omega_0=1.0).get_num_params() == 5
    assert mp.multilevel_susceptibility(1, 2).dump_params()["args"] == (1, 2)


# --------------------------------------------------------------- DFT views ---

FCEN, DF = 0.5, 0.2


@pytest.fixture(scope="module")
def flux_run():
    sim = mp.Simulation(
        cell_size=mp.Vector3(8, 8, 0), resolution=10,
        sources=[mp.Source(mp.GaussianSource(frequency=FCEN, fwidth=DF),
                           component=mp.Ez, center=mp.Vector3(-2, 0),
                           size=mp.Vector3(0, 2, 0))],
        boundary_layers=[mp.PML(1.0)])
    f = sim.add_flux(FCEN, DF, 5, mp.FluxRegion(center=mp.Vector3(2, 0),
                                                size=mp.Vector3(0, 4)))
    n2f = sim.add_near2far(FCEN, 0, 1,
                           mp.Near2FarRegion(center=mp.Vector3(2, 0),
                                             size=mp.Vector3(0, 4)))
    sim.run(until=30)
    return sim, f, n2f


def test_dft_flux_view_roundtrip(flux_run):
    sim, f, _ = flux_run
    v = mp.dft_flux(f)
    assert v.flux() == sim.get_fluxes(f)
    with pytest.raises(NotImplementedError):
        v.complexflux()
    v.invalidate_eigenmode_cache()      # no-op, must not raise


def test_dft_near2far_view_roundtrip(flux_run):
    sim, _, n2f = flux_run
    v = mp.dft_near2far(n2f)
    pt = mp.Vector3(20, 0)
    np.testing.assert_array_equal(v.farfield(pt), sim.get_farfield(n2f, pt))


def test_dft_views_standalone_raise():
    for cls in (mp.dft_flux, mp.dft_fields, mp.dft_near2far):
        view = cls()
        with pytest.raises((RuntimeError, NotImplementedError)):
            (view.flux if cls is mp.dft_flux else
             view.fourier_sourcedata if cls is mp.dft_fields else
             lambda: view.farfield(mp.Vector3()))()
    with pytest.raises(NotImplementedError):
        mp.dft_flux().save_hdf5("f")
    # dft_energy/dft_force are live views now (Simulation.add_energy/add_force);
    # standalone they raise the no-backing RuntimeError like the others
    with pytest.raises(RuntimeError):
        mp.dft_energy().total()
    with pytest.raises(RuntimeError):
        mp.dft_force().force()


# ------------------------------------------------------------ data carriers ---

def test_diffractedplanewave_and_gaussianbeam():
    dp = mp.diffractedplanewave([1, 0, 0], mp.Vector3(0, 0, 1), 1.0, 0.0)
    assert dp.get_g() == [1, 0, 0] and dp.get_s() == 1.0 and dp.get_p() == 0.0
    assert dp.get_axis() == mp.Vector3(0, 0, 1)
    b = mp.gaussianbeam(x0=mp.Vector3(), kdir=mp.Vector3(1, 0, 0), w0=1.5,
                        freq=1.0, eps=1.0, mu=1.0, E0=[0, 0, 1])
    assert b.get_E0() == [0, 0, 1]
    with pytest.raises(NotImplementedError):
        b.get_fields(None, None)


def test_near_src_data_vector_api():
    v = mp.near_src_data()
    assert v.empty() and v.size() == 0
    v.push_back("a")
    v.push_back("b")
    assert v.size() == 2 and v.front() == "a" and v.back() == "b"
    v.resize(4, None)
    assert v.size() == 4
    v.erase(2)
    assert list(v) == ["a", "b", None]
    assert v.pop_back() is None and v.size() == 2
    w = mp.near_src_data()
    w.assign(3, "x")
    v.swap(w)
    assert list(v) == ["x"] * 3 and list(w) == ["a", "b"]
    assert list(v.begin()) == ["x"] * 3 and list(v.rbegin()) == ["x"] * 3
