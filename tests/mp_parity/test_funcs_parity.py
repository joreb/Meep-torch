# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Module-level function parity: interpolate semantics, single-process MPI
semantics, enum helpers, and one output_* file round-trip."""
import math

import numpy as np
import pytest

import meep_torch.mp as mp

FCEN, DF = 0.15, 0.1


def _sim():
    sources = [mp.Source(mp.GaussianSource(frequency=FCEN, fwidth=DF),
                         component=mp.Ez, center=mp.Vector3(-2, 0),
                         size=mp.Vector3(0, 2, 0))]
    return mp.Simulation(cell_size=mp.Vector3(6, 6, 0), resolution=10,
                         sources=sources, boundary_layers=[mp.PML(1.0)])


# --- interpolate (meep docs: n new evenly spaced values between each pair) ---
def test_interpolate_numbers():
    assert mp.interpolate(2, [1.0, 4.0]) == [1.0, 2.0, 3.0, 4.0]
    # multi-segment list (the old 2-point-only version rejected this)
    assert mp.interpolate(1, [0.0, 1.0, 2.0]) == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_interpolate_vector3():
    ks = mp.interpolate(1, [mp.Vector3(0, 0, 0), mp.Vector3(1, 2, 4)])
    assert len(ks) == 3
    assert ks[1].close(mp.Vector3(0.5, 1, 2))
    assert ks[-1].close(mp.Vector3(1, 2, 4))


# --- single-process parallel semantics (genuine, not MPI fakes) ---
def test_single_process_mpi_semantics():
    assert mp.am_master() is True
    assert mp.am_really_master() is True
    assert mp.my_rank() == 0 and mp.my_global_rank() == 0
    assert mp.count_processors() == 1
    assert mp.with_mpi() is False
    assert mp.divide_parallel_processes(4) == 0    # sole process -> group 0
    assert mp.get_num_groups() == 1
    assert mp.merge_subgroup_data([1, 2]) == [1, 2]
    assert mp.sum_to_all(3.5) == 3.5
    assert mp.max_to_all(7) == 7
    mp.all_wait()                                  # barrier: no-op, no error


# --- enum helpers (values per meep src/vec.cpp) ---
def test_component_helpers():
    assert mp.component_name(mp.Ez) == "ez"
    assert mp.component_name("Dielectric") == "eps"
    assert mp.direction_name(mp.Y) == "y"
    assert mp.dimension_name(mp.D2) == "2D"
    assert mp.is_electric("Ex") and not mp.is_magnetic("Ex")
    assert mp.is_tm("Ez") and mp.is_tm("Hx") and not mp.is_tm("Hz")
    assert mp.direction_component("Ex", mp.Z) == "Ez"
    assert mp.direction_component("Hz", mp.X) == "Hx"
    assert mp.component_index("Hz") == 2 and mp.component_index("Dielectric") == -1
    assert mp.component_direction("Ey") == mp.Y


def test_linear_interpolate():
    data = [2.0] * 27                              # constant 3x3x3 grid
    assert mp.linear_interpolate(0.3, 0.7, 0.5, data, 3, 3, 3, 1) == \
        pytest.approx(2.0)


# --- output family: real file round-trip (h5 via h5py, else npz fallback) ---
def _load(path):
    if path.endswith(".h5"):
        h5py = pytest.importorskip("h5py")
        with h5py.File(path, "r") as f:
            return {k: np.asarray(f[k]) for k in f}
    return dict(np.load(path))


def test_output_epsilon_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _sim()
    sim.run(until=1)
    path = mp.output_epsilon(sim)
    data = _load(path)
    np.testing.assert_allclose(data["eps"], sim.get_epsilon())


def test_output_efield_z_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _sim()
    sim.run(until=2)
    path = mp.output_efield_z(sim)
    data = _load(path)
    ez = np.asarray(sim.get_array(component=mp.Ez))
    got = data["ez"] if "ez" in data else data["ez.r"] + 1j * data["ez.i"]
    np.testing.assert_allclose(got, ez)


def test_to_appended_stacks_time_axis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _sim()
    sim.run(mp.at_every(1.0, mp.to_appended("ezt", mp.output_efield_z)),
            until=3)
    path = sim.last_output_file
    data = _load(path)
    key = "ez" if "ez" in data else "ez.r"
    assert data[key].ndim == 3 and data[key].shape[0] >= 2  # time axis added


def test_stop_when_energy_decayed_runs():
    sim = _sim()
    sim.run(until=[mp.stop_when_energy_decayed(dt=5, decay_by=1e-3),
                   lambda s: s.round_time() > 60])
    assert sim.round_time() > 5                    # sampled at least once
    assert math.isfinite(sim.round_time())
