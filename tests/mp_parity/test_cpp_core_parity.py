# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""C++-binding core parity: fields/structure/boundary_region/h5file (+ the
unowned lowercase leftovers absorber, cond_profile, flux_vol, initialize).

Physics gate: a structure+fields pair built on a small vacuum grid, driven
one step at a time by fields.step() with a point source, reproduces the same
run through core.run.run_tm bit-for-bit (identical Ez/Hx/Hy tensors and DFT
flux accumulators after 100 steps); get_field samples the right cell.
"""
import json
import pathlib

import numpy as np
import pytest
import torch

import meep_torch.mp as mp
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import LineSource, PointSource, delta_weights
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.mp.cpp.core import flux_vol as core_flux_vol

API = json.load(open(pathlib.Path(__file__).parents[2] / "scripts" / "meep_api.json"))

NX, NY, DX = 40, 30, 0.05


def _grid():
    return Grid(nx=NX, ny=NY, dx=DX)


def _fields_with_point_source():
    grid = _grid()
    st = mp.structure(grid, 1.0)
    f = mp.fields(st)
    f.add_point_source(mp.Ez, mp.GaussianSource(frequency=1.0, fwidth=0.3),
                       mp.Vector3(0, 0))
    return grid, st, f


# ------------------------------------------------------------ physics gate ---

def test_step_matches_run_tm_bit_for_bit():
    grid, st, f = _fields_with_point_source()
    fh = f.add_dft_flux(mp.X, mp.Volume(center=mp.Vector3(0.5, 0),
                                        size=mp.Vector3(0, 1.0)), [1.0])
    for _ in range(100):
        f.step()

    # same run through the engine's own runner: the origin is a cell
    # BOUNDARY on this even grid, so meep's placement spreads the delta over
    # cells 19/20 x 14/15 (core.sources.delta_weights, gated against meep's
    # measured footprint); the facade must build the identical mask
    prof = mp.GaussianSource(frequency=1.0, fwidth=0.3).profile()
    src = PointSource(grid, delta_weights(0.0, NX * DX, DX, NX),
                      delta_weights(0.0, NY * DX, DX, NY), prof)
    mon = FluxMonitorTM(grid, [1.0], x_normal=True, idx=30, lo=5, hi=25)
    ref, (acc,) = run_tm(init_tm_state(grid), build_tm_coeffs(grid, eps_inf=1.0),
                         100, source=src, monitors=[mon])

    assert float(f._state.Ez.abs().max()) > 0            # the source actually fired
    assert torch.equal(f._state.Ez, ref.Ez)
    assert torch.equal(f._state.Hx, ref.Hx)
    assert torch.equal(f._state.Hy, ref.Hy)
    assert torch.equal(f._monitors[fh.keys[0]][1], acc)  # DFT fold bit-for-bit
    assert fh.flux() == mon.flux(acc, grid.dx).detach().cpu().numpy().tolist()


def test_get_field_samples_state():
    _, _, f = _fields_with_point_source()
    for _ in range(50):
        f.step()
    pt = mp.Vector3(0.2, -0.1)                           # cell (24, 13)
    assert f.get_field(mp.Ez, pt) == float(f._state.Ez[24, 13])
    assert f.get_field_from_comp(mp.Ez, pt) == f.get_field(mp.Ez, pt)
    assert f.get_field("Hx", mp.Vector3(0, 0)) == float(f._state.Hx[20, 15])
    with pytest.raises(NotImplementedError):
        f.get_field(mp.Hz, pt)                           # TE component on a TM run
    assert f.time() == f.round_time() == 50 * f.s.coeffs.dt


def test_fields_reset_and_energy():
    _, _, f = _fields_with_point_source()
    for _ in range(30):
        f.step()
    assert f.field_energy() > 0
    assert f.total_energy() == f.field_energy()
    assert f.max_abs("Ez") > 0
    f.zero_fields()
    assert float(f._state.Ez.abs().max()) == 0
    f.reset()
    assert f.time() == 0


# ----------------------------------------------------------- structure API ---

def test_structure_from_simulation_adopts_coeffs():
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 3), resolution=10,
        boundary_layers=[mp.PML(0.5)],
        sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.2),
                           component=mp.Ez, center=mp.Vector3())])
    st = mp.structure(sim)
    assert st.coeffs is sim._coeffs                      # same tensors, bit-for-bit
    assert st.npml == sim._npml
    f = mp.fields(st)
    f.add_point_source(mp.Ez, mp.GaussianSource(frequency=1.0, fwidth=0.2),
                       mp.Vector3())
    for _ in range(10):
        f.step()
    assert float(f._state.Ez.abs().max()) > 0


def test_structure_mutators_rebuild_coeffs():
    st = mp.structure(_grid(), 1.0)
    st.set_epsilon(4.0)
    assert st.get_eps(mp.Vector3()) == 4.0
    assert st.get_inveps(mp.Vector3()) == 0.25
    assert st.get_chi1inv(pt=mp.Vector3()) == 0.25
    assert st.max_eps() == 4.0
    assert torch.all(st.coeffs.eps_inf == 4.0)

    st.set_conductivity(mp.Ez, 2.0)                      # E/D component -> sigma_d
    assert st.coeffs.d1 is not None
    assert st.get_mu(mp.Vector3()) == 1.0
    st.set_mu(2.0)
    assert st.get_mu(mp.Vector3()) == 2.0

    assert not st.has_chi()
    st.add_susceptibility(mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5,
                                                      sigma=0.5))
    assert st.has_chi() and st.coeffs.c1.shape[0] == 1
    assert mp.fields(st)._state.P_cur.shape[0] == 1      # pole state follows
    st.remove_susceptibilities()
    assert st.coeffs.c1.shape[0] == 0

    # callable epsilon, evaluated at meep (center-origin) cell centers
    st.set_epsilon(lambda v: 2.0 if v.x > 0 else 1.0)
    assert st.get_eps(mp.Vector3(0.5, 0)) == 2.0
    assert st.get_eps(mp.Vector3(-0.5, 0)) == 1.0


def test_structure_mix_and_layout():
    a = mp.structure(_grid(), 1.0)
    b = mp.structure(_grid(), 4.0)
    assert a.equal_layout(b)
    a.mix_with(b, 0.5)                                   # chi1inv mixing: 1/(0.5 + 0.125)
    assert a.get_eps(mp.Vector3()) == pytest.approx(1.6)
    assert a.estimated_cost() == NX * NY
    with pytest.raises(NotImplementedError):
        mp.structure(_grid(), 1.0, num_chunks=4)
    for m in ("get_chunk_owners", "get_chunk_volumes", "get_binary_partition",
              "dump", "load"):
        with pytest.raises(NotImplementedError):
            getattr(a, m)()


# -------------------------------------------------------- boundary_region ---

def test_boundary_region_to_engine_pml():
    br = mp.boundary_region(thickness=0.5)
    br.check_ok()
    st = mp.structure(_grid(), 1.0, br)
    assert st.npml == 10
    ref = build_tm_coeffs(_grid(), eps_inf=1.0, npml=10)
    assert torch.equal(st.coeffs.b_hx, ref.b_hx)
    assert torch.equal(st.coeffs.a_ey, ref.a_ey)

    other = mp.boundary_region(thickness=1.0)
    br2 = mp.boundary_region(thickness=0.1).boundary_region_assign(other)
    assert br2.thickness == 1.0
    with pytest.raises(ValueError):
        mp.boundary_region(thickness=-1.0).check_ok()

    st2 = mp.structure(_grid(), 1.0)
    mp.boundary_region(thickness=0.25).br_apply(st2)
    assert st2.npml == 5


def test_absorber_maps_to_sigma_ramp():
    st = mp.structure(_grid(), 1.0, mp.absorber(0.25))
    assert st.npml == 0 and st._nabs == 5
    assert st.coeffs.d1 is not None                      # conductivity path engaged
    assert isinstance(mp.absorber(0.25), mp.boundary_region)


# ------------------------------------------- sources / monitors / flux_vol ---

def test_volume_source_and_flux_vol():
    grid = _grid()
    st = mp.structure(grid, 1.0)
    f = mp.fields(st)
    f.add_volume_source(mp.Ez, mp.GaussianSource(frequency=1.0, fwidth=0.3),
                        mp.Volume(center=mp.Vector3(-0.5, 0), size=mp.Vector3(0, 1.0)))
    fv = f.add_flux_vol(mp.X, mp.Volume(center=mp.Vector3(0.5, 0),
                                        size=mp.Vector3(0, 1.0)))
    assert isinstance(fv, core_flux_vol)
    for _ in range(60):
        f.step()
    assert isinstance(fv.flux(), float) and np.isfinite(fv.flux())
    assert fv.flux() == f.flux_in_box(mp.X, fv.where_)
    for m in ("update", "update_half"):
        with pytest.raises(NotImplementedError):
            getattr(fv, m)()

    ha = f.add_dft_fields([mp.Ez], [1.0])
    hp = f.add_dft_pt(mp.Ez, mp.Vector3(0.5, 0), [1.0])
    for _ in range(20):
        f.step()
    arr = f.get_dft_array(ha, mp.Ez)
    assert arr.shape == (NX, NY) and arr.dtype == np.complex128
    assert np.iscomplexobj(f.get_dft_array(hp))
    assert f.dft_maxfreq() == 1.0
    assert f.dft_monitor_size() > 0
    f.clear_dft_monitors()
    assert f.dft_monitor_size() == 0


def test_initialize_field_and_slices():
    st = mp.structure(_grid(), 1.0)
    f = mp.fields(st)
    f.initialize_field(mp.Ez, lambda v: v.x)
    a = f.get_array_slice(component=mp.Ez)
    assert a.shape == (NX, NY) and a[0, 0] < 0 < a[-1, 0]
    assert f.get_array_slice_dimensions() == (NX, NY)
    assert f.have_component("Hy") and not f.have_component("Hz")
    assert f.normal_direction(mp.Volume(size=mp.Vector3(0, 1))) == mp.X
    assert f.total_volume() == pytest.approx(NX * DX * NY * DX)


def test_fields_raise_groups():
    f = mp.fields(mp.structure(_grid(), 1.0))
    for m in ("loop_in_chunks", "process_incoming_chunk_data", "dump", "load",
              "output_hdf5", "get_eigenmode", "use_bloch", "energy_in_box",
              "get_timing_data", "add_dft_near2far", "synchronize_magnetic_fields"):
        with pytest.raises(NotImplementedError):
            getattr(f, m)()
    # solve_cw left that group when it was implemented; it now refuses for a
    # reason of its own (real fields) rather than as an unported stub
    with pytest.raises(RuntimeError, match="force_complex_fields"):
        f.solve_cw()


# ------------------------------------------------- h5file / small leftovers ---

def test_h5file(tmp_path):
    h = mp.h5file(str(tmp_path / "t.h5"), mp.h5file.WRITE)
    assert h.file_name().endswith("t.h5")
    with pytest.raises(ValueError):
        mp.h5file("x.h5", mode="bogus")
    with pytest.raises(NotImplementedError):
        mp.h5file("x.h5", parallel_=True)
    with pytest.raises(NotImplementedError):
        h.write_chunk()                                  # chunked/parallel IO
    try:
        import h5py  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="h5py"):
            h.write("d", np.ones(3))
        return
    h.write("d", np.arange(3.0))
    assert h.dataset_exists("d") and not h.dataset_exists("nope")
    assert h.read_size("d") == (3,)
    assert np.array_equal(h.read("d"), np.arange(3.0))
    h.remove_data("d")
    assert not h.dataset_exists("d")
    h.remove()


def test_small_leftovers():
    init = mp.initialize(argc=0, argv=None)
    assert init.elapsed_time() >= 0.0
    assert mp.cond_profile()(0.5) == 0.25                # meep quadratic default
    assert mp.cond_profile(lambda u: u)(0.5) == 0.5


# ----------------------------------------------- SWIG-surface completeness ---

@pytest.mark.parametrize("name", ["fields", "structure", "boundary_region",
                                  "h5file", "absorber", "cond_profile",
                                  "flux_vol", "initialize"])
def test_swig_surface_complete(name):
    spec = API["classes"][name]
    cls = getattr(mp, name)
    missing = set(spec["methods"]) - set(dir(cls))
    assert not missing, f"{name} missing methods: {sorted(missing)}"
