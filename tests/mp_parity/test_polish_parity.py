# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Polish-wave parity: real behavior behind the last-mile meep surface --
Courant/geometry_center Simulation params, Volume geometry methods, the
DftObj(func, args) protocol, Source.volume/add_source, EigenModeSource as a
Source subclass with eig_* params, the meep-facing MaterialGrid, and the
accept-and-raise Simulation params."""
import numpy as np
import pytest
import torch

import meep_torch.mp as mp

EPS12 = mp.Medium(epsilon=12.0)


def _sim(geometry=None, **kw):
    kw.setdefault("cell_size", mp.Vector3(4, 4))
    kw.setdefault("resolution", 10)
    return mp.Simulation(geometry=geometry or [], **kw)


# ------------------------------------------------------------------- Courant
def test_courant_passes_through_to_grid_dt():
    assert _sim(Courant=0.25).grid.dt == pytest.approx(0.25 * 0.1)
    assert _sim().grid.dt == pytest.approx(0.5 * 0.1)          # meep default


def test_courant_changes_step_count_for_same_physical_time():
    a, b = _sim(Courant=0.25), _sim(Courant=0.5)
    a.run(until=2)
    b.run(until=2)
    assert a._t_step == 2 * b._t_step


def test_courant_out_of_engine_stability_range_raises():
    with pytest.raises(ValueError):
        _sim(Courant=0.9)             # 2D stability bound 1/sqrt(2)


# ----------------------------------------------------------- geometry_center
def test_geometry_center_shifts_block_raster_exactly():
    shifted = _sim([mp.Block(size=mp.Vector3(1, 1), center=mp.Vector3(1, 0.5),
                             material=EPS12)],
                   geometry_center=mp.Vector3(1, 0.5)).get_epsilon()
    manual = _sim([mp.Block(size=mp.Vector3(1, 1), center=mp.Vector3(0, 0),
                            material=EPS12)]).get_epsilon()
    assert np.array_equal(shifted, manual)


def test_geometry_center_shifts_prism_vertices():
    v = [mp.Vector3(-0.5, -0.5), mp.Vector3(0.5, -0.5), mp.Vector3(0, 0.5)]
    shifted = _sim([mp.Prism([p + mp.Vector3(1, 1) for p in v], material=EPS12)],
                   geometry_center=mp.Vector3(1, 1)).get_epsilon()
    manual = _sim([mp.Prism(v, material=EPS12)]).get_epsilon()
    assert np.array_equal(shifted, manual)


def test_geometry_center_does_not_shift_sources():
    src = [mp.Source(mp.GaussianSource(1.0, fwidth=0.2), component=mp.Ez,
                     center=mp.Vector3(0, 0))]
    s = _sim(sources=src, geometry_center=mp.Vector3(1, 0))
    s.run(until=5)
    t = _sim(sources=src)
    t.run(until=5)
    assert np.allclose(s.get_efield_z(), t.get_efield_z())


# -------------------------------------------------------------------- Volume
def test_volume_from_vertices_computes_center_and_size():
    v = mp.Volume(vertices=[mp.Vector3(-1, -2), mp.Vector3(3, -2),
                            mp.Vector3(3, 2), mp.Vector3(-1, 2)])
    assert (v.center.x, v.center.y) == (1.0, 0.0)
    assert (v.size.x, v.size.y) == (4.0, 4.0)


def test_volume_geometry_methods():
    v = mp.Volume(center=mp.Vector3(1, 1), size=mp.Vector3(2, 4))
    verts = v.get_vertices()
    assert len(verts) == 4                       # z collapsed
    assert mp.Vector3(0, -1, 0) in verts and mp.Vector3(2, 3, 0) in verts
    assert v.pt_in_volume(mp.Vector3(1.5, 2.0))
    assert not v.pt_in_volume(mp.Vector3(2.5, 0.0))
    edges = v.get_edges()
    assert edges and all(len(e) == 2 for e in edges)
    assert v.is_cylindrical is False and v.dims == 2


# ------------------------------------------------------- DftObj(func, args)
def test_dft_handles_carry_meep_func_args_protocol():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Ez, center=mp.Vector3())])
    fr = mp.FluxRegion(center=mp.Vector3(1, 0), size=mp.Vector3(0, 2))
    fl = sim.add_flux(1.0, 0.2, 3, fr)
    assert fl.func == sim._add_flux and fl.args[1] == [fr]
    assert fl.swigobj is None and fl.nfreqs == 3 and fl.num_components == 4
    assert fl.regions == [fr]
    assert fl.swigobj_attr("name") == fl.name    # forwards to own attributes
    df = sim.add_dft_fields([mp.Ez], 1.0, 0.2, 2)
    assert df.num_components == 1 and df.nfreqs == 2
    sim.run(until=10)
    assert len(mp.get_fluxes(fl)) == 3           # retrieval path unchanged


def test_near2far_flux_of_radiating_dipole_is_positive_outward():
    sim = mp.Simulation(
        cell_size=mp.Vector3(6, 6), resolution=10,
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)])
    n2f = sim.add_near2far(
        1.0, 0.2, 3,
        mp.Near2FarRegion(center=mp.Vector3(+1.5, 0), size=mp.Vector3(0, 3), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-1.5, 0), size=mp.Vector3(0, 3), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(0, +1.5), size=mp.Vector3(3, 0), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -1.5), size=mp.Vector3(3, 0), weight=-1))
    sim.run(until=30)
    vals = n2f.flux(mp.X, mp.Volume(center=mp.Vector3(20, 0), size=mp.Vector3(0, 4)),
                    resolution=4)
    assert len(vals) == 3 and np.all(np.isfinite(vals))
    assert all(v > 0 for v in vals)              # outward through the +x far line


def test_near2far_region_direction_defaults_to_automatic():
    r = mp.Near2FarRegion(center=mp.Vector3())
    assert r.direction == mp.AUTOMATIC
    v = mp.Volume(center=mp.Vector3(1, 0), size=mp.Vector3(0, 2))
    assert mp.Near2FarRegion(volume=v).center == v.center


# -------------------------------------------------------------------- Source
def test_source_volume_supplies_center_and_size():
    v = mp.Volume(center=mp.Vector3(1, 0), size=mp.Vector3(0, 2))
    s = mp.Source(mp.GaussianSource(1.0, fwidth=0.2), component=mp.Ez, volume=v)
    assert s.center == v.center and s.size == v.size


def test_source_amp_data_and_amp_func_file_raise():
    with pytest.raises(NotImplementedError):
        mp.Source(mp.GaussianSource(1.0, fwidth=0.2), component=mp.Ez,
                  center=mp.Vector3(), amp_data=np.ones(4))
    with pytest.raises(NotImplementedError):
        mp.Source(mp.GaussianSource(1.0, fwidth=0.2), component=mp.Ez,
                  center=mp.Vector3(), amp_func_file="f.h5:amp")


def test_source_add_source_registers_with_sim():
    sim = _sim()
    s = mp.Source(mp.GaussianSource(1.0, fwidth=0.2), component=mp.Ez,
                  center=mp.Vector3())
    s.add_source(sim)
    assert sim._sources == [s]


# ---------------------------------------------------------- EigenModeSource
def test_eigenmodesource_is_a_source_with_meep_eig_params():
    assert issubclass(mp.EigenModeSource, mp.Source)
    es = mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                            center=mp.Vector3(-1, 0), size=mp.Vector3(0, 2))
    assert es.component == mp.Ez                 # ALL_COMPONENTS -> TM default
    assert es.direction == mp.AUTOMATIC and es.eig_band == 1
    assert es.eig_match_freq is True and es.eig_parity == mp.NO_PARITY
    assert es.eig_lattice_size == es.size and es.eig_lattice_center == es.center


def test_eigenmodesource_parity_selects_polarization():
    te = mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                            center=mp.Vector3(), eig_parity=mp.EVEN_Z)
    assert te.component == mp.Hz
    tm = mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                            center=mp.Vector3(), eig_parity=mp.ODD_Z)
    assert tm.component == mp.Ez
    with pytest.raises(ValueError):              # explicit component conflict
        mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                           center=mp.Vector3(), component=mp.Ez,
                           eig_parity=mp.EVEN_Z)
    # The transverse (y) parity filter is wired now: EVEN_Y selects the first
    # EVEN mode, which is band 2 of the neff ordering whenever the fundamental
    # is odd. This used to raise NotImplementedError.
    even_y = mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                                center=mp.Vector3(),
                                eig_parity=mp.EVEN_Y + mp.ODD_Z)
    assert even_y.eig_y_parity == +1
    odd_y = mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                               center=mp.Vector3(),
                               eig_parity=mp.ODD_Y + mp.ODD_Z)
    assert odd_y.eig_y_parity == -1
    assert mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                              center=mp.Vector3(),
                              eig_parity=mp.ODD_Z).eig_y_parity == 0
    with pytest.raises(ValueError):              # both parities at once
        mp.EigenModeSource(mp.GaussianSource(0.15, fwidth=0.1),
                           center=mp.Vector3(),
                           eig_parity=mp.EVEN_Y + mp.ODD_Y + mp.ODD_Z)


def test_eigenmodesource_eig_power_matches_gaussian_ft():
    g = mp.GaussianSource(0.15, fwidth=0.1)
    es = mp.EigenModeSource(g, center=mp.Vector3(), amplitude=2.0)
    assert es.eig_power(0.15) == pytest.approx(abs(2.0 * g.fourier_transform(0.15)) ** 2)


def test_eigenmodesource_unsupported_eig_params_raise():
    g = mp.GaussianSource(0.15, fwidth=0.1)
    with pytest.raises(NotImplementedError):
        mp.EigenModeSource(g, center=mp.Vector3(), eig_kpoint=mp.Vector3(0.4, 0.1))
    with pytest.raises(NotImplementedError):
        mp.EigenModeSource(g, center=mp.Vector3(), eig_match_freq=False)
    with pytest.raises(NotImplementedError):
        mp.EigenModeSource(g, center=mp.Vector3(), eig_vol=mp.Volume(center=mp.Vector3()))


# -------------------------------------------------------------- MaterialGrid
def test_material_grid_meep_signature_and_engine_mapping():
    from meep_torch.adjoint.problem import MaterialGrid as EngineMG
    mg = mp.MaterialGrid(mp.Vector3(4, 3), mp.Medium(epsilon=1.0),
                         mp.Medium(epsilon=6.0), grid_type="U_MEAN",
                         beta=8, eta=0.4, damping=0.1)
    assert isinstance(mg, EngineMG)
    assert mg.shape == (4, 3) and mg.num_params == 12    # z=0 -> 1, meep style
    assert mg.eps_min == 1.0 and mg.eps_max == 6.0
    assert mg.weights.shape == (12,) and np.all(mg.weights == 0.0)
    assert mg.grid_type == 2 and mg.beta == 8 and mg.eta == 0.4
    assert float(mg.eps(torch.tensor(1.0))) == pytest.approx(6.0)


def test_material_grid_weights_check_and_update():
    mg = mp.MaterialGrid((2, 2), 1.0, 12.0)
    mg.update_weights(np.full((2, 2), 0.25))
    assert np.all(mg.weights == 0.25)
    with pytest.raises(ValueError):
        mg.update_weights(np.ones(5))
    with pytest.warns(UserWarning):
        mg.update_weights(np.full(4, 1.5))       # clipped to 1
    assert np.all(mg.weights == 1.0)
    with pytest.raises(ValueError):
        mp.MaterialGrid((2, 2), 1.0, 12.0, grid_type="U_BOGUS")
    with pytest.raises(NotImplementedError):     # dispersive endpoint
        mp.MaterialGrid((2, 2), mp.Medium(epsilon=1.0), mp.Medium(
            epsilon=5.0, E_susceptibilities=[mp.LorentzianSusceptibility(
                frequency=1.1, gamma=1e-5, sigma=0.5)]))


def test_material_grid_drives_adjoint_problem_beta_eta():
    from meep_torch.adjoint import DesignRegion, OptimizationProblem
    mg = mp.MaterialGrid(mp.Vector3(4, 3), mp.Medium(epsilon=1.0),
                         mp.Medium(epsilon=6.0), beta=8, eta=0.4)
    region = DesignRegion(mg, ix0=0, ix1=8, iy0=0, iy1=6)
    base = torch.ones(10, 8, dtype=torch.float64)
    opt = OptimizationProblem(lambda eps: eps.sum(), base, region)
    assert opt.beta == 8 and opt.eta == 0.4      # picked up from the grid
    f0, grad = opt()
    assert grad.shape == (4, 3) and np.isfinite(f0)
    explicit = OptimizationProblem(lambda eps: eps.sum(), base, region, beta=2)
    assert explicit.beta == 2                    # explicit arg still wins


# ------------------------------------------------------- GeometricObject.info
def test_geometric_object_info_prints_properties(capsys):
    mp.Cylinder(radius=0.5, center=mp.Vector3(1, 0), material=EPS12).info(indent_by=2)
    out = capsys.readouterr().out
    assert "Cylinder" in out and "radius = 0.5" in out and out.startswith("  ")


# ------------------------------------------- Simulation params + new methods
def test_simulation_unsupported_params_raise():
    # k_point no longer raises: the Bloch shim (mp/bloch_shim.py) drives it
    for kw in ({"m": 1}, {"epsilon_func": lambda p: 1.0},
               {"epsilon_input_file": "eps.h5"}, {"force_all_components": True},
               {"output_volume": mp.Volume(center=mp.Vector3())},
               {"bfast_scaled_k": (0.1, 0, 0)}):
        with pytest.raises(NotImplementedError):
            _sim(**kw)
    # force_complex_fields left this group: 2D carries the companion
    # imaginary field set (solve_cw). Still unported outside 2D/cylindrical.
    assert _sim(force_complex_fields=True).force_complex_fields
    with pytest.raises(NotImplementedError):
        _sim(force_complex_fields=True, dimensions=3)


def test_simulation_stored_params_and_eps_averaging_warns():
    sim = _sim(filename_prefix="runA", progress_interval=10, num_chunks=4,
               extra_materials=[EPS12], subpixel_tol=1e-3)
    assert sim.get_filename_prefix() == "runA"
    assert sim.progress_interval == 10 and sim.num_chunks == 4
    assert sim.extra_materials == [EPS12] and sim.subpixel_tol == 1e-3
    assert _sim().filename_prefix is None
    with pytest.warns(UserWarning):
        _sim(eps_averaging=False)


def test_change_sources_switches_polarization_and_rebuilds():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Ez, center=mp.Vector3())])
    assert sim._pol == "tm"
    sim.change_sources([mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Hz, center=mp.Vector3())])
    assert sim._pol == "te"
    sim.run(until=2)                             # rebuilt coeffs actually step
    # (40, 40): get_array reports every component on the CELL-CENTRE grid, as
    # meep does, not on the staggered grid it is stored on. This asserted
    # (39, 39) when Hz came back on its own Yee subgrid.
    assert sim.get_hfield_z().shape == (40, 40)


def test_clear_dft_monitors_and_add_sources():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Ez, center=mp.Vector3())])
    sim.add_flux(1.0, 0.2, 2, mp.FluxRegion(center=mp.Vector3(1, 0),
                                            size=mp.Vector3(0, 2)))
    assert sim._monitors
    sim.clear_dft_monitors()
    assert sim._monitors == [] and sim._last_acc == {}
    sim.add_sources()                            # allocates state, meep no-op
    assert sim._state is not None


def test_field_getters_after_tm_run():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Ez, center=mp.Vector3())])
    sim.run(until=5)
    assert sim.get_efield_z().shape == (40, 40)
    assert sim.get_hfield_x().shape == (40, 39)  # staggered Hx nodes
    assert sim.get_hfield_y().shape == (39, 40)
    with pytest.raises(NotImplementedError):
        sim.get_efield_x()                       # TE-only component on a TM run


def test_predefined_media_and_materials_module():
    """meep geom.py constants + the meep materials library port."""
    assert mp.air.epsilon_diag.x == 1.0 and mp.vacuum.epsilon_diag.x == 1.0
    assert mp.metal.epsilon_diag.x == -np.inf
    assert mp.perfect_electric_conductor.epsilon_diag.x == -np.inf
    assert mp.perfect_magnetic_conductor.mu_diag.x == -np.inf
    from meep_torch.mp import materials
    assert isinstance(materials.Au, mp.Medium)
    assert isinstance(materials.fused_quartz, mp.Medium)


def test_sources_and_symmetries_mutable_after_construction():
    """meep sim.sources / sim.symmetries are appendable lists; polarization
    and mirror state settle when the fields are built (holey-wvg flow)."""
    sim = _sim(sources=[])
    sim.sources.append(mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                 component=mp.Ez, center=mp.Vector3()))
    sim.symmetries.append(mp.Mirror(mp.Y, phase=1))
    f = sim.add_flux(1.0, 0.2, 2, mp.FluxRegion(center=mp.Vector3(1, 0),
                                                size=mp.Vector3(0, 2)))
    assert sim._pol == "tm" and sim._sym_cy > 0
    sim.run(until=5)
    assert np.all(np.isfinite(mp.get_fluxes(f)))


def test_run_sources_until_and_run_index():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Ez, center=mp.Vector3())])
    sim._run_sources_until(1.0, [])
    cut = sim._sources[0].src.profile().t_end
    assert abs(sim.round_time() - (cut + 1.0)) <= sim.grid.dt + 1e-9
    assert sim.run_index == 1


def test_add_dft_fields_region_slices_array():
    sim = _sim(sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                  component=mp.Ez, center=mp.Vector3())])
    d = sim.add_dft_fields([mp.Ez], 1.0, 0, 1, center=mp.Vector3(0, 0),
                           size=mp.Vector3(2, 1))
    sim.run(until=10)
    arr = sim.get_dft_array(d, mp.Ez, 0)
    assert arr.shape == (20, 10)                # 2x1 region at resolution 10
    x, y, z, w = sim.get_array_metadata(dft_cell=d)
    assert len(x) == arr.shape[0] and len(y) == arr.shape[1]


def test_run_k_points_dispersion_branches():
    """0D homogeneous Bloch run: eps(w) = (k/w)^2 matches the Lorentz medium
    (the material-dispersion tutorial physics)."""
    susc = [mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)]
    dm = mp.Medium(epsilon=2.25, E_susceptibilities=susc)
    sim = mp.Simulation(cell_size=mp.Vector3(), resolution=20, geometry=[],
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=2.0),
                                           component=mp.Ez,
                                           center=mp.Vector3())],
                        default_material=dm)
    (freqs,) = sim.run_k_points(150, [mp.Vector3(1.25)])
    eps = sorted((1.25 / f.real) ** 2 for f in freqs if f.real > 0.1)
    ref = [2.25 + 0.5 * 1.1**2 / (1.1**2 - w**2)
           for w in sorted(f.real for f in freqs if f.real > 0.1)]
    np.testing.assert_allclose(eps, sorted(ref, reverse=False), rtol=0.05)


def test_complex_amp_func_launches_oblique_planewave():
    """pw-source: complex amp_func e^{ik.y} on the real-field engine (the
    imaginary part rides the quadrature carrier)."""
    import cmath
    k = mp.Vector3(1.0, 0.4).unit().scale(2 * np.pi * 1.0)

    def pw_amp(k0, x0):
        def _amp(x):
            return cmath.exp(1j * k0.dot(x + x0))
        return _amp

    sim = _sim(cell_size=mp.Vector3(6, 6), boundary_layers=[mp.PML(1.0)],
               sources=[mp.Source(mp.ContinuousSource(frequency=1.0, fwidth=100),
                                  component=mp.Ez,
                                  center=mp.Vector3(-2, 0),
                                  size=mp.Vector3(0, 6),
                                  amp_func=pw_amp(k, mp.Vector3(x=-2)))])
    sim.run(until=15)
    ez = sim.get_efield_z()
    assert np.all(np.isfinite(ez)) and np.abs(ez).max() > 1e-3
