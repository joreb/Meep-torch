# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Medium / susceptibility family / PML meep-signature parity + engine wiring."""
import math

import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.core.materials import LorentzPole, Material, analytic_eps


# ---------------------------------------------------------------- Medium core
def test_medium_scalar_folding_and_resolve():
    assert mp.Medium(epsilon=12).resolve() == (12.0, 0.0)
    assert mp.Medium(index=3.0).resolve() == (9.0, 0.0)
    m = mp.Medium(epsilon=4.0, chi3=0.02)
    assert m.epsilon_diag == mp.Vector3(4.0, 4.0, 4.0)
    assert m.E_chi3_diag == mp.Vector3(0.02, 0.02, 0.02)
    assert m.resolve() == (4.0, 0.02)


def test_medium_susceptibilities_resolve_to_material():
    lor = mp.LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)
    mat, chi3 = mp.Medium(epsilon=2.25, E_susceptibilities=[lor]).resolve()
    assert isinstance(mat, Material)
    assert mat.eps_inf == 2.25
    assert mat.poles == (LorentzPole(1.1, 1e-5, 0.5),)
    assert chi3 == 0.0
    # Drude -> engine 4-tuple pole (fn=0, gamma, sigma, fc=fp)
    dr = mp.DrudeSusceptibility(frequency=0.7, gamma=0.01, sigma=1.2)
    mat, _ = mp.Medium(epsilon=1.0, E_susceptibilities=[dr]).resolve()
    assert mat.poles == ((0.0, 0.01, 1.2, 0.7),)


def test_medium_engine_material_extension():
    from meep_torch.core.materials import SiO2
    assert mp.Medium(material=SiO2).resolve()[0] is SiO2
    assert mp.Medium(epsilon=SiO2).resolve()[0] is SiO2   # W1 compat form


def test_medium_shim_scalar_mirrors():
    m = mp.Medium(epsilon=1.0, D_conductivity=5.0, B_conductivity=2.0, mu=3.0)
    # mp/simulation.py + mp/geometry.py read these via float(getattr(...))
    assert float(getattr(m, "D_conductivity", 0.0) or 0.0) == 5.0
    assert float(getattr(m, "B_conductivity", 0.0) or 0.0) == 2.0
    assert float(getattr(m, "mu", 1.0) or 1.0) == 3.0
    assert m.D_conductivity_diag == mp.Vector3(5.0, 5.0, 5.0)


# --------------------------------------------------------- epsilon/mu accessor
def test_epsilon_accessor_matches_analytic_eps():
    lor = mp.LorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.5)
    m = mp.Medium(epsilon=2.25, E_susceptibilities=[lor])
    for f in (0.3, 0.9, 1.4):
        got = m.epsilon(f)
        assert got.shape == (3, 3)
        want = analytic_eps(2.25, [LorentzPole(1.1, 0.05, 0.5)], f)
        assert np.allclose(np.diag(got), want)
        assert np.count_nonzero(got - np.diag(np.diag(got))) == 0


def test_epsilon_accessor_array_and_conductivity():
    m = mp.Medium(epsilon=4.0, D_conductivity=0.8)
    freqs = np.array([0.5, 1.0, 2.0])
    got = m.epsilon(freqs)
    assert got.shape == (3, 3, 3)
    want = (1 + 1j * 0.8 / (2 * np.pi * freqs)) * 4.0
    assert np.allclose(got[:, 0, 0], want)
    assert np.allclose(m.mu(1.0), np.eye(3))


def test_valid_freq_range_enforced_by_accessor():
    m = mp.Medium(epsilon=2.0, valid_freq_range=mp.FreqRange(min=0.5, max=1.5))
    m.epsilon(1.0)
    with pytest.raises(ValueError, match="below the Medium's limit"):
        m.epsilon(0.1)
    with pytest.raises(ValueError, match="above the Medium's limit"):
        m.epsilon(2.0)


# ---------------------------------------------------------- transform / rotate
def test_medium_rotate_swaps_axes():
    m = mp.Medium(epsilon_diag=mp.Vector3(2.0, 3.0, 4.0))
    m.rotate(mp.Vector3(0, 0, 1), math.pi / 2)   # x <-> y about z
    assert np.allclose([m.epsilon_diag.x, m.epsilon_diag.y, m.epsilon_diag.z],
                       [3.0, 2.0, 4.0])
    assert np.allclose([m.epsilon_offdiag.x, m.epsilon_offdiag.y,
                        m.epsilon_offdiag.z], 0.0, atol=1e-12)


def test_transform_scales_by_det():
    # chi -> M chi M^T / |det M|; M = 2I gives 4/8 = 0.5x
    m = mp.Medium(epsilon_diag=mp.Vector3(2.0, 2.0, 2.0))
    m.transform(2.0 * np.eye(3))
    assert np.allclose([m.epsilon_diag.x, m.epsilon_diag.y, m.epsilon_diag.z], 1.0)


def test_susceptibility_transform_rotates_sigma():
    s = mp.LorentzianSusceptibility(frequency=1.0, sigma_diag=mp.Vector3(1.0, 2.0, 3.0))
    s.transform(mp.get_rotation_matrix(mp.Vector3(0, 0, 1), math.pi / 2))
    assert np.allclose([s.sigma_diag.x, s.sigma_diag.y, s.sigma_diag.z],
                       [2.0, 1.0, 3.0])


# ------------------------------------------------- engine-limitation raising
def test_unsupported_params_raise_at_resolve_not_construction():
    aniso = mp.Medium(epsilon_diag=mp.Vector3(2.0, 3.0, 2.0))   # constructs fine
    with pytest.raises(ValueError, match="anisotropic epsilon_diag"):
        aniso.resolve()
    with pytest.raises(ValueError, match="epsilon_offdiag"):
        mp.Medium(epsilon_offdiag=mp.Vector3(0.1, 0, 0)).resolve()
    with pytest.raises(ValueError, match="anisotropic mu_diag"):
        mp.Medium(mu_diag=mp.Vector3(1.0, 2.0, 1.0)).resolve()
    with pytest.raises(ValueError, match="H_susceptibilities"):
        mp.Medium(H_susceptibilities=[mp.LorentzianSusceptibility(frequency=1.0)]).resolve()
    with pytest.raises(ValueError, match="chi3 only"):
        mp.Medium(chi2=0.5).resolve()
    with pytest.raises(ValueError, match="D_conductivity_offdiag"):
        mp.Medium(D_conductivity_offdiag=mp.Vector3(1, 0, 0)).resolve()


def test_gyrotropic_noisy_multilevel_use_time_raises():
    gyro = mp.GyrotropicLorentzianSusceptibility(bias=mp.Vector3(0, 0, 1),
                                                 frequency=1.0, sigma=0.1)
    with pytest.raises(NotImplementedError, match="port_gyrotropic-2d"):
        mp.Medium(E_susceptibilities=[gyro]).resolve()
    with pytest.raises(NotImplementedError, match="port_gyrotropic-2d"):
        mp.GyrotropicDrudeSusceptibility(bias=mp.Vector3(1, 0, 0)).to_pole()
    with pytest.raises(NotImplementedError, match="port_gyrotropic-2d"):
        mp.GyrotropicSaturatedSusceptibility(frequency=1.0).to_pole()
    with pytest.raises(NotImplementedError, match="port_noisy-susceptibilities"):
        mp.NoisyLorentzianSusceptibility(noise_amp=0.1, frequency=1.0).to_pole()
    with pytest.raises(NotImplementedError, match="stepper_1d"):
        mp.MultilevelAtom(transitions=[mp.Transition(1, 2, frequency=1.0)]).to_pole()


def test_zero_bias_and_zero_noise_degrade_to_plain_poles():
    g = mp.GyrotropicLorentzianSusceptibility(frequency=1.1, gamma=0.05, sigma=0.5)
    assert g.to_pole() == mp.LorentzianSusceptibility(
        frequency=1.1, gamma=0.05, sigma=0.5).to_pole()
    n = mp.NoisyDrudeSusceptibility(frequency=0.7, gamma=0.01, sigma=1.2)
    assert n.to_pole() == mp.DrudeSusceptibility(
        frequency=0.7, gamma=0.01, sigma=1.2).to_pole()


def test_transition_validates_levels():
    t = mp.Transition(0, 1, frequency=2.0, gamma=0.1, pumping_rate=0.3)
    assert (t.from_level, t.to_level, t.frequency) == (0, 1, 2.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        mp.Transition(-1, 1)


# -------------------------------------------------------------- PML / Absorber
def test_pml_full_signature_and_defaults():
    p = mp.PML(thickness=1.0)
    assert (p.thickness, p.direction, p.side) == (1.0, mp.ALL, mp.ALL)
    assert (p.R_asymptotic, p.mean_stretch) == (1e-15, 1.0)
    assert p.pml_profile(0.5) == 0.25
    with pytest.raises(ValueError, match="thickness must be specified"):
        mp.PML()
    with pytest.raises(ValueError, match="all cell edges"):
        mp.PML(1.0, direction=mp.X)
    with pytest.raises(ValueError, match="all cell edges"):
        mp.Absorber(1.0, side=mp.Low)
    with pytest.raises(ValueError, match="mean_stretch"):
        mp.PML(1.0, mean_stretch=0.5)
    with pytest.raises(ValueError, match="pml_profile"):
        mp.PML(1.0, pml_profile=lambda u: u ** 3)


def test_absorber_subclasses_pml_but_dispatches_separately():
    assert issubclass(mp.Absorber, mp.PML)               # meep hierarchy
    a = mp.Absorber(1.0)
    # mp/simulation.py checks isinstance(bl, PML) before its Absorber elif;
    # the exact-PML instancecheck keeps absorbers out of the CPML branch
    assert not isinstance(a, mp.PML)
    assert isinstance(a, mp.Absorber)
    assert isinstance(mp.PML(1.0), mp.PML)


def test_simulation_routes_pml_and_absorber():
    def sim(bl):
        return mp.Simulation(cell_size=mp.Vector3(4, 4), resolution=10,
                             sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2),
                                                component=mp.Ez,
                                                center=mp.Vector3())],
                             boundary_layers=[bl])
    s_pml = sim(mp.PML(1.0))
    assert s_pml._npml > 0 and s_pml._nabs == 0
    s_abs = sim(mp.Absorber(1.0))
    assert s_abs._nabs > 0 and s_abs._npml == 0


def test_simulation_runs_with_meep_style_medium():
    # mu/D_conductivity through the shim's float(getattr(...)) reads
    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 4), resolution=10,
        geometry=[mp.Block(size=mp.Vector3(1, 1),
                           material=mp.Medium(epsilon=4.0, D_conductivity=1.0, mu=2.0))],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3(-1, 0))],
        boundary_layers=[mp.PML(1.0)])
    sim.run(until=2)
