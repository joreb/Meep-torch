# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Final-12 parity classes: Energy/Force/ModeRegion, DftEnergy, DftForce,
GaussianBeam2D/3DSource, IndexedSource, DiffractedPlanewave, PadeDFT,
Verbosity, dft_data. Physics gates: vacuum-planewave energy equality,
radiation-pressure MST force, Pade-vs-harminv cavity peak."""
import math

import numpy as np
import pytest

import meep_torch.mp as mp

FCEN, DF, NF = 1.0, 0.2, 3


# ---------------------------------------------------------------- regions
def _monitor_sim():
    return mp.Simulation(
        cell_size=mp.Vector3(10, 4), resolution=10,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3(-3, 0))],
        boundary_layers=[mp.PML(1.0)])


def test_regions_structure_and_mode_monitor():
    assert mp.ModeRegion is mp.FluxRegion          # meep alias
    er = mp.EnergyRegion(center=mp.Vector3(), size=mp.Vector3(1, 1), weight=2.0)
    fr = mp.ForceRegion(center=mp.Vector3(), size=mp.Vector3(0, 2), direction=mp.X)
    assert isinstance(er, mp.FluxRegion) and er.weight == 2.0
    assert isinstance(fr, mp.FluxRegion) and fr.direction == mp.X
    vol = mp.Volume(center=mp.Vector3(1, 0), size=mp.Vector3(0, 2))
    assert mp.FluxRegion(volume=vol).center is vol.center
    m = _monitor_sim().add_mode_monitor(
        FCEN, DF, NF, mp.ModeRegion(center=mp.Vector3(3, 0), size=mp.Vector3(0, 2)))
    assert isinstance(m, mp.DftFlux) and len(m.freqs) == NF


# ------------------------------------------- DftEnergy / DftForce physics
def _planewave_sim(mirror=False):
    geometry = []
    if mirror:
        geometry.append(mp.Block(center=mp.Vector3(5.5, 0),
                                 size=mp.Vector3(1, mp.inf),
                                 material=mp.Medium(epsilon=1.0, D_conductivity=1e4)))
    return mp.Simulation(
        cell_size=mp.Vector3(20, 8), resolution=10, geometry=geometry,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3(-8, 0), size=mp.Vector3(0, mp.inf))],
        boundary_layers=[mp.PML(1.0)])


def test_dft_energy_vacuum_planewave_electric_equals_magnetic():
    sim = _planewave_sim()
    e = sim.add_energy(FCEN, DF, NF, mp.EnergyRegion(center=mp.Vector3(0, 0),
                                                     size=mp.Vector3(4, 4)))
    assert isinstance(e, mp.DftEnergy) and e.nfreqs == NF
    sim.run(until=60)
    ee = mp.get_electric_energy(e)
    me = mp.get_magnetic_energy(e)
    te = mp.get_total_energy(e)
    assert len(ee) == len(me) == len(te) == NF
    for a, b, t in zip(ee, me, te):
        assert a > 0
        assert abs(a - b) / a < 0.02, f"electric {a} vs magnetic {b}"
        assert t == pytest.approx(a + b)
    assert mp.get_energy_freqs(e) == pytest.approx(
        np.linspace(FCEN - DF / 2, FCEN + DF / 2, NF).tolist())


def test_dft_force_radiation_pressure_on_mirror():
    """Fx on the mirror box == (1 + R - T) * S_inc * Ly, all measured through
    the shim: 2*flux0(left) - flux_m(left) - flux_m(right) (the physics of
    tests/examples/test_w6_force.py driven via add_force/add_flux)."""
    LY = 4.0
    left = dict(center=mp.Vector3(4, 0), size=mp.Vector3(0, LY))
    right = dict(center=mp.Vector3(7, 0), size=mp.Vector3(0, LY))

    sim0 = _planewave_sim()
    l0 = sim0.add_flux(FCEN, DF, NF, mp.FluxRegion(**left))
    lw = sim0.add_flux(FCEN, DF, NF, mp.FluxRegion(weight=-1.0, **left))
    sim0.run(until=80)
    f_l0 = np.array(mp.get_fluxes(l0))
    assert np.allclose(np.array(mp.get_fluxes(lw)), -f_l0)   # weight wiring

    sim = _planewave_sim(mirror=True)
    l1 = sim.add_flux(FCEN, DF, NF, mp.FluxRegion(**left))
    r1 = sim.add_flux(FCEN, DF, NF, mp.FluxRegion(**right))
    box = [mp.ForceRegion(direction=mp.X, **left),
           mp.ForceRegion(direction=mp.X, **right),
           mp.ForceRegion(center=mp.Vector3(5.5, LY / 2), size=mp.Vector3(3, 0),
                          direction=mp.X),
           mp.ForceRegion(center=mp.Vector3(5.5, -LY / 2), size=mp.Vector3(3, 0),
                          direction=mp.X)]
    fX = sim.add_force(FCEN, DF, NF, *box)
    fY = sim.add_force(FCEN, DF, NF,
                       *[mp.ForceRegion(r.center, r.size, direction=mp.Y)
                         for r in box])
    assert isinstance(fX, mp.DftForce) and fX.nfreqs == NF
    sim.run(until=80)
    Fx = np.array(mp.get_forces(fX))
    Fy = np.array(mp.get_forces(fY))
    want = 2.0 * f_l0 - np.array(mp.get_fluxes(l1)) - np.array(mp.get_fluxes(r1))
    for k in range(NF):
        assert Fx[k] > 0
        assert abs(Fx[k] - want[k]) / want[k] < 0.10, \
            f"f={mp.get_force_freqs(fX)[k]}: Fx={Fx[k]:.4e} vs (1+R-T)S Ly={want[k]:.4e}"
        assert abs(Fy[k]) < 0.05 * Fx[k]           # symmetric in y


# ------------------------------------------------------- GaussianBeam family
def _beam(cls, **kw):
    return cls(mp.GaussianSource(FCEN, fwidth=0.4), center=mp.Vector3(-3, 0),
               size=mp.Vector3(0, 4), beam_w0=1.0, beam_kdir=mp.Vector3(1, 0),
               beam_E0=mp.Vector3(0, 0, 1), **kw)


def test_gaussian_beam_hierarchy_and_envelope():
    b2 = _beam(mp.GaussianBeam2DSource)
    bd = _beam(mp.GaussianBeamSource)
    b3 = _beam(mp.GaussianBeam3DSource)
    assert isinstance(b2, mp.GaussianBeam3DSource)
    assert isinstance(bd, mp.GaussianBeam3DSource)
    p = mp.Vector3(0, 0.5, 0)
    assert b2.amp_func(p) == bd.amp_func(p) == pytest.approx(math.exp(-0.25))
    # exact-3D class: accepted at construction, raises when the 2D engine runs it
    sim3 = mp.Simulation(cell_size=mp.Vector3(10, 6), resolution=10,
                         sources=[b3], boundary_layers=[mp.PML(1.0)])
    with pytest.raises(NotImplementedError):
        sim3.run(until=1)
    # the 2D classes launch (envelope approximation)
    sim2 = mp.Simulation(cell_size=mp.Vector3(10, 6), resolution=10,
                         sources=[b2], boundary_layers=[mp.PML(1.0)])
    sim2.run(until=5)
    assert np.abs(sim2.get_array(component=mp.Ez)).max() > 0
    with pytest.raises(NotImplementedError):
        b2.get_fields(sim2)          # complex equivalent currents: honest raise
    # incoming_mask splits the plane by the waist
    X = np.stack(np.meshgrid(np.array([-1.0, 0.0, 1.0]), np.array([0.0]),
                             indexing="ij"))
    inc, out, waist = b2.incoming_mask(0.0, 0.0, 1.0, 0.0, X)
    assert inc[2, 0] and out[0, 0] and waist[1, 0]


def test_green2d_matches_meep_literal_formulas():
    """b.green2d (refactored) vs a literal transcription of meep's green2d."""
    from scipy.special import hankel1

    def meep_green2d(X, freq, eps, mu, X0, kdir, hankel, beam_E0):
        EH = np.zeros([6] + list(X.shape)[1:], dtype=complex)
        rhat = X - np.repeat(np.repeat(X0[:, np.newaxis, np.newaxis],
                                       X.shape[1], axis=1), X.shape[2], axis=2)
        r = np.sqrt(np.sum(rhat * rhat, axis=0))
        rhat = rhat / r[np.newaxis, :, :]
        omega = 2 * np.pi * freq
        k = omega * np.sqrt(eps * mu)
        ik = 1j * k
        kr = k * r
        Z = np.sqrt(mu / eps)
        H0_kr, H1_kr, H2_kr = hankel(0, kr), hankel(1, kr), hankel(2, kr)
        ikH1 = 0.25 * ik * H1_kr
        p = np.zeros(3, dtype=complex)
        p[0], p[1], p[2] = beam_E0.x, beam_E0.y, beam_E0.z
        pdotrhat = p[0] * rhat[0] + p[1] * rhat[1]
        rhatcrossp = rhat[0] * p[1] - rhat[1] * p[0]
        eHx = -rhat[1] * ikH1 * p[2]
        eHy = rhat[0] * ikH1 * p[2]
        eHz = -rhatcrossp * ikH1
        eEx = -(rhat[0] * (pdotrhat / r * 0.25 * Z)) * H1_kr + \
            (rhat[1] * (rhatcrossp * omega * mu * 0.125)) * (H0_kr - H2_kr)
        eEy = -(rhat[1] * (pdotrhat / r * 0.25 * Z)) * H1_kr - \
            (rhat[0] * (rhatcrossp * omega * mu * 0.125)) * (H0_kr - H2_kr)
        eEz = (-0.25 * omega * mu) * H0_kr * p[2]
        p = -np.array([-p[2] * np.imag(kdir), p[2] * np.real(kdir),
                       p[0] * np.imag(kdir) - p[1] * np.real(kdir)])
        pdotrhat = p[0] * rhat[0] + p[1] * rhat[1]
        rhatcrossp = rhat[0] * p[1] - rhat[1] * p[0]
        hEx = rhat[1] * ikH1 * p[2]
        hEy = -rhat[0] * ikH1 * p[2]
        hEz = rhatcrossp * ikH1
        hHx = -(rhat[0] * (pdotrhat / r * 0.25 / Z)) * H1_kr + \
            (rhat[1] * (rhatcrossp * omega * eps * 0.125)) * (H0_kr - H2_kr)
        hHy = -(rhat[1] * (pdotrhat / r * 0.25 / Z)) * H1_kr - \
            (rhat[0] * (rhatcrossp * omega * eps * 0.125)) * (H0_kr - H2_kr)
        hHz = (-0.25 * omega * eps) * H0_kr * p[2]
        EH[:] = np.array([eEx, eEy, eEz, eHx, eHy, eHz]) + \
            np.array([hEx, hEy, hEz, hHx, hHy, hHz])
        return EH

    b = _beam(mp.GaussianBeam2DSource)
    xs, ys = np.linspace(1.0, 8.0, 9), np.linspace(-2.0, 2.0, 7)
    X = np.stack(np.meshgrid(xs, ys, indexing="ij"))
    X0 = np.array([0.3, -0.2])
    E0 = mp.Vector3(0.4 + 0.1j, -0.7, 1.2 - 0.5j)
    args = (X, 0.9, 2.25, 1.0, X0, 0.6 + 0.8j, hankel1, E0)
    assert np.allclose(b.green2d(*args), meep_green2d(*args), rtol=1e-12)


# --------------------------------------------------------------- IndexedSource
def test_indexed_source_equals_uniform_line_source():
    src_t = mp.GaussianSource(FCEN, fwidth=0.5)
    region = mp.Volume(center=mp.Vector3(-3, 0), size=mp.Vector3(0, 2))

    def base(sources):
        return mp.Simulation(cell_size=mp.Vector3(10, 6), resolution=10,
                             sources=sources, boundary_layers=[mp.PML(1.0)])

    s1 = base([mp.IndexedSource(src_t, region, np.full(20, 0.7))])
    s1.run(until=20)
    s2 = base([mp.Source(src_t, component=mp.Ez, center=region.center,
                         size=region.size, amplitude=0.7)])
    s2.run(until=20)
    assert np.allclose(s1.get_array(component=mp.Ez),
                       s2.get_array(component=mp.Ez), atol=1e-12)

    with pytest.raises(ValueError):                 # wrong point count
        base([mp.IndexedSource(src_t, region, np.ones(7))]).run(until=1)
    with pytest.raises(NotImplementedError):        # complex amplitudes
        base([mp.IndexedSource(src_t, region, np.full(20, 1j))]).run(until=1)


# --------------------------------------------------------- DiffractedPlanewave
def test_diffracted_planewave_structural_and_raises():
    dp = mp.DiffractedPlanewave(g=[1, 0, 0], axis=mp.Vector3(0, 1, 0), s=1, p=0)
    assert dp.g == [1, 0, 0] and dp.s == 1 + 0j and dp.p == 0j
    with pytest.raises(NotImplementedError):
        mp.EigenModeSource(mp.GaussianSource(FCEN, fwidth=0.2), eig_band=dp)
    with pytest.raises(NotImplementedError):
        _monitor_sim().get_eigenmode_coefficients("m", dp)


# --------------------------------------------------------------------- PadeDFT
def test_pade_dft_peak_matches_harminv_on_cavity_ringdown():
    fcen = math.sqrt(0.5)
    metal = mp.Medium(epsilon=1.0, D_conductivity=1e4)
    a, t = 1.0, 0.1
    sim = mp.Simulation(
        cell_size=mp.Vector3(3, 3), resolution=20,
        geometry=[mp.Block(size=mp.Vector3(a + 2 * t, a + 2 * t, mp.inf),
                           material=metal),
                  mp.Block(size=mp.Vector3(a, a, mp.inf),
                           material=mp.Medium(epsilon=1.0))],
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3(0.1, 0.07))],
        boundary_layers=[mp.PML(0.5)])
    pt = mp.Vector3(0.1, 0.07)
    h = mp.Harminv(mp.Ez, pt, fcen, 0.4)
    p = mp.PadeDFT(c=mp.Ez, center=pt, sampling_interval=8)
    sim.run(mp.after_sources(h), mp.after_sources(p), until_after_sources=100)
    m0 = max(h.modes, key=lambda m: abs(m.amp))
    freqs = np.linspace(0.55, 0.85, 301)
    spec = p.spectrum(freqs)
    f_pade = freqs[int(np.argmax(spec))]
    assert abs(f_pade - m0.freq) / m0.freq < 0.02, \
        f"pade peak {f_pade:.4f} vs harminv {m0.freq:.4f}"


# ------------------------------------------------------------------- Verbosity
def test_verbosity_singleton_semantics():
    v = mp.verbosity
    assert isinstance(v, mp.Verbosity)
    assert mp.Verbosity() is v                     # singleton
    old = v.get()
    try:
        assert v.set(2) == old and v.get() == 2 and int(v) == 2
        assert v >= 2 and v <= 2 and v == 2 and v != 3 and not v > 2
        assert v.meep == 2                         # per-flag property
        v.meep = 1
        assert v.meep == 1 and v.get() == 2        # flag moves, master stays
        assert v(3) == 2 and v.get() == 3          # callable sets
        with pytest.raises(ValueError):
            v.set(4)

        class _CVar:
            verbosity = 1

        v.add_verbosity_var(_CVar(), "extra")
        v.set(1)
        assert v.extra == 1 and 1 in v.get_all()
    finally:
        v.set(old if old >= 0 else 1)


# -------------------------------------------------------------------- dft_data
def test_dft_data_structural():
    d = mp.dft_data([1.0, 2.0], ["Ez"], [None])
    assert d.freqs == [1.0, 2.0] and d.components == ["Ez"]
