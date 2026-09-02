# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp shim parity: SourceTime (start_time/is_integrated), GaussianSource
fourier_transform, Susceptibility/DrudeSusceptibility, GaussianBeamSource.

Drude gate: a Drude slab's reflectance through mp.Simulation (Medium ->
DrudeSusceptibility.to_pole -> engine 4-tuple pole) matches the analytic 1D
transfer matrix within 5% (dx=0.025; the engine-level pole is converged to
~2.5% at this resolution, see tests/examples/test_peraxis_dispersion.py).
"""
import math

import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.mp.material import (DrudeSusceptibility, LorentzianSusceptibility,
                                    Susceptibility)
from meep_torch.mp.sources import (ContinuousSource, CustomSource,
                                   GaussianBeamSource, GaussianSource, Source,
                                   SourceTime)
from meep_torch.mp.vec import Vector3


# --- SourceTime: base class, start_time, is_integrated ---

def test_sourcetime_base_class():
    for cls in (GaussianSource, ContinuousSource, CustomSource):
        assert issubclass(cls, SourceTime)


@pytest.mark.parametrize("mk", [
    lambda st: GaussianSource(frequency=1.0, fwidth=0.2, start_time=st),
    lambda st: ContinuousSource(frequency=0.8, width=2.0, start_time=st),
], ids=["gaussian", "continuous"])
def test_start_time_gates_and_shifts_profile(mk):
    delay = 4.0
    base = mk(0.0).profile()
    shifted = mk(delay).profile()
    for t in (0.0, 1.0, delay - 1e-9):
        assert shifted.amplitude(t) == 0.0
    for t in (0.3, 1.7, 5.0, 20.0):
        assert shifted.amplitude(t + delay) == pytest.approx(base.amplitude(t))


def test_custom_source_windows_without_shifting():
    # meep custom_src_time: src_func(t) inside [start_time, end_time], 0 outside
    fn = lambda t: math.sin(3.0 * t)
    src = CustomSource(fn, start_time=4.0, end_time=10.0)
    prof = src.profile()
    assert prof.amplitude(3.9) == 0.0
    assert prof.amplitude(10.1) == 0.0
    for t in (4.0, 5.3, 10.0):
        assert prof.amplitude(t) == pytest.approx(fn(t))
    # defaults (-1e20, 1e20) -> bare engine profile, no window
    from meep_torch.core.sources import CustomSource as _CS
    assert isinstance(CustomSource(fn).profile(), _CS)


def test_meep_signature_equivalences():
    # width and fwidth are two spellings of the same thing: width=max(width, 1/fwidth)
    a = GaussianSource(frequency=1.0, width=5.0).profile()
    b = GaussianSource(frequency=1.0, fwidth=0.2).profile()
    for t in (0.0, 3.0, 25.0, 40.0):
        assert a.amplitude(t) == pytest.approx(b.amplitude(t))
    # wavelength -> frequency = 1/wavelength
    assert GaussianSource(wavelength=0.5, fwidth=0.2).frequency == pytest.approx(2.0)
    assert ContinuousSource(wavelength=2.0).frequency == pytest.approx(0.5)
    with pytest.raises(ValueError):
        GaussianSource()
    with pytest.raises(NotImplementedError):
        ContinuousSource(frequency=1.0, end_time=50.0).profile()


def test_start_time_zero_keeps_bare_engine_profile():
    from meep_torch.core.sources import GaussianPulse
    assert isinstance(GaussianSource(frequency=1.0, fwidth=0.2).profile(),
                      GaussianPulse)


def test_is_integrated_stored_and_raises_at_profile():
    s = GaussianSource(frequency=1.0, fwidth=0.2, is_integrated=True)
    assert s.is_integrated is True
    with pytest.raises(NotImplementedError):
        s.profile()
    assert GaussianSource(frequency=1.0, fwidth=0.2).is_integrated is False


# --- GaussianSource.fourier_transform ---

def test_fourier_transform_matches_sampled_profile():
    """meep analytic FT (exp(+i*omega*t) convention, complex profile) vs a
    Riemann-sum FT of the sampled real engine profile: the real cos profile
    transforms as 0.5*(F(f) + conj(F(-f)))."""
    src = GaussianSource(frequency=1.0, fwidth=0.2)
    prof = src.profile()
    dt = 1e-3
    ts = np.arange(0.0, prof.t_end + 20.0, dt)
    g = np.array([prof.amplitude(t) for t in ts])
    scale = abs(src.fourier_transform(src.frequency))     # peak magnitude
    for f in (0.9, 1.0, 1.08):
        num = np.sum(g * np.exp(2j * np.pi * f * ts)) * dt
        exact = 0.5 * (src.fourier_transform(f) + np.conj(src.fourier_transform(-f)))
        assert abs(num - exact) <= 1e-4 * scale, f"f={f}: {num} vs {exact}"


def test_fourier_transform_peak_value():
    # |F(fcen)| = width * sqrt(2*pi), width = 1/fwidth
    src = GaussianSource(frequency=1.0, fwidth=0.2)
    assert abs(src.fourier_transform(1.0)) == pytest.approx(5.0 * math.sqrt(2 * math.pi))


# --- Susceptibility / DrudeSusceptibility ---

def test_susceptibility_hierarchy_and_fields():
    lo = LorentzianSusceptibility(frequency=1.1, gamma=1e-5, sigma=0.5)
    dr = DrudeSusceptibility(frequency=0.4, gamma=0.03, sigma=0.6)
    assert isinstance(lo, Susceptibility) and isinstance(dr, Susceptibility)
    assert (lo.frequency, lo.gamma) == (1.1, 1e-5)
    assert lo.sigma_diag == Vector3(0.5, 0.5, 0.5)   # meep: sigma -> sigma_diag
    assert tuple(lo.to_pole()) == (1.1, 1e-5, 0.5)
    # Drude -> engine 4-tuple pole (fn=0, gamma, sigma, fc=fp): no restoring
    # force, coupling at the plasma frequency (see coeffs._pole_recurrence)
    assert tuple(dr.to_pole()) == (0.0, 0.03, 0.6, 0.4)
    # engine takes scalar sigma only: anisotropic sigma raises at to_pole
    with pytest.raises(NotImplementedError):
        LorentzianSusceptibility(frequency=1.0, sigma_diag=Vector3(1, 0, 0)).to_pole()
    with pytest.raises(NotImplementedError):
        LorentzianSusceptibility(frequency=1.0, sigma=1.0,
                                 sigma_offdiag=Vector3(0.1, 0, 0)).to_pole()


def test_analytic_eps_honors_drude_pole():
    from meep_torch.core.materials import analytic_eps
    f, fp, gamma, sigma = 0.3, 0.4, 0.03, 0.6
    got = analytic_eps(1.0, [DrudeSusceptibility(fp, gamma, sigma=sigma).to_pole()], f)
    want = 1.0 - sigma * fp ** 2 / (f ** 2 + 1j * gamma * f)
    assert got == pytest.approx(want)


def _drude_R_analytic(f0, d, gamma, sigma, fp, eps_inf=1.0):
    """1D slab-in-vacuum power reflectance at normal incidence."""
    eps = eps_inf - sigma * fp ** 2 / (f0 ** 2 + 1j * gamma * f0)
    n = np.sqrt(eps)
    r01 = (1.0 - n) / (1.0 + n)
    e2 = np.exp(2j * 2 * math.pi * f0 * n * d)
    r = r01 * (1.0 - e2) / (1.0 - r01 ** 2 * e2)
    return float(abs(r) ** 2)


def test_drude_slab_reflectance_vs_transfer_matrix():
    sz, res, f0, d_slab = 12.0, 40, 0.3, 2.0
    fp, gamma, sigma = 0.4, 0.03, 0.6

    def run(geom):
        sim = mp.Simulation(
            cell_size=mp.Vector3(0, 0, sz), resolution=res, dimensions=1,
            geometry=geom,
            sources=[mp.Source(mp.GaussianSource(frequency=f0, fwidth=0.1),
                               component=mp.Ex, center=mp.Vector3(z=-0.5 * sz + 1.5))],
            boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
        name = sim.add_flux(f0, 0.0, 1, mp.FluxRegion(center=mp.Vector3(z=-0.5 * sz + 3.0)))
        sim.run(n_steps=int(round(150 / sim.grid.dt)))
        return np.array(sim.get_fluxes(name))

    drude = DrudeSusceptibility(frequency=fp, gamma=gamma, sigma=sigma)
    empty = run([])
    slab = run([mp.Block(size=mp.Vector3(mp.inf, mp.inf, d_slab), center=mp.Vector3(z=1.0),
                         material=mp.Medium(epsilon=1.0, E_susceptibilities=[drude]))])
    R_sim = float((-(slab - empty) / empty)[0])
    R_an = _drude_R_analytic(f0, d_slab, gamma, sigma, fp)
    rel = abs(R_sim - R_an) / R_an
    print(f"Drude slab (shim): R_sim={R_sim:.5f} R_analytic={R_an:.5f} rel={rel:.4f}")
    assert rel < 0.05, f"R_sim={R_sim:.5f} vs R_analytic={R_an:.5f} (rel {rel:.4f})"


# --- GaussianBeamSource ---

def test_gaussian_beam_profile_shape():
    w0 = 0.8
    src = GaussianBeamSource(mp.GaussianSource(frequency=1.0, fwidth=0.3),
                             center=Vector3(-4.0, 0.0), size=Vector3(0, 14.0),
                             component=mp.Ez, beam_x0=Vector3(5.0, 0.0),
                             beam_kdir=Vector3(1.0, 0.0), beam_w0=w0)
    assert isinstance(src, Source)          # rides the existing Source plumbing
    for y in (0.0, 0.4, 1.0, 2.5):
        # amp_func gets points relative to the source center (meep convention)
        assert src.amp_func(Vector3(0.0, y)) == pytest.approx(math.exp(-(y / w0) ** 2))
        assert src.amp_func(Vector3(0.0, -y)) == pytest.approx(src.amp_func(Vector3(0.0, y)))
    with pytest.raises(NotImplementedError):
        GaussianBeamSource(mp.GaussianSource(frequency=1.0, fwidth=0.3),
                           volume=object(), component=mp.Ez, beam_w0=w0)


def test_gaussian_beam_e0_scales_amplitude():
    src = GaussianBeamSource(mp.GaussianSource(frequency=1.0, fwidth=0.3),
                             center=Vector3(), size=Vector3(0, 4.0),
                             component=mp.Ez, beam_w0=1.0, beam_E0=Vector3(0, 0, 2.0))
    assert src.amp_func(Vector3(0.0, 0.0)) == pytest.approx(2.0)


def test_gaussian_beam_runs_through_simulation():
    sim = mp.Simulation(
        cell_size=mp.Vector3(6.0, 6.0, 0), resolution=10,
        sources=[GaussianBeamSource(mp.GaussianSource(frequency=1.0, fwidth=0.3),
                                    center=Vector3(-2.0, 0.0), size=Vector3(0, 6.0),
                                    component=mp.Ez, beam_w0=0.8)],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    sim.run(n_steps=300)
    ez = np.asarray(sim.get_array(component=mp.Ez))
    assert np.abs(ez).max() > 0.0
