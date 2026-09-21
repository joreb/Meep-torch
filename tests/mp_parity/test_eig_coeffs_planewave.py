# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Simulation.get_eigenmode_coefficients on a flux monitor: the 0th
diffraction order of a Bloch-periodic plane with uniform transverse eps
(binary_grating_phasemap / metasurface_lens shape).

meep's normalization gate: |alpha_+|^2 - |alpha_-|^2 == the monitor flux.
Verified on real meep 1.34 for the binary_grating_phasemap grating at
resolution 40 (nfreq=3, gp=0.35, gh=1.2, gdc=0.5); mode transmission
|alpha_+|^2 / input_flux, meep vs meep-torch:

    ODD_Z+EVEN_Y  0.9855 / 0.9829,  0.8817 / 0.8782,  0.9331 / 0.9354
    EVEN_Z+ODD_Y  0.9688 / 0.9656,  0.9630 / 0.9692,  0.9843 / 0.9678

(the residual is subpixel eps smoothing, as with bend-flux). meep's band 1
is the m=0 planewave for BOTH parities -- kdom came back (f, 0, 0) in each
case -- which is why the Y parity bit only picks the polarization pairing
here.
"""
import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.mp.run_control import EigCoeffsResult

FCEN, DF, NF = 1.0, 0.4, 3
GP = 0.6                       # cell period in y (Bloch, k_point = 0)


def _sim(oddz=True, k_point=mp.Vector3(), geometry=()):
    sx = 6.0
    comp = mp.Ez if oddz else mp.Hz
    return mp.Simulation(
        cell_size=mp.Vector3(sx, GP), resolution=20,
        boundary_layers=[mp.PML(1.0, direction=mp.X)],
        geometry=list(geometry), k_point=k_point,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=comp,
                           center=mp.Vector3(-0.5 * sx + 1.5),
                           size=mp.Vector3(0, GP))])


def _run(oddz=True, **kw):
    sim = _sim(oddz, **kw)
    mon = sim.add_flux(FCEN, DF, NF,
                       mp.FluxRegion(center=mp.Vector3(1.5), size=mp.Vector3(0, GP)))
    sim.run(until_after_sources=40)
    return sim, mon


@pytest.mark.parametrize("oddz", [True, False])
def test_planewave_alpha_matches_monitor_flux(oddz):
    """meep's normalization: |a+|^2 - |a-|^2 is the monitor flux, and a
    source launched into vacuum is (almost) purely forward."""
    sim, mon = _run(oddz)
    par = mp.ODD_Z + mp.EVEN_Y if oddz else mp.EVEN_Z + mp.ODD_Y
    res = sim.get_eigenmode_coefficients(mon, [1], eig_parity=par)
    assert isinstance(res, EigCoeffsResult)
    assert res.alpha.shape == (1, NF, 2)
    flux = np.array(mp.get_fluxes(mon))
    a_p, a_m = res.alpha[0, :, 0], res.alpha[0, :, 1]
    net = np.abs(a_p) ** 2 - np.abs(a_m) ** 2
    assert np.allclose(net, flux, rtol=1e-9), f"{net} vs {flux}"
    assert (np.abs(a_m) ** 2 < 1e-2 * np.abs(a_p) ** 2).all()


def test_slab_transmission_matches_fresnel():
    """A quarter-cell-thick eps=4 slab: |a+|^2 relative to the empty run is
    the analytic Fabry-Perot transmittance of a lossless slab."""
    n, d = 2.0, 0.5
    _, mon0 = _run()
    ref = np.abs(_run()[0].get_eigenmode_coefficients(mon0, [1],
                                                      eig_parity=mp.ODD_Z).alpha[0, :, 0]) ** 2
    slab = [mp.Block(size=mp.Vector3(d, mp.inf, mp.inf), center=mp.Vector3(),
                     material=mp.Medium(index=n))]
    sim, mon = _run(geometry=slab)
    got = np.abs(sim.get_eigenmode_coefficients(mon, [1],
                                                eig_parity=mp.ODD_Z).alpha[0, :, 0]) ** 2
    freqs = np.array(mp.get_flux_freqs(mon))
    r = (1 - n) / (1 + n)
    ph = 2 * np.pi * freqs * n * d
    t = (1 - r ** 2) / (1 - r ** 2 * np.exp(2j * ph)) * np.exp(1j * ph)
    # atol is resolution-limited (res=20, no subpixel smoothing): the gate is
    # the normalization/sign convention, not the last percent of the FDTD.
    assert np.allclose(got / ref, np.abs(t) ** 2, atol=0.08), \
        f"{got / ref} vs {np.abs(t) ** 2}"


def test_named_raises():
    sim, mon = _run()
    with pytest.raises(TypeError, match="list or DiffractedPlanewave"):
        sim.get_eigenmode_coefficients(mon, 1)
    with pytest.raises(NotImplementedError, match="DiffractedPlanewave"):
        sim.get_eigenmode_coefficients(
            mon, mp.DiffractedPlanewave((0, 1, 0), mp.Vector3(0, 1, 0), 0, 1))
    with pytest.raises(NotImplementedError, match="kpoint_func"):
        sim.get_eigenmode_coefficients(mon, [1], kpoint_func=lambda *a: None)
    with pytest.raises(NotImplementedError, match="direction"):
        sim.get_eigenmode_coefficients(mon, [1], direction=mp.NO_DIRECTION)
    with pytest.raises(NotImplementedError, match="eig_vol"):
        sim.get_eigenmode_coefficients(mon, [1], eig_vol=mp.Volume(size=mp.Vector3(0, GP)))
    # bands beyond 1 USED to raise "only the 0th/specular order". The
    # projection for order m was in ports/orders.py the whole time; only the
    # band -> order map was missing, and it is a sort on the longitudinal
    # wavenumber. Higher bands now return coefficients.
    a = sim.get_eigenmode_coefficients(mon, [1, 2, 3]).alpha
    assert a.shape[0] == 3
    # at normal incidence with a uniform line, all the power is specular:
    # bands 2 and 3 are the +/-1 orders and carry nothing here
    assert abs(a[0, 0, 0]) > 0.0
    assert abs(a[1, 0, 0]) < 1e-6 * abs(a[0, 0, 0])
    assert abs(a[2, 0, 0]) < 1e-6 * abs(a[0, 0, 0])
    with pytest.raises(ValueError, match="eig_parity implies"):
        sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.EVEN_Z)


def test_oblique_now_works_and_a_structured_line_still_raises():
    # oblique incidence used to raise: the planewave basis was built at
    # k_transverse = 0. It is now built at k_y = 2 pi k_point.y, which is what
    # lifts the +/-m degeneracy in the band -> order map.
    sim, mon = _run(k_point=mp.Vector3(0, 0.2))
    a = sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.ODD_Z).alpha
    assert a.shape == (1, a.shape[1], 2) and np.isfinite(a).all()
    # a guided monitor (structured transverse eps) has no flux-monitor path
    wg = [mp.Block(size=mp.Vector3(mp.inf, 0.2, mp.inf), material=mp.Medium(epsilon=12))]
    sim, mon = _run(geometry=wg)
    with pytest.raises(NotImplementedError, match="not uniform"):
        sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.ODD_Z)


def test_meep_normalisation_identity():
    """|a+|^2 - |a-|^2 == the monitor flux, which is meep's whole contract.

    `planewave_coeffs` stated this in a docstring and nothing asserted it, so
    when the DFT moved to meep's unitary convention the flux lost its 1/2 and
    the amplitude scale did not -- the identity read 0.5000 for an unknown
    number of runs, halving every mode coefficient, and the shape-and-
    finiteness tests above stayed green throughout.
    """
    sim, mon = _run()
    a = sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.ODD_Z).alpha
    flux = np.asarray(mp.get_fluxes(mon))
    ident = np.abs(a[0, :, 0]) ** 2 - np.abs(a[0, :, 1]) ** 2
    live = np.abs(flux) > 1e-20 * np.abs(flux).max()
    assert live.any()
    np.testing.assert_allclose(ident[live], flux[live], rtol=2e-3)
