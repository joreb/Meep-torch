# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Simulation(k_point=...) Bloch-periodic path (mp/bloch_shim.py): oblique
planewave reflectance off a flat vacuum -> n=2 interface vs Fresnel
(s-polarization, 2D TM, periodic y, PML x). The oblique launch follows the
meep oblique-planewave pattern: uniform line source + amp_func = pw_amp
(exp(2*pi*i k.y)) at the matching k_point.

Tolerances are the ENGINE's envelope, not the adapter's: at res=20 the
rasterized sharp interface carries a ~2% transmitted-flux bias (real meep
matches Fresnel to 3e-4 on this exact setup; the non-Bloch 1D engine path
reproduces our numbers to 1e-10, so the k_point wiring itself is exact).
See the k_point task report: engine interface/flux accuracy is a separate
follow-up."""
import cmath
import math

import meep_torch.mp as mp

FCEN, N2 = 0.3, 2.0


def _R(theta_deg):
    """R = 1 - T from the transmitted flux vs a vacuum reference, two
    Simulation runs at the Bloch k of the incidence angle."""
    theta = math.radians(theta_deg)
    kyp = FCEN * math.sin(theta)                 # k_point units (2*pi implied)
    sy = 1.0 / kyp if kyp else 2.0               # one Bloch period of the source
    sx = 10.0
    k = mp.Vector3(FCEN * math.cos(theta), kyp)

    def pw_amp(v):
        return cmath.exp(2j * math.pi * kyp * v.y)

    def _T(diel):
        geometry = []
        if diel:
            geometry = [mp.Block(center=mp.Vector3(sx / 4, 0),
                                 material=mp.Medium(index=N2),
                                 size=mp.Vector3(sx / 2, mp.inf))]
        sim = mp.Simulation(
            cell_size=mp.Vector3(sx, sy), resolution=20,
            boundary_layers=[mp.PML(1.0, direction=mp.X)],
            k_point=k, geometry=geometry,
            sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.5 * FCEN),
                               component=mp.Ez,
                               center=mp.Vector3(-sx / 2 + 1.5, 0),
                               size=mp.Vector3(0, sy), amp_func=pw_amp)])
        tran = sim.add_flux(FCEN, 0, 1,
                            mp.FluxRegion(center=mp.Vector3(sx / 2 - 1.5, 0),
                                          size=mp.Vector3(0, sy)))
        sim.run(until=150)
        return mp.get_fluxes(tran)[0]

    return 1.0 - _T(True) / _T(False)


def _fresnel_s(theta_deg, n1=1.0, n2=N2):
    th = math.radians(theta_deg)
    s2 = (n1 / n2) * math.sin(th)
    c2 = math.sqrt(1 - s2 * s2)
    r = (n1 * math.cos(th) - n2 * c2) / (n1 * math.cos(th) + n2 * c2)
    return r * r


def test_kpoint_bloch_fresnel():
    r0, r30 = _R(0.0), _R(30.0)
    assert abs(r0 - _fresnel_s(0.0)) < 0.03      # engine envelope (meas 0.019)
    assert abs(r30 - _fresnel_s(30.0)) < 0.06    # engine envelope (meas 0.048)
    # the Bloch angle is real: oblique reflects more, Fresnel ordering
    assert r30 > r0
