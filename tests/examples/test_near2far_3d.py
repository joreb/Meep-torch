# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""3D near-to-far gates (task port_3d-near2far).

(a) Green-kernel self-test: farfield_3d over an analytic point-dipole's
    fields on a closed box reproduces the dipole far field (mirror of
    test_w4_green2d for the 3D scalar Green e^{-ikR}/4pi R).
(b) FDTD antenna radiation, mirror of test_w4_antenna_radiation in 3D:
    a z dipole in vacuum radiates sin^2(theta) — normalized |E|^2 pattern
    on the far circle vs analytic, RMS < 2%.
"""
import math

import numpy as np
import torch

from meep_torch.core.monitors_3d import PointSource3D
from meep_torch.core.near2far_3d import (Near2FarBox3D, analytic_box_faces,
                                         dipole_fields_3d, farfield_3d)
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean

FREQ = 1.0
K = 2 * math.pi * FREQ


def test_green3d_selftest():
    """Surface equivalence must reproduce the analytic dipole it encloses."""
    src = (0.13, -0.07, 0.21)                 # off-center: asymmetric faces
    faces = analytic_box_faces(K, (1.0, 41), src, comp=2)
    th = np.linspace(0.05, math.pi - 0.05, 24)
    ph = np.linspace(-math.pi, math.pi, 12, endpoint=False)
    thg, phg = np.meshgrid(th, ph, indexing="ij")
    obs = np.stack([np.sin(thg) * np.cos(phg), np.sin(thg) * np.sin(phg),
                    np.cos(thg)], axis=-1).reshape(-1, 3) * 1000.0
    E, H = farfield_3d(faces, [K], obs)
    Ea, Ha = dipole_fields_3d(K, obs, src_pt=src, comp=2)
    relE = np.linalg.norm(E[0] - Ea) / np.linalg.norm(Ea)
    relH = np.linalg.norm(H[0] - Ha) / np.linalg.norm(Ha)
    print(f"green3d selftest: rel E {relE:.2e}  rel H {relH:.2e}")
    assert relE < 2e-3 and relH < 2e-3


def test_green3d_selftest_medium():
    """Same box, homogeneous eps = 2.25 (n = 1.5) outside and in.

    The kernel takes the VACUUM 2 pi f and the medium separately: the
    wavenumber is n omega, the electric-current term carries omega mu, and
    H = rhat x E / Z. The analytic dipole in the medium is the vacuum formula
    at k = n omega with E / eps and H / n. Before the medium reached the
    kernel this ran at the vacuum k and the six faces stopped summing
    coherently -- the 2D twin of this defect gave an antenna tutorial
    rel err 0.645 against meep's 0.005."""
    eps = 2.25
    n = math.sqrt(eps)
    src = (0.13, -0.07, 0.21)
    # 61 points, not 41: the box is n times larger in wavelengths, and the
    # midpoint rule's error tracks samples per wavelength (41 -> 61 halves
    # it here exactly as it does in vacuum), so this is the vacuum gate's
    # quadrature at the medium's k, not a looser kernel.
    faces = analytic_box_faces(K * n, (1.0, 61), src, comp=2)
    for f in faces:                          # vacuum formula -> medium fields
        f.E /= eps
        f.H /= n
    th = np.linspace(0.05, math.pi - 0.05, 24)
    ph = np.linspace(-math.pi, math.pi, 12, endpoint=False)
    thg, phg = np.meshgrid(th, ph, indexing="ij")
    obs = np.stack([np.sin(thg) * np.cos(phg), np.sin(thg) * np.sin(phg),
                    np.cos(thg)], axis=-1).reshape(-1, 3) * 1000.0
    E, H = farfield_3d(faces, [K], obs, eps=eps)
    Ea, Ha = dipole_fields_3d(K * n, obs, src_pt=src, comp=2, eps=eps)
    relE = np.linalg.norm(E[0] - Ea) / np.linalg.norm(Ea)
    relH = np.linalg.norm(H[0] - Ha) / np.linalg.norm(Ha)
    print(f"green3d selftest n=1.5: rel E {relE:.2e}  rel H {relH:.2e}")
    assert relE < 2e-3 and relH < 2e-3
    # and the vacuum kernel on the same faces is NOT a small error
    Ev, _ = farfield_3d(faces, [K], obs)
    assert np.linalg.norm(Ev[0] - Ea) / np.linalg.norm(Ea) > 0.3


def test_green3d_far_field_is_box_independent():
    """Surface equivalence does not care where the box sits: the same
    dipole enclosed by a box of half-size 0.7 and one of 1.3 must give the
    same far field, in vacuum and in a medium. A kernel with the wrong k or
    impedance passes the self-test at ONE box only by accident of that box;
    this is the check nothing ran before."""
    src = (0.13, -0.07, 0.21)
    th = np.linspace(0.05, math.pi - 0.05, 16)
    ph = np.linspace(-math.pi, math.pi, 8, endpoint=False)
    thg, phg = np.meshgrid(th, ph, indexing="ij")
    obs = np.stack([np.sin(thg) * np.cos(phg), np.sin(thg) * np.sin(phg),
                    np.cos(thg)], axis=-1).reshape(-1, 3) * 1000.0
    for eps in (1.0, 2.25):
        n = math.sqrt(eps)
        outs = []
        for half in (0.7, 1.3):
            npts = int(round(61 * half))          # same samples per wavelength
            faces = analytic_box_faces(K * n, (half, npts), src, comp=2)
            for f in faces:
                f.E /= eps
                f.H /= n
            E, _ = farfield_3d(faces, [K], obs, eps=eps)
            outs.append(E[0])
        rel = np.linalg.norm(outs[0] - outs[1]) / np.linalg.norm(outs[1])
        print(f"green3d box independence eps={eps}: rel {rel:.2e}")
        assert rel < 3e-3


def test_fdtd_dipole_pattern_3d():
    n = 64
    npml = 8
    dx = 0.05
    eps = torch.ones(n, n, n, dtype=torch.float64)
    c = build_lean(n, n, n, dx, eps, npml, dtype=torch.float64, device="cpu")
    ctr = n // 2
    b = npml + 6                                  # face distance from edge
    box = (b, n - b, b, n - b, b, n - b)
    mon = Near2FarBox3D(c.dt, [FREQ], box, dx, dtype=torch.float64)
    src = PointSource3D(c.dt, "Ez", ctr, ctr, ctr, GaussianPulse(FREQ, 0.5))
    t_end = 50.0
    n_steps = int(round(t_end / c.dt))
    _, accs = run_3d_lean(c, n_steps, source=src, monitors=[mon])
    faces = mon.faces(accs[0])

    th = np.linspace(0.05, math.pi - 0.05, 36)
    obs = np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1) * 1000.0
    obs = obs + np.array([ctr, ctr, ctr]) * dx     # transform frame
    E, _H = farfield_3d(faces, [K], obs)
    p = np.sum(np.abs(E[0]) ** 2, axis=1)
    got = p / p.max()
    ref = np.sin(th) ** 2
    ref = ref / ref.max()
    rms = float(np.sqrt(np.mean((got - ref) ** 2)))
    print(f"3D dipole pattern: rms vs sin^2 {rms:.4f}")
    assert rms <= 0.02, f"normalized rms vs analytic dipole {rms:.4f}"


def test_get_farfield_3d_phase_advances_outgoing():
    """meep's get_farfield is e^{-i w t}, so an outgoing wave's phase grows
    along a ray as e^{+ikR}: the phase of get_farfield(R2)/get_farfield(R1)
    is +k (R2 - R1). Meep gives +44.9 deg per 0.25 at k = pi; the shim once
    returned the conjugate (-k dR), because the 3D kernel is e^{-ikR} and its
    output was never conjugated back. Checked in the shim, not in the core
    kernel, which is convention-blind to a global conjugate."""
    import meep_torch.mp as mp
    fcen = 0.5
    k = 2 * math.pi * fcen
    sim = mp.Simulation(cell_size=mp.Vector3(8, 8, 8), resolution=10, dimensions=3,
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    regs = []
    for ax in range(3):
        for s in (+1, -1):
            c = [0.0, 0.0, 0.0]
            c[ax] = s * 1.0
            sz = [2.0 if o != ax else 0.0 for o in range(3)]
            regs.append(mp.Near2FarRegion(center=mp.Vector3(*c), size=mp.Vector3(*sz),
                                          weight=s))
    n2f = sim.add_near2far(fcen, 0, 1, *regs)
    sim.run(until=80)
    ez = np.array([complex(sim.get_farfield(n2f, mp.Vector3(r, 0, 0))[2])
                   for r in (6.0, 6.25, 6.5, 6.75)])
    dph = np.angle(ez[1:] / ez[:-1])
    assert np.all(np.abs(dph - k * 0.25) < 0.2), \
        f"phase steps {np.degrees(dph)} deg, expect +{math.degrees(k * 0.25):.1f}"
