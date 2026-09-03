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
