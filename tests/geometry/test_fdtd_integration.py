# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import numpy as np
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

FREQS = [0.60, 0.65, 0.70]


def test_rasterized_waveguide_flux_conserved():
    g = Grid(nx=1200, ny=120, dx=0.02)
    wg = Polygon(np.array(
        [[-1.0, 1.0], [25.0, 1.0], [25.0, 1.4], [-1.0, 1.4]], dtype=float))  # 0.4um-wide guide
    rm = rasterize_scene([(wg, 6.0)], g, background=2.1025)  # neff-ish contrast, nondispersive
    m = build_tm_coeffs(g, eps_inf=rm.eps_tm, npml=40)
    src = LineSource(g, ix=120, j0=40, j1=80, profile=GaussianPulse(0.65, 0.2))
    monA = FluxMonitorTM(g, FREQS, x_normal=True, idx=400, lo=20, hi=100)
    monB = FluxMonitorTM(g, FREQS, x_normal=True, idx=900, lo=20, hi=100)
    s = init_tm_state(g)
    _, (a, b) = run_tm(s, m, 9000, source=src, monitors=[monA, monB])
    fA, fB = monA.flux(a, g.dx), monB.flux(b, g.dx)
    assert torch.all(fA > 0)
    assert torch.allclose(fB / fA, torch.ones(3, dtype=torch.float64, device=g.device), atol=0.01)
