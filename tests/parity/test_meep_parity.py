# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import pytest
import torch

mp = pytest.importorskip("meep")
# the repo-root reference clone `meep/` (gitignored) shadows pymeep as an
# empty namespace package when cwd is on sys.path -- only the real
# pymeep (conda oracle env) has Vector3
if not hasattr(mp, "Vector3"):
    pytest.skip("real pymeep not installed (namespace-dir shadow)",
                allow_module_level=True)

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

# Both engines: quasi-1D vacuum, Gaussian line source, two flux planes.
# Compare the RATIO flux_far/flux_near (normalization-independent).
FREQS = [0.55, 0.65, 0.75]


def torch_flux_ratio() -> torch.Tensor:
    g = Grid(nx=1200, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=GaussianPulse(0.65, 0.2))
    monA = FluxMonitorTM(g, FREQS, x_normal=True, idx=400, lo=0, hi=8)
    monB = FluxMonitorTM(g, FREQS, x_normal=True, idx=900, lo=0, hi=8)
    s = init_tm_state(g, y_pmc=True)
    _, (a, b) = run_tm(s, m, 5000, source=src, monitors=[monA, monB])
    return monB.flux(b, g.dx) / monA.flux(a, g.dx)


def meep_flux_ratio():
    cell = mp.Vector3(24, 0, 0)
    sim = mp.Simulation(
        cell_size=cell,
        resolution=50,
        boundary_layers=[mp.PML(0.8, direction=mp.X)],
        sources=[mp.Source(mp.GaussianSource(0.65, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3(-10, 0))],
        dimensions=1,
    )
    fA = sim.add_flux(0.65, 0.2, len(FREQS),
                      mp.FluxRegion(center=mp.Vector3(-4, 0)))
    fB = sim.add_flux(0.65, 0.2, len(FREQS),
                      mp.FluxRegion(center=mp.Vector3(6, 0)))
    sim.run(until_after_sources=100)
    import numpy as np
    return np.array(mp.get_fluxes(fB)) / np.array(mp.get_fluxes(fA))


def test_vacuum_flux_ratio_parity():
    ours = torch_flux_ratio()
    theirs = torch.tensor(meep_flux_ratio(), dtype=torch.float64)
    assert torch.allclose(ours, theirs, rtol=0.01)
