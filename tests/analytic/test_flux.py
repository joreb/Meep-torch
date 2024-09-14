# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

FREQS = [0.55, 0.65, 0.75]


def quasi_1d_sim():
    g = Grid(nx=1200, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=GaussianPulse(0.65, 0.2))
    return g, m, src


def test_flux_conserved_between_two_lines():
    g, m, src = quasi_1d_sim()
    monA = FluxMonitorTM(g, FREQS, x_normal=True, idx=400, lo=0, hi=8)
    monB = FluxMonitorTM(g, FREQS, x_normal=True, idx=900, lo=0, hi=8)
    s = init_tm_state(g, y_pmc=True)
    # 8000 steps (t=80) ensures the full Gaussian pulse (t_end=50) passes both
    # monitors before the DFT integral is evaluated. monB is 16 units from the
    # source; its trailing edge arrives at t≈66, which is beyond the brief's
    # 5000-step horizon (t=50).
    _, (accA, accB) = run_tm(s, m, 8000, source=src, monitors=[monA, monB])
    fA, fB = monA.flux(accA, g.dx), monB.flux(accB, g.dx)
    assert torch.all(fA > 0)                      # +x flux positive to the right of source
    assert torch.allclose(fA, fB, rtol=0.01)      # vacuum: no loss between lines


def test_flux_sign_flips_left_of_source():
    g, m, src = quasi_1d_sim()
    mon = FluxMonitorTM(g, FREQS, x_normal=True, idx=60, lo=0, hi=8)
    s = init_tm_state(g, y_pmc=True)
    _, (acc,) = run_tm(s, m, 5000, source=src, monitors=[mon])
    assert torch.all(mon.flux(acc, g.dx) < 0)     # energy flows -x left of source
