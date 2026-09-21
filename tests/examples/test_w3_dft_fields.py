# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W3 T3: area DFT fields — DFTFieldsMonitor + mp shim add_dft_fields."""
import math

import numpy as np
import torch

import meep_torch.mp as mp
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor


class _FakeState:
    def __init__(self, Ez):
        self.Ez = Ez


def test_standing_wave_area_dft():
    # Ez(x,y,t) = P(x,y) cos(2 pi f0 tE): the area DFT at f0 recovers
    # P * T/2 / sqrt(2 pi) (Re part), and ~0 at a well-separated frequency.
    #
    # The 1/sqrt(2 pi) is meep's UNITARY DFT weight, `dt / sqrt(2 pi)` per step
    # (src/dft.cpp:225); the monitors carry it so that every absolute DFT and
    # flux value matches meep rather than being pi out. Without it this test
    # pins the plain convention and reads 0.399 = 1/sqrt(2 pi) of what it
    # expects.
    grid = Grid(nx=16, ny=12, dx=0.1)
    f0, f1 = 0.8, 1.6
    mon = DFTFieldsMonitor(grid, [f0, f1], component="Ez")
    P = torch.rand(16, 12, dtype=grid.dtype, device=grid.device)
    T = 100.0                                   # integer periods of f0 and f1
    n_steps = int(round(T / grid.dt))
    acc = torch.zeros_like(mon.sample(_FakeState(P), 0))
    for t in range(n_steps):
        tE = (t + 1.0) * grid.dt                # TM: Ez advanced to (t+1)*dt
        acc = acc + mon.sample(_FakeState(P * math.cos(2 * math.pi * f0 * tE)), t)
    re0, im0 = acc[0, 0], acc[0, 1]
    scale = T / 2.0 / math.sqrt(2.0 * math.pi)   # meep's unitary weight
    assert torch.allclose(re0, P * scale, rtol=1e-2, atol=1e-3 * scale), \
        f"max err {(re0 - P * scale).abs().max()}"
    assert im0.abs().max() < 1e-2 * scale
    assert acc[1].abs().max() < 1e-2 * scale    # off-resonance leakage


def test_shim_add_dft_fields():
    sim = mp.Simulation(cell_size=mp.Vector3(8, 8, 0), resolution=10,
        sources=[mp.Source(mp.ContinuousSource(frequency=0.5), component=mp.Ez,
                           center=mp.Vector3(0, 0), size=mp.Vector3(0, 1, 0))],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    dft = sim.add_dft_fields([mp.Ez], 0.5, 0.2, 3)
    sim.run(until=60)
    arr = sim.get_dft_array(dft, mp.Ez, 1)      # center frequency
    assert arr.shape == (80, 80)
    assert np.iscomplexobj(arr)
    # field concentrates near the source line at the drive frequency
    assert np.abs(arr).max() > 0
    peak = np.unravel_index(np.abs(arr).argmax(), arr.shape)
    assert abs(peak[0] - 40) < 12, f"peak at {peak}, expected near source x=40"
