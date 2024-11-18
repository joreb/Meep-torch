# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5 T3: mp.Absorber boundary + Medium conductivity plumbing."""
import torch

import meep_torch.mp as mp

FCEN = 1.0


def _run_until(t_end):
    sim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.Absorber(1.0)], default_material=mp.Medium(epsilon=1.0))
    sim.run(until=t_end)
    return float(sim._state.Ez.abs().max())


def test_absorber_residual():
    # shim run() re-inits state, so replay the same deterministic sim twice:
    # peak while the pulse is in flight, residual long after it exited
    peak = _run_until(8)
    residual = _run_until(60)
    assert residual / peak <= 5e-3, f"absorber residual {residual / peak:.2e}"


def test_medium_conductivity_map():
    # a D_conductivity block damps the field passing through it
    sim = mp.Simulation(cell_size=mp.Vector3(10, 10, 0), resolution=20,
        geometry=[mp.Block(size=mp.Vector3(2, mp.inf, mp.inf), center=mp.Vector3(2.5, 0),
                           material=mp.Medium(epsilon=1.0, D_conductivity=5.0))],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.3), component=mp.Ez,
                           center=mp.Vector3(-2.5, 0), size=mp.Vector3(0, 4))],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    sim.run(until=30)
    ez = sim._state.Ez
    left = float(ez[20:60, 60:140].abs().max())       # source side
    right = float(ez[160:180, 60:140].abs().max())    # behind the block
    assert right < 0.15 * left, f"no attenuation: right/left={right / left:.3f}"
