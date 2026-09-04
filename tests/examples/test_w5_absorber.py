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


def test_absorber_1d_vs_oracle():
    """Gate (b): 1D conductive-background ringdown, Absorber + PML residuals
    within 10x of the meep-oracle floors (absorber_1d.json)."""
    import json
    import pathlib

    import numpy as np

    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.grid import Grid
    from meep_torch.core.monitors import PointProbe
    from meep_torch.core.pml import absorber_sigma
    from meep_torch.core.run import run_tm
    from meep_torch.core.sources import GaussianPulse, LineSource
    from meep_torch.core.stepper_tm import init_tm_state

    with open(pathlib.Path(__file__).parent / "data" / "absorber_1d.json") as f:
        ref = json.load(f)
    grid = Grid(nx=400, ny=8, dx=1.0 / 40)
    nlayer = 40

    def ringdown(use_absorber):
        sd = grid.zeros(400, 8) + ref["sigma_d"]
        kwargs = dict(sigma_d=sd, y_pmc=True)
        npml = nlayer
        if use_absorber:
            ramp = absorber_sigma(400, nlayer, grid.dx,
                                  dtype=grid.dtype, device=grid.device)[:, None]
            kwargs["sigma_d"] = sd + ramp
            kwargs["sigma_b"] = ramp.expand(400, 8).clone()
            npml = 0
        coeffs = build_tm_coeffs(grid, eps_inf=grid.zeros(400, 8) + 1.0,
                                 npml=npml, **kwargs)
        src = LineSource(grid, ix=200, j0=0, j1=8,
                         profile=GaussianPulse(ref["fcen"], ref["df"]))
        _, (s,) = run_tm(init_tm_state(grid, y_pmc=True), coeffs,
                         int(round(240.0 / grid.dt)), source=src,
                         monitors=[PointProbe(200, 4)])
        a = np.abs(s.detach().cpu().numpy())
        return float(a[2 * len(a) // 3:].max() / a.max())

    for mode, key in ((True, "absorber"), (False, "pml")):
        got = ringdown(mode)
        want = ref[key]["residual"] / ref[key]["peak"]
        assert got <= 5e-3, f"{key} residual {got:.2e}"
        assert got <= 10 * want, f"{key}: {got:.2e} vs oracle floor {want:.2e}"


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
