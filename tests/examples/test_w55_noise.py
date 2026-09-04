# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5.5 T4: RandomSource — noise-driven field spectrum matches the
deterministic transfer function (flat white drive) within +-3 dB."""
import math

import numpy as np

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource, RandomSource
from meep_torch.core.stepper_tm import init_tm_state

FCEN, DF = 1.0, 0.4
BAND = np.linspace(0.85, 1.15, 7)


def _grid_coeffs():
    grid = Grid(nx=200, ny=160, dx=0.05)
    return grid, build_tm_coeffs(grid, eps_inf=grid.zeros(200, 160) + 1.0, npml=20)


def _spectrum(series, dt, freqs):
    t = np.arange(len(series)) * dt
    return np.array([abs(np.trapezoid(series * np.exp(-2j * math.pi * f * t), t))
                     for f in freqs])


def test_noise_spectrum_matches_transfer():
    grid, coeffs = _grid_coeffs()
    probe = PointProbe(140, 80)
    n_steps = int(round(120.0 / grid.dt))

    # deterministic transfer |G(f)| = |E_pulse(f)| / |src_pulse(f)|
    pulse = GaussianPulse(FCEN, DF)
    src = LineSource(grid, ix=60, j0=78, j1=82, profile=pulse)
    _, (s_det,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                         source=src, monitors=[probe])
    t = np.arange(n_steps) * grid.dt
    src_amp = np.array([pulse.amplitude((i + 0.5) * grid.dt) for i in range(n_steps)])
    G = _spectrum(s_det.detach().cpu().numpy(), grid.dt, BAND) / _spectrum(src_amp, grid.dt, BAND)

    # noise-driven: average |E(f)|^2 over seeds; white drive -> proportional
    # to |G(f)|^2, so the ratio must be band-flat
    acc = np.zeros(len(BAND))
    n_seeds = 8
    for seed in range(n_seeds):
        rnd = RandomSource(grid, ix=60, j0=78, j1=82, seed=seed)
        _, (s,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                         source=rnd, monitors=[probe])
        acc += _spectrum(s.detach().cpu().numpy(), grid.dt, BAND) ** 2
    ratio = (acc / n_seeds) / G ** 2
    ratio_db = 10 * np.log10(ratio / ratio.mean())
    assert np.abs(ratio_db).max() <= 3.0, f"flatness {ratio_db} dB"
