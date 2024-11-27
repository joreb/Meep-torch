# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W6 T1-T2: point LDOS from source-field overlap."""
import math

import numpy as np

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.ldos import ldos_spectrum
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

FCEN, DF = 1.0, 0.4
BAND = np.linspace(0.85, 1.15, 7)


def _vacuum_ldos(ix, iy):
    grid = Grid(nx=200, ny=200, dx=0.05)
    coeffs = build_tm_coeffs(grid, eps_inf=grid.zeros(200, 200) + 1.0, npml=20)
    pulse = GaussianPulse(FCEN, DF)
    src = LineSource(grid, ix=ix, j0=iy, j1=iy + 1, profile=pulse)
    n_steps = int(round(100.0 / grid.dt))
    _, (e,) = run_tm(init_tm_state(grid), coeffs, n_steps,
                     source=src, monitors=[PointProbe(ix, iy)])
    j = np.array([pulse.amplitude((t + 0.5) * grid.dt) for t in range(n_steps)])
    return ldos_spectrum(e.detach().numpy(), j, grid.dt, BAND)


def test_vacuum_ldos_linear_in_f_and_translation_invariant():
    l1 = _vacuum_ldos(100, 100)
    assert (l1 > 0).all()
    # 2D vacuum LDOS is proportional to f: ldos/f band-flat <= 3%
    r = l1 / BAND
    assert (r.max() - r.min()) / r.mean() <= 0.03, f"ldos/f spread {r}"
    # translation invariance <= 2%
    l2 = _vacuum_ldos(80, 120)
    assert np.abs(l2 / l1 - 1.0).max() <= 0.02, f"translation dev {l2 / l1}"
