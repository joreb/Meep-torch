# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W6 T3-T4: Maxwell-stress-tensor force — radiation pressure on a mirror."""
import math

import numpy as np

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.forces import mst_force_tm
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor, PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

FCEN = 1.0
FREQS = [0.95, 1.0, 1.05]


def _run(with_mirror):
    grid = Grid(nx=600, ny=8, dx=0.05)
    kwargs = {}
    if with_mirror:
        sd = grid.zeros(600, 8)
        sd[300:320, :] = 1e4                      # mirror x in [15, 16]
        kwargs["sigma_d"] = sd
    coeffs = build_tm_coeffs(grid, eps_inf=grid.zeros(600, 8) + 1.0,
                             npml=30, y_pmc=True, **kwargs)
    src = LineSource(grid, ix=60, j0=0, j1=8, profile=GaussianPulse(FCEN, 0.2))
    mons = [DFTFieldsMonitor(grid, FREQS, c) for c in ("Ez", "Hx", "Hy")]
    mons.append(PointProbe(280, 4))               # left of the mirror (x=14)
    n_steps = int(round(80.0 / grid.dt))
    _, accs = run_tm(init_tm_state(grid, y_pmc=True), coeffs, n_steps,
                     source=src, monitors=mons)
    planes = {c: (a[:, 0] + 1j * a[:, 1]).detach().cpu().numpy()
              for c, a in zip(("Ez", "Hx", "Hy"), accs[:3])}
    return planes, accs[3].detach().cpu().numpy(), grid


def test_radiation_pressure():
    import json
    import pathlib
    with open(pathlib.Path(__file__).parent / "data" / "radiation_force.json") as fh:
        oracle = json.load(fh)

    planes_m, sr_m, grid = _run(True)
    planes_0, sr_0, _ = _run(False)
    fx, fy = mst_force_tm(planes_m, grid, ix0=280, ix1=340, iy0=2, iy1=6)

    t = np.arange(len(sr_m)) * grid.dt
    ly = 4 * grid.dx                              # contour height (iy0..iy1)
    for k, f in enumerate(FREQS):
        kern = np.exp(-2j * math.pi * f * t)
        inc = np.trapezoid(sr_0 * kern, t)
        refl = np.trapezoid((sr_m - sr_0) * kern, t)
        R = abs(refl / inc) ** 2
        # incident spectral Poynting density at the left face from the
        # no-mirror planes: <S_x> = -1/2 Re[Ez conj(Hy)] (TM convention)
        ez = planes_0["Ez"][k][280, 4]
        hy = 0.5 * (planes_0["Hy"][k][279, 4] + planes_0["Hy"][k][280, 4])
        s_inc = -0.5 * np.real(ez * np.conj(hy))
        want = (1 + R) * s_inc * ly
        got = fx[k]
        assert abs(got - want) / abs(want) < 0.05, \
            f"f={f}: Fx={got:.4e} vs (1+R) S_inc Ly={want:.4e} (R={R:.3f})"
        assert abs(fy[k]) < 0.02 * abs(want)      # symmetric in y
        # oracle half of gate (c): dimensionless pressure ratio vs meep's
        ratio_ours = got / want
        ratio_meep = oracle["pressure_ratio"][k]
        assert abs(ratio_ours - ratio_meep) <= 0.10, \
            f"f={f}: ratio {ratio_ours:.3f} vs meep {ratio_meep:.3f}"