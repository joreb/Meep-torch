# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W5 T1-T2: D/B-conductivity + scalar mu — analytic slab attenuation gate."""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state

FCEN = 1.0


def test_sigma_zero_bit_parity():
    grid = Grid(nx=60, ny=40, dx=0.1)
    eps = grid.zeros(60, 40) + 1.0
    fast = build_tm_coeffs(grid, eps_inf=eps, npml=8)
    slow = build_tm_coeffs(grid, eps_inf=eps, npml=8,
                           sigma_d=grid.zeros(60, 40),
                           sigma_b=grid.zeros(60, 40))
    src = LineSource(grid, ix=15, j0=15, j1=25, profile=GaussianPulse(FCEN, 0.3))
    probe = PointProbe(40, 20)
    n_steps = 400
    _, (a,) = run_tm(init_tm_state(grid), fast, n_steps, source=src, monitors=[probe])
    _, (b,) = run_tm(init_tm_state(grid), slow, n_steps, source=src, monitors=[probe])
    assert torch.equal(a, b), f"max dev {(a - b).abs().max()}"


def _slab_T(sigma_val, freqs=(FCEN,)):
    """TM 1D-style (y_pmc) transmission amplitudes through an L=2 sigma slab."""
    grid = Grid(nx=600, ny=8, dx=0.05)
    eps = grid.zeros(600, 8) + 1.0
    kwargs = {}
    if sigma_val:
        sd = grid.zeros(600, 8)
        sd[300:340, :] = sigma_val                # slab x in [15, 17], L = 2
        kwargs["sigma_d"] = sd
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=30, y_pmc=True, **kwargs)
    src = LineSource(grid, ix=60, j0=0, j1=8, profile=GaussianPulse(FCEN, 0.2))
    probe = PointProbe(420, 4)                    # x=21, past the slab
    n_steps = int(round(60.0 / grid.dt))
    _, (series,) = run_tm(init_tm_state(grid, y_pmc=True), coeffs, n_steps,
                          source=src, monitors=[probe])
    s = series.detach().cpu().numpy()
    t = np.arange(len(s)) * grid.dt
    return np.array([abs(np.trapezoid(s * np.exp(-2j * math.pi * f * t), t))
                     for f in freqs])


def _t_analytic(sigma, L):
    w = 2 * math.pi * FCEN
    n_eff = complex(np.sqrt(1.0 + 1j * sigma / w))
    alpha = w * abs(n_eff.imag)
    # amplitude transmission incl. the two interface factors |t12 t21|
    return math.exp(-alpha * L) * abs(4 * n_eff / (1 + n_eff) ** 2)


def test_conductive_slab_attenuation_tm():
    T = float(_slab_T(1.0)[0] / _slab_T(0.0)[0])
    t_an = _t_analytic(1.0, 2.0)
    assert abs(T - t_an) / t_an < 0.02, f"|T|={T:.4f} vs analytic {t_an:.4f}"


def test_conductive_slab_vs_oracle():
    import json
    import pathlib
    with open(pathlib.Path(__file__).parent / "data" / "conductive_slab.json") as f:
        ref = json.load(f)
    freqs = ref["freqs"]
    t_ref = np.sqrt(np.array(ref["T_power"]))
    t_got = _slab_T(ref["sigma_d"], freqs) / _slab_T(0.0, freqs)
    rms = float(np.sqrt(np.mean((t_got / t_ref - 1.0) ** 2)))
    assert rms < 0.03, f"amplitude T rms vs oracle {rms:.4f}"


def _slab_T_te(sigma_val):
    """TE transmission |T(FCEN)|: wide aperture, ratio-normalized by caller."""
    from meep_torch.core.coeffs import build_te_coeffs
    from meep_torch.core.run import run_te
    from meep_torch.core.stepper_te import init_te_state

    grid = Grid(nx=600, ny=120, dx=0.05)
    kwargs = {}
    if sigma_val:
        sd = grid.zeros(600, 120)
        sd[300:340, :] = sigma_val                # slab x in [15, 17], L = 2
        kwargs["sigma_d"] = sd
    coeffs = build_te_coeffs(grid, eps_inf=grid.zeros(600, 120) + 1.0,
                             npml=30, **kwargs)
    src = LineSource(grid, ix=60, j0=35, j1=85, profile=GaussianPulse(FCEN, 0.2))
    src.mask = src.mask[:, :-1]                   # Ey grid


    class _EyProbe:
        reduce = "stack"

        def sample(self, state, t_idx):
            return state.Ey[420, 60]

    n_steps = int(round(60.0 / grid.dt))
    _, (series,) = run_te(init_te_state(grid), coeffs, n_steps,
                          source=src, monitors=[_EyProbe()])
    s = series.detach().cpu().numpy()
    t = np.arange(len(s)) * grid.dt
    return abs(np.trapezoid(s * np.exp(-2j * math.pi * FCEN * t), t))


def test_conductive_slab_attenuation_te():
    T = _slab_T_te(1.0) / _slab_T_te(0.0)
    t_an = _t_analytic(1.0, 2.0)
    assert abs(T - t_an) / t_an < 0.02, f"|T|={T:.4f} vs analytic {t_an:.4f}"
