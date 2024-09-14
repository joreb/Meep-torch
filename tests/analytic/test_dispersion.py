# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import cmath

import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import LorentzPole, analytic_eps
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state

POLE = LorentzPole(fn=2.0, gamma=0.1, sigma=1.5)
EPS_INF = 2.25
FREQS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def slab_transmission(n2: complex, d: float, f: float) -> float:
    delta = 2 * cmath.pi * f * n2 * d
    r12 = (1 - n2) / (1 + n2)
    t = (2 / (1 + n2)) * (2 * n2 / (1 + n2)) * cmath.exp(1j * delta)
    t /= 1 - r12**2 * cmath.exp(2j * delta)
    return abs(t) ** 2


def run_transmission(with_slab: bool) -> torch.Tensor:
    g = Grid(nx=1400, ny=8, dx=0.02)
    slab = torch.zeros(g.nx, g.ny, dtype=torch.float64)
    slab[600:700, :] = 1.0                     # 2 um slab
    eps = 1.0 + (EPS_INF - 1.0) * slab if with_slab else None
    poles = [POLE] if with_slab else []
    masks = [slab] if with_slab else None
    m = build_tm_coeffs(g, eps_inf=eps, poles=poles, pole_masks=masks, npml=40, y_pmc=True)
    src = LineSource(g, ix=100, j0=0, j1=8, profile=GaussianPulse(0.65, 0.35))
    mon = FluxMonitorTM(g, FREQS, x_normal=True, idx=1200, lo=0, hi=8)
    s = init_tm_state(g, n_poles=len(poles), y_pmc=True)
    _, (acc,) = run_tm(s, m, 9000, source=src, monitors=[mon])
    return mon.flux(acc, g.dx)


def test_dispersive_slab_matches_transfer_matrix():
    ref = run_transmission(with_slab=False)
    tr = run_transmission(with_slab=True)
    T_sim = tr / ref
    for i, f in enumerate(FREQS):
        n2 = cmath.sqrt(analytic_eps(EPS_INF, [POLE], f))
        T_th = slab_transmission(n2, d=2.0, f=f)
        assert abs(T_sim[i].item() - T_th) / T_th < 0.03, (f, T_sim[i].item(), T_th)
