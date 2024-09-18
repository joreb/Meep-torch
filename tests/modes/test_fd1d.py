# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import numpy as np
import pytest
import torch

from meep_torch.modes.fd1d import Modes1D, solve_1d, solve_1d_tm_torch


def analytic_slab_te0_sym(n_core, n_clad, d, lam):
    """Fundamental even TE mode of symmetric slab (E parallel to interfaces == our
    'tm'/Ez operator). Dispersion: tan(kappa d/2) = gamma/kappa, with kappa d/2 in
    (0, pi/2) => neff in (neff_pole1, n_core), where neff_pole1 is the neff at which
    kappa d/2 = pi/2 (the first tan pole). g(lo)->+inf, g(hi)->-inf, single root.

    NOTE: the original brief fixture clamped `hi` to BELOW the first pole, which lands
    in the odd-mode region and returns a near-cutoff higher mode (~1.456) instead of
    the fundamental (~3.27 for Si/SiO2, lam=1.55, d=0.5, V=3.2). Fixed per the brief's
    own NOTE that buggy analytic fixtures must be recomputed, not the solver fudged."""
    k0 = 2 * math.pi / lam

    def g(neff):
        kap = k0 * math.sqrt(n_core**2 - neff**2)
        gam = k0 * math.sqrt(neff**2 - n_clad**2)
        return math.tan(kap * d / 2) - gam / kap

    # First tan pole: kappa d/2 = pi/2 => kappa = pi/d => neff_pole1 = sqrt(n_core^2 - (pi/(k0 d))^2).
    # The fundamental even branch sits ABOVE this pole (toward n_core). If the pole
    # falls below n_clad (V < pi/2), the whole guided range (n_clad, n_core) is one branch.
    pole_sq = n_core**2 - (math.pi / (k0 * d))**2
    lo = math.sqrt(pole_sq) + 1e-9 if pole_sq > n_clad**2 else n_clad + 1e-9
    hi = n_core - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def slab_line(n_core, n_clad, d, dx, n_cells):
    y = (np.arange(n_cells) + 0.5) * dx
    ctr = n_cells * dx / 2
    eps = np.where(np.abs(y - ctr) < d / 2, n_core**2, n_clad**2)
    return torch.tensor(eps, dtype=torch.float64)


def test_tm_slab_matches_analytic():
    lam, d = 1.55, 0.5
    eps = slab_line(3.4777, 1.444, d, dx=0.005, n_cells=1200)
    m = solve_1d(eps, dx=0.005, f=1 / lam, pol="tm", k=1)
    ref = analytic_slab_te0_sym(3.4777, 1.444, d, lam)
    assert m.neff[0].item() == pytest.approx(ref, abs=2e-4)  # FD discretization limit
    assert m.profiles.shape == (1, 1200)
    peak = torch.argmax(torch.abs(m.profiles[0]))
    assert 500 < peak.item() < 700  # confined to core


def test_mode_count_vs_v_parameter():
    lam, d = 1.55, 1.2
    V = 2 * math.pi / lam * d / 2 * math.sqrt(3.4777**2 - 1.444**2)
    # Symmetric slab guided-mode count (neff > n_clad): floor(2V/pi) + 1.
    n_guided_expected = int(2 * V / math.pi) + 1
    eps = slab_line(3.4777, 1.444, d, dx=0.005, n_cells=1600)
    m = solve_1d(eps, dx=0.005, f=1 / lam, pol="tm", k=8)
    guided = (m.neff > 1.444 + 1e-6).sum().item()
    assert guided == n_guided_expected


def test_te_slab_close_to_tm_weak_guiding():
    # low contrast: TE and TM neff nearly equal
    eps = slab_line(1.50, 1.48, 1.0, dx=0.01, n_cells=800)
    mt = solve_1d(eps, dx=0.01, f=1 / 1.55, pol="tm", k=1)
    me = solve_1d(eps, dx=0.01, f=1 / 1.55, pol="te", k=1)
    assert me.neff[0].item() == pytest.approx(mt.neff[0].item(), abs=2e-4)
    # high contrast: they must differ measurably
    eps2 = slab_line(3.4777, 1.444, 0.3, dx=0.005, n_cells=1200)
    mt2 = solve_1d(eps2, dx=0.005, f=1 / 1.55, pol="tm", k=1)
    me2 = solve_1d(eps2, dx=0.005, f=1 / 1.55, pol="te", k=1)
    assert abs(me2.neff[0].item() - mt2.neff[0].item()) > 0.05


def test_torch_tm_grad_path():
    eps = slab_line(3.4777, 1.444, 0.5, dx=0.01, n_cells=400).requires_grad_(True)
    neff = solve_1d_tm_torch(eps, dx=0.01, f=1 / 1.55, k=1)[0]
    neff.backward()
    assert eps.grad is not None and torch.isfinite(eps.grad).all()
    core = eps.grad[180:220]
    assert torch.all(core > 0)  # raising core eps raises neff