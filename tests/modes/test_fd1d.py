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


def test_mode_parity_classifies_a_symmetric_slab():
    """Bands of a symmetric slab alternate even, odd, even, odd.

    meep's eig_parity selects a mode by its symmetry, not by its position in
    the neff ordering, so "EVEN_Y band 1" is the first EVEN mode -- which is
    band 2 of the full ordering whenever the fundamental is odd. Turning a
    meep band index into a solved mode therefore needs the profiles
    classified, and this pins that classification.
    """
    import torch

    from meep_torch.modes.fd1d import mode_parity, solve_1d

    n, dx = 120, 0.05
    eps = torch.ones(n, dtype=torch.float64)
    eps[50:70] = 12.0                      # slab centred on the line
    modes = solve_1d(eps, dx, 0.15, "tm", k=4)
    assert [mode_parity(modes.profiles[i]) for i in range(4)] == [1, -1, 1, -1]


def test_mode_parity_refuses_an_asymmetric_line():
    """No parity to report when the structure has no mirror: 0, not a guess.

    Returning the nearest parity here would let a caller asking for EVEN_Y
    receive a mode that is not even, which is the plausible-wrong-answer
    failure this codebase keeps paying for. 0 means "ask a different
    question".
    """
    import torch

    from meep_torch.modes.fd1d import mode_parity, solve_1d

    n, dx = 120, 0.05
    eps = torch.ones(n, dtype=torch.float64)
    eps[40:70] = 12.0                      # off-centre slab: no mirror
    modes = solve_1d(eps, dx, 0.15, "tm", k=2)
    assert mode_parity(modes.profiles[0]) == 0


def test_select_mode_counts_only_modes_of_the_requested_parity():
    """meep's band index counts within the parity, not within the ordering.

    THE ODD CASE IS THE ONLY ONE THAT TESTS ANYTHING. On a symmetric slab the
    fundamental is already even, so EVEN_Y band 1 is the same mode the
    unfiltered band 1 gives -- that assertion passes whether the filter works
    or not. ODD_Y band 1 is the mode at ordering index 2, so it moves only if
    the selection is really happening.
    """
    import torch

    from meep_torch.modes.fd1d import select_mode, solve_1d

    n, dx = 120, 0.05
    eps = torch.ones(n, dtype=torch.float64)
    eps[50:70] = 12.0
    ordering = solve_1d(eps, dx, 0.15, "tm", k=2).neff

    neff_any, _ = select_mode(eps, dx, 0.15, "tm", band=0, parity=0)
    neff_even, _ = select_mode(eps, dx, 0.15, "tm", band=0, parity=+1)
    neff_odd, _ = select_mode(eps, dx, 0.15, "tm", band=0, parity=-1)

    assert neff_any == pytest.approx(float(ordering[0]))
    assert neff_even == pytest.approx(float(ordering[0]))   # no-op here
    assert neff_odd == pytest.approx(float(ordering[1]))    # the real check
    assert neff_odd != pytest.approx(neff_even)


def test_select_mode_refuses_a_parity_the_line_cannot_have():
    """An asymmetric line has no parity, so a filtered request must raise.

    Handing back the nearest mode would answer a question the structure cannot
    answer, which is the plausible-wrong-answer failure mode this codebase
    keeps paying for.
    """
    import torch

    from meep_torch.modes.fd1d import select_mode

    n, dx = 120, 0.05
    eps = torch.ones(n, dtype=torch.float64)
    eps[40:70] = 12.0                      # off-centre: no mirror
    with pytest.raises(ValueError, match="even"):
        select_mode(eps, dx, 0.15, "tm", band=0, parity=+1)


def test_select_mode_handles_the_half_domain_basis():
    """A pmc_lo profile is half a mode, so its parity is not in the profile.

    bc="pmc_lo" solves the y >= 0 half against a magnetic wall at the cut
    plane, so every mode it returns is the half of an EVEN mode: the parity is
    enforced by the boundary condition and is invisible to a classifier, which
    sees a profile that is not symmetric about its own centre and reports 0.
    Filtering on that would refuse a request that is actually satisfiable, so
    EVEN_Y is a no-op here and ODD_Y is refused for a physical reason rather
    than by finding no match.
    """
    import torch

    from meep_torch.modes.fd1d import select_mode

    n, dx = 120, 0.05
    eps = torch.ones(n, dtype=torch.float64)
    eps[:20] = 12.0                        # core sitting against the cut plane
    plain, _ = select_mode(eps, dx, 0.15, "tm", band=0, parity=0, bc="pmc_lo")
    even, _ = select_mode(eps, dx, 0.15, "tm", band=0, parity=+1, bc="pmc_lo")
    assert even == pytest.approx(plain)
    with pytest.raises(ValueError, match="pmc_lo"):
        select_mode(eps, dx, 0.15, "tm", band=0, parity=-1, bc="pmc_lo")


def test_select_modes_returns_a_basis_all_of_one_parity():
    """Every mode in a filtered basis has the requested parity.

    This is the property that keeps a source and a monitor talking about the
    same thing. compute_smatrix injects the parity-selected mode and projects
    onto select_modes' basis; if that basis were the unfiltered ordering, the
    decomposition would be in a different basis than the one driven. On a
    symmetric guide with EVEN_Y the two coincide, so the mismatch is invisible
    in the case anyone would test -- it only appears for ODD_Y.
    """
    import torch

    from meep_torch.modes.fd1d import mode_parity, select_modes

    n, dx = 200, 0.05
    eps = torch.ones(n, dtype=torch.float64)
    eps[70:130] = 12.0
    for parity in (+1, -1):
        _, profiles = select_modes(eps, dx, 0.15, "tm", n=3, parity=parity)
        assert [mode_parity(p) for p in profiles] == [parity] * 3

    # and the two families interleave in the raw ordering, which is why a band
    # index cannot stand in for a parity selection
    even, _ = select_modes(eps, dx, 0.15, "tm", n=2, parity=+1)
    odd, _ = select_modes(eps, dx, 0.15, "tm", n=2, parity=-1)
    assert even[0] > odd[0] > even[1] > odd[1]
