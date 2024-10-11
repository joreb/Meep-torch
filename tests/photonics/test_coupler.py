# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Photonics parity group 3: coupler even/odd supermode split.

Mirrors Photonics_Simulation's `EffectiveIndex_Coupler`
(`src/photonic/coupler/__init__.py`, `Testing_DBR_documented.ipynb` cell 37):
two SiN cores (width 0.86um, height 0.15um) on SiO2, separated by a gap g.
The notebook solves the 2D two-core cross-section with tidy3d
(`gt.modes.WaveguideCoupler`), extracts even/odd neff, and fits each to
n_E(g) = n_eff + a_E*exp(-gamma_E*g), n_O(g) = n_eff - a_O*exp(-gamma_O*g)
(`coupler/__init__.py:165-207`). The coupler transfer matrix uses
kappa = sin((2*pi/lambda)*dz*(n_even - n_odd)/2) -- i.e. the coupling rate is
proportional to the even/odd neff split.

Here the two-core cross-section is the slab-collapsed 1D lateral line (EIM
core eps = n_v^2, two width-w cores separated by a gap of clad), solved with
fd1d (k=2). The even (symmetric, higher neff) and odd (antisymmetric, lower
neff) supermodes are the top two eigenvalues. The fitted lmfit coefficients are
runtime-only (not persisted), so the gate is the qualitative behavior the
notebook's exponential model encodes: even > odd, the split decreases with
gap, and it decays roughly exponentially to zero (decoupled cores).
"""
import numpy as np
import pytest

from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.modes.effindex import neff_table
from meep_torch.modes.fd1d import solve_1d

W = 0.86          # core width (um) -- notebook coupler width
H = 0.150         # core height (um) -- notebook value
LAM = 0.674       # design wavelength (um)
DX = 0.005
GAPS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.5]
PAD = 2.0         # clad padding each side (um)


def _n_v():
    return float(neff_table([(2.0, SiO2), (H, Si3N4), (2.0, SiO2)], [1.0 / LAM], "tm", dx=DX)[0])


def _supermodes(gap):
    f = 1.0 / LAM
    n_v = _n_v()
    n_sio2 = SiO2.n(f).real
    eps_line = np.concatenate([
        np.full(int(PAD / DX), n_sio2 ** 2),
        np.full(int(W / DX), n_v ** 2),
        np.full(int(gap / DX), n_sio2 ** 2),
        np.full(int(W / DX), n_v ** 2),
        np.full(int(PAD / DX), n_sio2 ** 2),
    ])
    m = solve_1d(eps_line, DX, f, "tm", k=2)
    return float(m.neff[0]), float(m.neff[1])  # even (higher), odd (lower)


@pytest.fixture(scope="module")
def splits():
    return [(_supermodes(g)) for g in GAPS]


def test_even_higher_than_odd(splits):
    for g, (n_even, n_odd) in zip(GAPS, splits):
        assert n_even > n_odd, (
            f"even supermode {n_even:.5f} not higher than odd {n_odd:.5f} at gap={g}")


def test_split_decreases_with_gap(splits):
    sp = [n_e - n_o for n_e, n_o in splits]
    for i in range(len(sp) - 1):
        assert sp[i] > sp[i + 1], (
            f"split not decreasing: gap {GAPS[i]}->{GAPS[i+1]} "
            f"split {sp[i]:.5e}->{sp[i+1]:.5e}")


def test_split_decays_exponentially(splits):
    # The notebook model n_E-n_O ~ a*exp(-gamma*g): log(split) is roughly linear
    # in g. Fit log(split) vs g over the decaying range and check R^2.
    sp = np.array([n_e - n_o for n_e, n_o in splits])
    mask = sp > 1e-9
    g = np.array(GAPS)[mask]
    logsp = np.log(sp[mask])
    # linear fit logsp = a + b*g
    b, a = np.polyfit(g, logsp, 1)
    fit = a + b * g
    ss_res = float(np.sum((logsp - fit) ** 2))
    ss_tot = float(np.sum((logsp - logsp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    assert r2 > 0.9, f"split not exponentially decaying in gap: R^2={r2:.3f}"
    # decay constant positive (split shrinks as gap grows)
    assert b < 0, f"split decay rate {b:.3f} not negative"


def test_split_vanishes_at_large_gap(splits):
    # Decoupled cores at large gap: even == odd (split -> 0).
    n_even, n_odd = splits[-1]  # gap = 1.5um
    assert abs(n_even - n_odd) < 1e-4, (
        f"split at large gap (1.5um) {n_even - n_odd:.2e} not -> 0")


def test_kappa_increases_as_gap_shrinks(splits):
    # coupler kappa ~ sin((2*pi/lam)*dz*(n_even-n_odd)/2) is monotone in the
    # split, so the coupling strength tracks the split: smaller gap -> stronger.
    sp = [n_e - n_o for n_e, n_o in splits]
    assert sp[0] > sp[3], f"coupling not stronger at small gap: {sp[0]:.2e} vs {sp[3]:.2e}"