# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import pytest
import torch

from meep_torch.core.materials import SiO2, cSi
from meep_torch.modes.effindex import Medium2D, group_index, neff_table

STACK = [(2.0, SiO2), (0.22, cSi), (2.0, SiO2)]   # SOI 220nm
FREQS = [1 / 1.60, 1 / 1.55, 1 / 1.50]


def test_neff_table_monotone_and_bounded():
    t = neff_table(STACK, FREQS, pol="tm")
    n_sio2 = SiO2.n(1 / 1.55).real
    n_si = cSi.n(1 / 1.55).real
    assert torch.all(t > n_sio2) and torch.all(t < n_si)
    assert t[2] > t[0]   # higher frequency -> higher neff (normal dispersion)


def test_group_index_matches_finite_difference():
    f0 = 1 / 1.55
    ng = group_index(STACK, f0, pol="tm")
    df = 1e-4
    tp = neff_table(STACK, [f0 + df], pol="tm")[0].item()
    tm_ = neff_table(STACK, [f0 - df], pol="tm")[0].item()
    n0 = neff_table(STACK, [f0], pol="tm")[0].item()
    ng_fd = n0 + f0 * (tp - tm_) / (2 * df)
    assert ng == pytest.approx(ng_fd, abs=1e-4)
    assert 3.5 < ng < 5.0   # SOI 220nm slab ng ballpark


def test_medium2d_interpolation():
    t = neff_table(STACK, FREQS, pol="tm")
    med = Medium2D(torch.tensor(FREQS, dtype=torch.float64), t)
    mid = med.eps_at(1 / 1.55)
    assert mid == pytest.approx(t[1].item() ** 2, rel=1e-12)
    interp = med.eps_at(0.5 * (FREQS[0] + FREQS[1]))
    assert min(t[0], t[1]) ** 2 < interp < max(t[0], t[1]) ** 2


def test_lossy_material_rejected():
    class Fake:
        eps_inf = 1.0
        poles = ()
        def n(self, f):
            return 2.0 + 0.5j
    with pytest.raises(ValueError, match="loss"):
        neff_table([(1.0, Fake())], [1 / 1.55], pol="tm")