# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import numpy as np
import pytest
import torch

from meep_torch.geometry.smoothing import blend_te, blend_tm


def test_tm_arithmetic():
    cov = torch.tensor([[0.0, 0.25, 1.0]], dtype=torch.float64)
    out = blend_tm(12.0, 2.0, cov)
    assert torch.allclose(out, torch.tensor([[2.0, 4.5, 12.0]], dtype=torch.float64))


def test_te_axis_aligned_vertical_interface():
    # normal = x-hat: eps_xx harmonic, eps_yy arithmetic
    cov = torch.tensor([[0.5]], dtype=torch.float64)
    nrm = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    exx, eyy = blend_te(12.0, 2.0, cov, nrm)
    h = 1 / (0.5 / 12 + 0.5 / 2)   # 24/7
    a = 0.5 * 12 + 0.5 * 2         # 7
    assert exx.item() == pytest.approx(h)
    assert eyy.item() == pytest.approx(a)


def test_te_no_boundary_is_arithmetic():
    cov = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    nrm = torch.zeros(1, 2, 2, dtype=torch.float64)
    exx, eyy = blend_te(12.0, 2.0, cov, nrm)
    assert torch.allclose(exx, torch.tensor([[12.0, 2.0]], dtype=torch.float64))
    assert torch.allclose(eyy, exx)


def test_te_45deg_between_h_and_a():
    cov = torch.tensor([[0.5]], dtype=torch.float64)
    s = 1 / np.sqrt(2)
    nrm = torch.tensor([[[s, s]]], dtype=torch.float64)
    exx, eyy = blend_te(12.0, 2.0, cov, nrm)
    h, a = 24 / 7, 7.0
    mid = 0.5 * h + 0.5 * a
    assert exx.item() == pytest.approx(mid)
    assert eyy.item() == pytest.approx(mid)
