# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import pytest

import torch

from meep_torch.core.grid import Grid
from meep_torch.core.sources import CWRamp, GaussianPulse, LineSource


def test_gaussian_peak_and_decay():
    p = GaussianPulse(fcen=0.65, fwidth=0.1)
    t0 = 5.0 / 0.1
    assert p.amplitude(t0) == 1.0                       # cos(0)*exp(0)
    assert abs(p.amplitude(0.0)) < 1e-5                 # cut off at start
    assert abs(p.amplitude(2 * t0)) < 1e-5


def test_gaussian_spectrum_peaks_at_fcen():
    p = GaussianPulse(fcen=0.65, fwidth=0.1)
    dt = 0.01
    n = int(p.t_end / dt) + 1
    ts = torch.arange(n, dtype=torch.float64) * dt
    sig = torch.tensor([p.amplitude(float(t)) for t in ts], dtype=torch.float64)
    spec = torch.fft.rfft(sig).abs()
    freqs = torch.fft.rfftfreq(n, dt)
    assert abs(freqs[spec.argmax()].item() - 0.65) < 0.02


def test_line_source_mask():
    g = Grid(nx=20, ny=10, dx=0.02)
    src = LineSource(g, ix=5, j0=2, j1=8, profile=GaussianPulse(0.65, 0.1))
    assert src.mask.shape == (20, 10)
    # The mask is a current DENSITY: each of the 6 cells carries dx of length
    # over the dx^2 cell area, i.e. 1/dx. Injecting a bare 1.0 per cell made
    # the field scale as dx instead of being resolution-independent as in
    # meep, so the weight is 1/dx = 50 here (dx = 0.02).
    assert src.mask[5, 2:8].sum().item() == pytest.approx(6.0 / g.dx)
    assert src.mask.sum().item() == pytest.approx(6.0 / g.dx)
    # a single-cell (point) source is a delta in both axes: 1/dx^2
    pt = LineSource(g, ix=5, j0=2, j1=3, profile=GaussianPulse(0.65, 0.1))
    assert pt.mask.sum().item() == pytest.approx(1.0 / g.dx ** 2)


def test_cw_ramp():
    p = CWRamp(fcen=0.5, t_ramp=10.0)
    assert abs(p.amplitude(5.0)) <= 0.5 + 1e-12          # still ramping
    assert p.amplitude(20.25) == math.sin(2 * math.pi * 0.5 * 20.25) * 1.0
