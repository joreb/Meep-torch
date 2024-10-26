# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2 Wave 1 source-semantics tests."""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.sources import (ContinuousSource, CustomSource, GaussianPulse,
                                     LineSource, ObliqueLineSource, SoftSource)
from meep_torch.core.stepper import init_tm_state


def test_custom_source_passthrough():
    s = CustomSource(lambda t: math.sin(t))
    assert abs(s.amplitude(0.5) - math.sin(0.5)) < 1e-12


def test_continuous_source_is1cos_ramp():
    s = ContinuousSource(frequency=0.25, width=10.0, slowness=1.0)  # t_ramp=10
    assert abs(s.amplitude(0.0)) < 1e-12
    # at t=5 (mid-ramp, ramp=0.5) carrier sin(2*pi*0.25*5)=sin(2.5pi)=1 -> 0.5
    assert abs(s.amplitude(5.0) - 0.5) < 1e-6
    # at t=13 (past ramp, ramp=1) carrier sin(2*pi*0.25*13)=sin(6.5pi)=1 -> 1.0
    assert abs(s.amplitude(13.0) - 1.0) < 1e-6


def test_line_source_amp_func_spatial_profile():
    g = Grid(nx=40, ny=30, dx=0.1)
    src = LineSource(g, ix=5, j0=10, j1=20, profile=GaussianPulse(1.0, 0.2),
                     amp_func=lambda y: math.exp(-((y - 1.55) ** 2) / 0.02))
    m = src.mask[5, 10:20].numpy()
    assert int(np.argmax(m)) == 5              # j=15 -> y=1.55 (gaussian center)
    assert abs(m.max() - 1.0) < 1e-6 and m.min() < 0.01


def test_soft_source_area_launches_field():
    g = Grid(nx=60, ny=40, dx=0.1)
    mask = g.zeros(g.nx, g.ny)
    mask[10, 15:25] = 1.0
    src = SoftSource(g, mask, GaussianPulse(0.5, 0.2))
    c = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64), npml=8)
    st = init_tm_state(g)
    st, _ = run_tm(st, c, 200, source=src, monitors=())
    assert float(st.Ez.abs().max()) > 1e-6


def test_oblique_line_source_launch_angle():
    theta = math.radians(20.0)
    fcen = 0.3
    g = Grid(nx=300, ny=300, dx=0.05)
    npml = 20
    src = ObliqueLineSource(g, ix=npml + 5, j0=npml, j1=g.ny - npml,
                            profile=GaussianPulse(fcen, 0.1), theta=theta, fcen=fcen)
    c = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64), npml=npml)
    st = init_tm_state(g)
    n_steps = int(round(60.0 / g.dt))            # let the pulse propagate +x
    st, _ = run_tm(st, c, n_steps, source=src, monitors=())
    # k_y from FFT of Ez(y) at a downstream column
    col = g.nx - npml - 30
    Ez_y = st.Ez[col, npml:g.ny - npml].numpy()
    if Ez_y.std() < 1e-8:                         # pulse not reached -> skip angle check
        assert False, "oblique pulse did not reach the downstream column"
    spec = np.abs(np.fft.rfft(Ez_y - Ez_y.mean(), n=8192))
    kys = np.fft.rfftfreq(8192, g.dx) * 2 * math.pi
    k_target = 2 * math.pi * fcen * math.sin(theta)
    k_meas = float(kys[int(spec.argmax())])
    dtheta = abs(math.degrees(math.asin(k_meas / (2 * math.pi * fcen))) - 20.0)
    assert dtheta < 1.0, f"oblique angle {20 - dtheta:.2f}+{dtheta:.2f}deg, off by {dtheta:.2f}deg"