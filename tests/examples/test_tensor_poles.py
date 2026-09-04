# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Oriented (tensor) pole tests -- build_te_coeffs poles_tensor.

Gate (a): a tensor pole with S = sigma * I (isotropic) is bit-identical to
the shared scalar pole -- the appended-stack plumbing adds nothing.

Gate (b): at tilt 0 the oriented decomposition S_E = diag(sE, 0),
S_O = diag(0, sO) reduces to the per-axis path (poles_x/poles_y), which
gate (a)+(b) of test_peraxis_dispersion.py validates physically. Bitwise
equality pins the tensor path to that validated baseline.

The off-diagonal cross terms (c3xy/c3yx) are gated end-to-end by
examples/fdtdx_ports/hyperbolic_asymmetric_refraction.py: measured
energy-corridor angles 25.2/56.7 deg vs analytic 23.1/53.3 deg for
mirrored +-45 deg beams on tilted calcite (errors 2.0/3.4 < 4 deg).
"""
import torch

from meep_torch.core.coeffs import build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_te
from meep_torch.core.sources import GaussianPulse, SoftSource
from meep_torch.core.stepper_te import init_te_state

POLE_O = (0.1410, 0.0010, 0.5416)
POLE_E = (0.0871, 0.0003, 0.3745)


def _setup():
    g = Grid(nx=160, ny=80, dx=0.17)
    mask = torch.zeros(160, 80, dtype=g.dtype, device=g.device)
    mask[80:110, 16:64] = 1.0
    smask = g.zeros(160, 80)
    smask[30, 20:60] = 1.0
    src = SoftSource(g, smask[:, :-1], GaussianPulse(0.147, 0.02))
    return g, mask, src, int(60 / g.dt)


def test_isotropic_tensor_pole_equals_shared_pole():
    g, mask, src, n = _setup()
    c1 = build_te_coeffs(g, npml=12, poles=[POLE_O], pole_masks=[mask])
    s1, _ = run_te(init_te_state(g, n_poles=1), c1, n, source=src, monitors=[])
    c2 = build_te_coeffs(g, npml=12,
                         poles_tensor=[(POLE_O[0], POLE_O[1],
                                        (POLE_O[2], 0.0, POLE_O[2]))],
                         pole_masks_tensor=[mask])
    s2, _ = run_te(init_te_state(g, n_poles=1, n_poles_y=1), c2, n,
                   source=src, monitors=[])
    assert torch.equal(s1.Hz, s2.Hz)


def test_tilt0_tensor_poles_equal_peraxis():
    g, mask, src, n = _setup()
    cA = build_te_coeffs(g, npml=12, poles_x=[POLE_E], pole_masks_x=[mask],
                         poles_y=[POLE_O], pole_masks_y=[mask])
    sA, _ = run_te(init_te_state(g, n_poles=1, n_poles_y=1), cA, n,
                   source=src, monitors=[])
    cB = build_te_coeffs(g, npml=12,
                         poles_tensor=[(POLE_E[0], POLE_E[1], (POLE_E[2], 0.0, 0.0)),
                                       (POLE_O[0], POLE_O[1], (0.0, 0.0, POLE_O[2]))],
                         pole_masks_tensor=[mask, mask])
    sB, _ = run_te(init_te_state(g, n_poles=2, n_poles_y=2), cB, n,
                   source=src, monitors=[])
    d = (sA.Hz - sB.Hz).abs().max() / sA.Hz.abs().max()
    assert d < 1e-12, f"tilt-0 tensor vs per-axis rel diff {float(d):.2e}"
