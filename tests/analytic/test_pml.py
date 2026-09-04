# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import step_tm
from tests.analytic.test_energy import total_energy, yee_invariant
from tests.analytic.test_propagation import plane_pulse_state


def test_pml_absorbs_below_minus_40db():
    g = Grid(nx=800, ny=8, dx=0.02)
    npml = 40
    m = build_tm_coeffs(g, npml=npml, y_pmc=True)
    s = plane_pulse_state(g, x0=400.0, w=15.0)
    e0 = total_energy(s, m, g.dx)
    # pulse reaches +x PML after ~360 cells -> 720 steps; run 3x for full decay
    for _ in range(2400):
        s = step_tm(s, m)
    e1 = total_energy(s, m, g.dx)
    assert e1 / e0 < 1e-4  # -40 dB energy


def test_pec_reflects_without_pml():
    # Deviation note: brief used naive total_energy with bound > 0.99.
    # naive total_energy carries ~3% staggering oscillation (documented in
    # test_energy.py). Replaced with yee_invariant (consecutive-state pairs)
    # which is conserved to machine precision in a lossless PEC/PMC box.
    # Bound remains > 0.99 -- yee_invariant ratio lands at ~1 (drift ~1e-12).
    g = Grid(nx=800, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=0, y_pmc=True)
    s = plane_pulse_state(g, x0=400.0, w=15.0)
    # sample initial invariant from first consecutive pair
    s_prev = s
    s = step_tm(s, m)
    e0 = yee_invariant(s_prev, s, m, g.dx)
    for _ in range(2399):
        s_prev = s
        s = step_tm(s, m)
    e1 = yee_invariant(s_prev, s, m, g.dx)
    assert e1 / e0 > 0.99  # sanity contrast: PEC/PMC box keeps energy
