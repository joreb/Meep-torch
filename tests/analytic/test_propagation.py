# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import init_tm_state, step_tm


def plane_pulse_state(g, x0, w, y_pmc=True):
    s = init_tm_state(g, y_pmc=y_pmc)
    x = torch.arange(g.nx, dtype=g.dtype)
    ez = torch.exp(-((x - x0) ** 2) / (2 * w**2))
    Ez = ez[:, None].repeat(1, g.ny)
    # +x propagating: Hy = -Ez interpolated to half-grid
    Hy = -0.5 * (Ez[1:, :] + Ez[:-1, :])
    return s._replace(Ez=Ez, Dz=Ez.clone(), Hy=Hy)


def test_pulse_speed_is_c():
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=0, y_pmc=True)
    s = plane_pulse_state(g, x0=150.0, w=15.0)
    n_steps = 1200  # travels 1200*0.5 = 600 cells
    for _ in range(n_steps):
        s = step_tm(s, m)
    peak = s.Ez[:, 4].abs().argmax().item()
    travelled_cells = peak - 150
    expected_cells = n_steps * g.courant  # c=1: distance = n*dt = n*courant*dx
    assert abs(travelled_cells - expected_cells) / expected_cells < 0.01


def test_pulse_is_unidirectional():
    g = Grid(nx=1000, ny=8, dx=0.02)
    m = build_tm_coeffs(g, npml=0, y_pmc=True)
    s = plane_pulse_state(g, x0=500.0, w=15.0)
    for _ in range(400):
        s = step_tm(s, m)
    left = s.Ez[:400, 4].abs().max().item()
    right = s.Ez[600:, 4].abs().max().item()
    assert left < 0.05 * right
