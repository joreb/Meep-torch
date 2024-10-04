import math

import pytest
import torch
from meep_torch.core.grid import Grid
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix

FCEN, FWIDTH = 0.15, 0.1
FREQS = [0.13, 0.15, 0.17]


@pytest.fixture(scope="module")
def straight():
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    ports = [
        Port(axis="x", index=40, lo=20, hi=60, direction=+1),
        Port(axis="x", index=160, lo=20, hi=60, direction=-1),
    ]
    return compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                           freqs=FREQS, n_steps=3000, npml=10), grid


def test_straight_guide_transmission(straight):
    sm, _ = straight
    for i in range(len(FREQS)):
        assert abs(abs(sm.s[i, 1, 0]) - 1.0) < 1e-2
        assert abs(sm.s[i, 0, 0]) < 5e-2          # |S11| small


def test_straight_guide_unitarity(straight):
    sm, _ = straight
    for i in range(len(FREQS)):
        col = sm.s[i, :, 0]
        assert abs(float((col.abs() ** 2).sum()) - 1.0) < 1e-2


def test_phase_slope_matches_group_index():
    # S21 phase accumulates -2*pi*f*neff(f)*L over the inter-port distance L.
    # With L=12um and neff~2.85 that is ~5 phase wraps over [0.13,0.17], so the
    # 3-freq 0.02 spacing aliases (delta-phi = 2*pi*ng*L*0.02 ~ 5.6 rad > pi) and
    # np.unwrap cannot resolve it. Use a fine 0.005-spaced sweep near fcen so
    # delta-phi ~ 1.4 rad < pi and the unwrapped slope recovers the group index.
    grid = Grid(nx=200, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    ports = [
        Port(axis="x", index=40, lo=20, hi=60, direction=+1),
        Port(axis="x", index=160, lo=20, hi=60, direction=-1),
    ]
    fine = [0.140, 0.145, 0.150, 0.155, 0.160]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=fine, n_steps=3000, npml=10)
    L = (160 - 40) * grid.dx
    ph = torch.from_numpy(__import__("numpy").unwrap(
        torch.tensor([torch.angle(sm.s[i, 1, 0]) for i in range(len(fine))]).numpy()))
    slope = float((ph[-1] - ph[0]) / (fine[-1] - fine[0]))
    ng_est = -slope / (2 * math.pi * L)
    from meep_torch.modes.fd1d import solve_1d
    eps_line = torch.ones(40, dtype=torch.float64); eps_line[15:25] = 12.0
    n_lo = float(solve_1d(eps_line, grid.dx, 0.140, "tm", k=1).neff[0])
    n_hi = float(solve_1d(eps_line, grid.dx, 0.160, "tm", k=1).neff[0])
    n_c = float(solve_1d(eps_line, grid.dx, 0.150, "tm", k=1).neff[0])
    ng_ref = n_c + 0.15 * (n_hi - n_lo) / (0.160 - 0.140)  # neff + f*dneff/df at fcen
    assert abs(ng_est - ng_ref) / ng_ref < 0.05


def test_reciprocity_taper():
    grid = Grid(nx=240, ny=100, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    # width 1um -> 2um linear taper between x-cells 100..140
    for ix in range(grid.nx):
        w = 10 if ix < 100 else (20 if ix >= 140 else 10 + (ix - 100) * 10 // 40)
        eps[ix, 50 - w // 2 : 50 + (w + 1) // 2] = 12.0
    ports = [
        Port(axis="x", index=40, lo=25, hi=75, direction=+1),
        Port(axis="x", index=200, lo=25, hi=75, direction=-1),
    ]
    sm = compute_smatrix(eps, grid, ports, pol="tm", fcen=FCEN, fwidth=FWIDTH,
                         freqs=[FCEN], n_steps=3500, npml=10)
    s21, s12 = sm.s[0, 1, 0], sm.s[0, 0, 1]
    assert abs(abs(s21) - abs(s12)) < 5e-3