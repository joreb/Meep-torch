import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.port import Port
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.mode_monitor import decompose, dft_fields

FCEN, FWIDTH = 0.15, 0.1
FREQS = [0.13, 0.15, 0.17]


def _run_straight():
    grid = Grid(nx=160, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64, device=grid.device)
    eps[:, 35:45] = 12.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
    port = Port(axis="x", index=30, lo=20, hi=60, direction=+1)
    src = ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
    # Near-source monitor, run just long enough to capture the forward pass
    # before the right-PML reflection returns — the DFT then sees an almost
    # pure forward wave (ratio > 100). A far monitor over a long run sees the
    # PML-reflected backward wave (~20% at npml=10), which is real, not a
    # decomposition bug (the S-matrix uses the forward a_p, which is conserved).
    mon = FluxMonitorTM(grid, FREQS, x_normal=True, idx=50, lo=20, hi=60)
    _, (acc,) = run_tm(init_tm_state(grid), coeffs, 2000, source=src, monitors=[mon])
    eps_line = eps[50, 20:60]
    return acc, eps_line, grid.dx


def test_mode_orthonormality():
    # decompose's own overlap of mode m against mode n is delta_mn.
    # 3um core (eps[15:45]=12 on 60 nodes) supports 3 guided TM modes so
    # every neff > 0 (a 1um core only guides 1 mode -> higher modes have
    # neff=0 and the norm divides by zero).
    eps_line = torch.ones(60, dtype=torch.float64)
    eps_line[15:45] = 12.0
    dx = 0.1
    modes = solve_1d(eps_line, dx, FCEN, "tm", k=3)
    for m in range(3):
        Em = modes.profiles[m]
        for n in range(3):
            Hn = float(modes.neff[n]) * modes.profiles[n]
            ovl = float((Em * Hn).sum() * dx)
            norm = float(modes.neff[m]) * float((modes.profiles[m] ** 2).sum()) * dx
            if m == n:
                assert abs(ovl / norm - 1.0) < 1e-10
            else:
                assert abs(ovl) / norm < 1e-8


def test_forward_dominates_backward():
    acc, eps_line, dx = _run_straight()
    a_p, a_m = decompose(acc, eps_line, dx, FREQS, "tm", 1)
    for i in range(len(FREQS)):
        assert abs(a_p[i, 0]) > 100.0 * abs(a_m[i, 0])


def test_dft_fields_shapes():
    acc, eps_line, dx = _run_straight()
    E, H = dft_fields(acc)
    assert E.shape == (3, 40) and E.dtype == torch.complex128
    assert H.shape == (3, 40) and H.dtype == torch.complex128