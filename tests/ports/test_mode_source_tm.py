import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.ports.port import Port
from meep_torch.ports.mode_source import ModeSourceTM

FCEN, FWIDTH = 0.15, 0.05


def _waveguide():
    # straight waveguide eps=12, width 1um, along x; dx=0.1 -> 10 cells wide
    grid = Grid(nx=160, ny=80, dx=0.1)
    eps = torch.ones(grid.nx, grid.ny, dtype=torch.float64)
    eps[:, 35:45] = 12.0
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=10)
    return grid, coeffs


def test_one_way_launch_30db():
    grid, coeffs = _waveguide()
    port = Port(axis="x", index=80, lo=20, hi=60, direction=+1)
    src = ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
    state = init_tm_state(grid)
    # run long enough for the pulse to clear both probe planes
    n = 1200
    fwd_e, bwd_e = 0.0, 0.0
    for t in range(n):
        from meep_torch.core.stepper import step_tm
        state = step_tm(state, coeffs, src.amplitude(t), src.mask,
                        src.h_amplitude(t), **src.h_masks)
        fwd_e += float((state.Ez[110, 20:60] ** 2).sum())
        bwd_e += float((state.Ez[50, 20:60] ** 2).sum())
    ratio_db = 10.0 * torch.log10(torch.tensor(fwd_e / bwd_e))
    assert ratio_db > 30.0, f"one-way ratio {ratio_db:.1f} dB < 30 dB"


def test_one_way_launch_neg_direction_30db():
    # mirror of test_one_way_launch_30db: launch -x, assert the -x probe
    # dominates. Traps the direction=-1 delay bug (delay must not scale with d).
    grid, coeffs = _waveguide()
    port = Port(axis="x", index=80, lo=20, hi=60, direction=-1)
    src = ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)
    state = init_tm_state(grid)
    n = 1200
    fwd_e, bwd_e = 0.0, 0.0
    for t in range(n):
        from meep_torch.core.stepper import step_tm
        state = step_tm(state, coeffs, src.amplitude(t), src.mask,
                        src.h_amplitude(t), **src.h_masks)
        # -x launch: probe at 50 (downstream, -x of source) should dominate 110
        fwd_e += float((state.Ez[50, 20:60] ** 2).sum())
        bwd_e += float((state.Ez[110, 20:60] ** 2).sum())
    ratio_db = 10.0 * torch.log10(torch.tensor(fwd_e / bwd_e))
    assert ratio_db > 30.0, f"one-way (-x) ratio {ratio_db:.1f} dB < 30 dB"


def test_guided_mode_required():
    grid = Grid(nx=160, ny=80, dx=0.1)
    coeffs = build_tm_coeffs(grid, eps_inf=1.0, npml=10)  # vacuum: nothing guided
    port = Port(axis="x", index=80, lo=20, hi=60, direction=+1)
    import pytest
    with pytest.raises(ValueError, match="guided"):
        ModeSourceTM(grid, coeffs, port, GaussianPulse(FCEN, FWIDTH), FCEN)