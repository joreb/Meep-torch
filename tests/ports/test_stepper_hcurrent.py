import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs, build_te_coeffs
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.core.stepper_te import init_te_state, step_te


def test_tm_hy_current_injects():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_tm_state(grid)
    mask = grid.zeros(grid.nx - 1, grid.ny)
    mask[10, 10] = 1.0
    out = step_tm(state, m, src_h_amp=2.0, src_hy_mask=mask)
    assert torch.isclose(out.Hy[10, 10], torch.tensor(2.0 * m.dt, dtype=torch.float64))
    # no-current call unchanged (back-compat)
    out2 = step_tm(state, m)
    assert out2.Hy.abs().max() == 0.0


def test_tm_hx_current_injects():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_tm_state(grid)
    mask = grid.zeros(grid.nx, grid.ny - 1)
    mask[10, 10] = 1.0
    out = step_tm(state, m, src_h_amp=-1.5, src_hx_mask=mask)
    assert torch.isclose(out.Hx[10, 10], torch.tensor(-1.5 * m.dt, dtype=torch.float64))


def test_te_hz_current_injects():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_te_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_te_state(grid)
    mask = grid.zeros(grid.nx - 1, grid.ny - 1)
    mask[10, 10] = 1.0
    out = step_te(state, m, src_h_amp=3.0, src_hz_mask=mask)
    assert torch.isclose(out.Hz[10, 10], torch.tensor(3.0 * m.dt, dtype=torch.float64))


def test_h_current_seen_by_same_step_d_update():
    # injected Hy at column i must contribute to Dz curl in the SAME step
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state = init_tm_state(grid)
    mask = grid.zeros(grid.nx - 1, grid.ny)
    mask[10, 10] = 1.0
    out = step_tm(state, m, src_h_amp=1.0, src_hy_mask=mask)
    assert out.Dz.abs().max() > 0.0