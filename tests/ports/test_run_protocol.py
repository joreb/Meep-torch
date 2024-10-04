import torch
from meep_torch.core.grid import Grid
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.run import run_tm
from meep_torch.core.stepper import init_tm_state


class DualSource:
    def __init__(self, grid):
        self.mask = grid.zeros(grid.nx, grid.ny)
        self.mask[8, 10] = 1.0
        hy = grid.zeros(grid.nx - 1, grid.ny)
        hy[7, 10] = 1.0
        self.h_masks = {"src_hy_mask": hy}

    def amplitude(self, t_idx):
        return 1.0

    def h_amplitude(self, t_idx):
        return -1.0


def test_dual_source_drives_both_fields():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    state, _ = run_tm(init_tm_state(grid), m, 3, source=DualSource(grid))
    assert state.Hy[7, 10].abs() > 0.0
    assert state.Dz[8, 10].abs() > 0.0


def test_dual_source_checkpointed_matches_plain():
    grid = Grid(nx=20, ny=20, dx=0.1)
    m = build_tm_coeffs(grid, eps_inf=1.0, npml=0)
    s1, _ = run_tm(init_tm_state(grid), m, 8, source=DualSource(grid))
    s2, _ = run_tm(init_tm_state(grid), m, 8, source=DualSource(grid), checkpoint_every=3)
    assert torch.allclose(s1.Ez, s2.Ez)