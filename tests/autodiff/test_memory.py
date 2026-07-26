import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state


def forward_saved_count(checkpoint_every: int) -> tuple[int, torch.Tensor, torch.Tensor]:
    g = Grid(nx=128, ny=128, dx=0.02, dtype=torch.float32)
    eps = torch.full((128, 128), 2.0, requires_grad=True)
    m = build_tm_coeffs(g, eps_inf=eps, npml=8)
    src = LineSource(g, ix=20, j0=30, j1=98, profile=GaussianPulse(0.65, 0.2))
    s = init_tm_state(g)
    count = 0

    def pack(x):
        nonlocal count
        count += 1
        return x

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        s, _ = run_tm(s, m, 200, source=src, checkpoint_every=checkpoint_every)
        loss = (s.Ez**2).sum()
    return count, loss, eps


@pytest.mark.slow
def test_checkpointing_bounds_memory():
    # Approved deviation from the task brief: the original RSS-delta metric is
    # defective on CPU -- torch's caching allocator reuses freed blocks and
    # saved activations are references into shared storage, so the
    # uncheckpointed forward peaked at ~380 MB RSS (not the predicted O(GB))
    # and the 3x RSS margin was unreachable (measured in-process, 384x384:
    # 171.0 vs 171.4 MB; 512x512: 258.3 vs 286.3 MB).
    #
    # Instead we count saved activation tensors during the forward pass, the
    # structural quantity checkpointing actually bounds: an uncheckpointed
    # forward saves O(n_steps) activation tensors for backward, while a
    # checkpointed one saves only chunk-boundary inputs, O(n_steps / chunk).
    # Per-chunk recomputation happens at backward time and does not count
    # during forward. Measured: 101 (checkpointed, every=20) vs 3379 (full).
    count_ckpt, loss_ckpt, eps_ckpt = forward_saved_count(checkpoint_every=20)
    count_full, loss_full, eps_full = forward_saved_count(checkpoint_every=0)
    assert count_ckpt < count_full / 5, (count_ckpt, count_full)
    loss_ckpt.backward()
    loss_full.backward()
    assert eps_full.grad.abs().sum() > 0  # guard: equality below is not vacuous
    assert torch.allclose(eps_ckpt.grad, eps_full.grad, rtol=1e-4, atol=1e-6)
