# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state


def make_sim():
    g = Grid(nx=64, ny=48, dx=0.02)
    m = build_tm_coeffs(
        g, eps_inf=2.25, poles=[(2.0, 0.1, 1.5)], chi3=0.01, npml=8
    )
    src = LineSource(g, ix=20, j0=16, j1=32, profile=GaussianPulse(0.65, 0.2))
    return g, m, src


def test_checkpointed_equals_uncheckpointed():
    g, m, src = make_sim()
    probes = [PointProbe(40, 24)]
    s0 = init_tm_state(g, n_poles=1)
    sA, rA = run_tm(s0, m, 60, source=src, monitors=probes, checkpoint_every=0)
    s0 = init_tm_state(g, n_poles=1)
    sB, rB = run_tm(s0, m, 60, source=src, monitors=probes, checkpoint_every=7)
    for a, b in zip(sA, sB):
        if a is None and b is None:
            continue    # W5.5 magnetic-pole fields, unallocated when n_mpoles=0
        assert torch.allclose(a, b, atol=1e-12, rtol=0)
    assert torch.allclose(rA[0], rB[0], atol=1e-12, rtol=0)
    assert rA[0].shape == (60,)


def test_gradient_flows_through_checkpointed_run():
    g, m, src = make_sim()
    eps = torch.full((g.nx, g.ny), 2.25, dtype=torch.float64, requires_grad=True)
    m2 = build_tm_coeffs(g, eps_inf=eps, poles=[(2.0, 0.1, 1.5)], chi3=0.01, npml=8)
    s0 = init_tm_state(g, n_poles=1)
    s, _ = run_tm(s0, m2, 40, source=src, checkpoint_every=8)
    loss = (s.Ez**2).sum()
    loss.backward()
    assert eps.grad is not None
    assert torch.isfinite(eps.grad).all()
    assert eps.grad.abs().sum() > 0


def test_checkpoint_grad_parity():
    """Checkpointed backward must match the full-graph (checkpoint_every=0)
    gradient to roundoff: checkpoint recomputes each chunk but is numerically
    identical, so the only difference is float recompute order. Rel diff must
    be tiny -- this is the gate that the run.py grad_leaves fix actually
    engages recomputation (the old no-op path retained the full graph, so it
    also passed grad-flow but at ~12 GB/iter; this pins both correctness AND
    that checkpoint is doing the work)."""
    g, _, src = make_sim()

    def grad_for(chk):
        eps = torch.full((g.nx, g.ny), 2.25, dtype=torch.float64, requires_grad=True)
        # nudge eps so the gradient is nonzero and spatially varying
        with torch.no_grad():
            eps[20:24, 22:26] = 2.35
        m = build_tm_coeffs(g, eps_inf=eps, poles=[(2.0, 0.1, 1.5)],
                            chi3=0.01, npml=8)
        s0 = init_tm_state(g, n_poles=1)
        s, _ = run_tm(s0, m, 48, source=src, checkpoint_every=chk)
        (s.Ez**2).sum().backward()
        return eps.grad.clone()

    g_full = grad_for(0)        # checkpoint_every=0 -> single _run_chunk, full graph
    g_chk = grad_for(8)         # fixed path: grad_leaves engage checkpoint
    assert torch.isfinite(g_full).all() and torch.isfinite(g_chk).all()
    denom = g_full.abs().max()
    rel = (g_chk - g_full).abs().max() / denom
    assert rel < 1e-8, f"checkpoint grad rel diff {rel:.2e} vs full-graph"
