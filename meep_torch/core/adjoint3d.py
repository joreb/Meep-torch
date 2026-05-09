# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Checkpointed 3D adjoint: gradients through the functional ``step_3d``
with chunk-boundary states offloaded to CPU (docs/plans/2026-08-18).

Two-phase manual scheme (NOT torch.utils.checkpoint -- its boundary
states stay on the GPU and accumulate O(n_chunks x state), the known
3D-adjoint memory wall):

1. no_grad forward, saving the full ``State3D`` at chunk starts to CPU
   and accumulating monitor DFTs as plain tensors;
2. loss head on re-leafed accumulators -> per-monitor cotangents;
3. reverse sweep: per chunk (last first) restore the start state as
   leaves, re-run WITH graph incl. monitor samples, one backward seeded
   with the younger chunk's state cotangent + the accumulator
   cotangents. eps gradients flow through ``Coeffs3D.eps`` (``step_3d``
   averages eps in-step) and accumulate on the eps leaf across chunks.

GPU peak = one chunk's graph. FD-gated in
tests/examples/test_adjoint3d.py (single-application objective).
"""
import torch

from .stepper_3d import State3D, step_3d

__all__ = ["run_3d_adjoint"]


def _state_to_cpu(s: State3D):
    return tuple(t.detach().to("cpu", copy=True) if t is not None else None
                 for t in s)


def _state_leaves(saved, device):
    return [t.to(device).requires_grad_(True) if t is not None else None
            for t in saved]


@torch.no_grad()
def _forward_chunks(c, n_steps, chunk, source, monitors, state0):
    device = c.eps.device
    boundaries = [_state_to_cpu(state0)]
    accs = [None] * len(monitors)
    state = state0
    for t in range(n_steps):
        amp = source.amplitude(t) if source is not None else 0.0
        state = step_3d(state, c, amp, source.src if source is not None else None)
        for i, mon in enumerate(monitors):
            smp = mon.sample(state, t)
            accs[i] = smp if accs[i] is None else accs[i] + smp
        if (t + 1) % chunk == 0 and (t + 1) < n_steps:
            boundaries.append(_state_to_cpu(state))
    return boundaries, accs


def run_3d_adjoint(c, n_steps: int, chunk: int, source, monitors, loss_fn,
                   state0: State3D):
    """Returns (loss_value, accs). Gradients accumulate on any leaf the
    graph reaches -- in practice the eps tensor inside ``c`` (pass eps
    with requires_grad=True into build_3d_coeffs). ``loss_fn(accs)``
    must return a real scalar built from the monitor accumulators."""
    device = c.eps.device
    boundaries, accs = _forward_chunks(c, n_steps, chunk, source, monitors,
                                       state0)

    accs_leaf = [a.detach().clone().requires_grad_(True) for a in accs]
    L = loss_fn(accs_leaf)
    g_accs = torch.autograd.grad(L, accs_leaf)

    n_chunks = len(boundaries)
    cot = None                                    # state cotangent, per tensor
    for k in range(n_chunks - 1, -1, -1):
        t0 = k * chunk
        t1 = min((k + 1) * chunk, n_steps)
        leaves = _state_leaves(boundaries[k], device)
        state = State3D(*leaves)
        chunk_accs = [None] * len(monitors)
        with torch.enable_grad():
            for t in range(t0, t1):
                amp = source.amplitude(t) if source is not None else 0.0
                state = step_3d(state, c, amp,
                                source.src if source is not None else None)
                for i, mon in enumerate(monitors):
                    smp = mon.sample(state, t)
                    chunk_accs[i] = (smp if chunk_accs[i] is None
                                     else chunk_accs[i] + smp)
        outputs, grads = [], []
        if cot is not None:
            for t_out, g in zip(state, cot):
                if t_out is not None and g is not None and t_out.requires_grad:
                    outputs.append(t_out)
                    grads.append(g.to(device))
        for ca, ga in zip(chunk_accs, g_accs):
            if ca is not None and ca.requires_grad:
                outputs.append(ca)
                grads.append(ga.to(device, dtype=ca.dtype))
        torch.autograd.backward(outputs, grads)
        cot = [lf.grad.detach().to("cpu", copy=True) if (lf is not None and
               lf.grad is not None) else None for lf in leaves]
        del leaves, state, chunk_accs, outputs, grads
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return float(L), accs
