# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch
from torch.utils.checkpoint import checkpoint

from .stepper import TMState, step_tm
from .stepper_te import TEState, step_te


def _run_chunk(step_fn, state, coeffs, step0: int, k: int, source, monitors):
    # sum-reduce monitors (DFT accumulators) fold in-place: storing every
    # step's sample first (old torch.stack path) held O(n_steps) copies --
    # tens of GB for long broadband runs -- before summing.
    partials = [None if mon.reduce == "sum" else [] for mon in monitors]
    for s in range(k):
        t_idx = step0 + s
        if source is not None:
            h_masks = getattr(source, "h_masks", None)
            if h_masks:
                state = step_fn(
                    state, coeffs, source.amplitude(t_idx), source.mask,
                    source.h_amplitude(t_idx), **h_masks,
                )
            else:
                state = step_fn(state, coeffs, source.amplitude(t_idx), source.mask)
        else:
            state = step_fn(state, coeffs)
        for i, mon in enumerate(monitors):
            smp = mon.sample(state, t_idx)
            if mon.reduce == "sum":
                partials[i] = smp if partials[i] is None else partials[i] + smp
            else:
                partials[i].append(smp)
    outs = []
    for mon, ps in zip(monitors, partials):
        if mon.reduce == "sum":
            outs.append(ps)
        else:
            outs.append(torch.stack(ps))
    return state, outs


def _make_runner(step_fn, state_cls):
    def run(
        state,
        coeffs,
        n_steps: int,
        *,
        source=None,
        monitors=(),
        checkpoint_every: int = 0,
    ):
        monitors = list(monitors)
        acc = [None] * len(monitors)

        def absorb(outs):
            for i, (mon, out) in enumerate(zip(monitors, outs)):
                if acc[i] is None:
                    acc[i] = out
                elif mon.reduce == "sum":
                    acc[i] = acc[i] + out
                else:
                    acc[i] = torch.cat([acc[i], out])

        if checkpoint_every <= 0:
            state, outs = _run_chunk(step_fn, state, coeffs, 0, n_steps, source, monitors)
            absorb(outs)
            return state, acc

        n_tensors = len(state)

        def chunk_fn(step0, k, *tensors):
            st, outs = _run_chunk(
                step_fn, state_cls(*tensors), coeffs, step0, k, source, monitors
            )
            return (*st, *outs)

        step0 = 0
        while step0 < n_steps:
            k = min(checkpoint_every, n_steps - step0)
            # Monitor sampling runs inside the checkpointed region by design:
            # per-chunk partials are O(1)-sized outputs of the chunk, so memory
            # stays bounded; they are recomputed on backward (cheap vs. field state).
            flat = checkpoint(chunk_fn, step0, k, *state, use_reentrant=False)
            state = state_cls(*flat[:n_tensors])
            absorb(list(flat[n_tensors:]))
            step0 += k
        return state, acc

    return run


run_tm = _make_runner(step_tm, TMState)
run_te = _make_runner(step_te, TEState)


def run_until_decayed(step_fn, state, coeffs, source, monitors, probe_fn,
                      decay_ratio: float = 1e-6, n_decay: int = 50,
                      max_steps: int = 100000):
    """Step until |probe_fn(state)|^2 decays by ``decay_ratio`` below its peak
    for ``n_decay`` consecutive steps (meep stop_when_fields_decayed). Returns
    (state, monitor_outputs). No checkpointing (decay runs are short).
    """
    monitors = list(monitors)
    acc = [None] * len(monitors)
    peak = 0.0
    decay_run = 0
    for t in range(max_steps):
        h_masks = getattr(source, "h_masks", None) if source is not None else None
        if h_masks:
            state = step_fn(state, coeffs, source.amplitude(t), source.mask,
                            source.h_amplitude(t), **h_masks)
        elif source is not None:
            state = step_fn(state, coeffs, source.amplitude(t), source.mask)
        else:
            state = step_fn(state, coeffs)
        outs = [mon.sample(state, t) for mon in monitors]
        for i, (mon, out) in enumerate(zip(monitors, outs)):
            if acc[i] is None:
                acc[i] = out
            elif mon.reduce == "sum":
                acc[i] = acc[i] + out
            else:
                acc[i] = torch.cat([acc[i], out])
        v = float(probe_fn(state))
        if v > peak:
            peak = v
            decay_run = 0
        elif peak > 0 and v < decay_ratio * peak:
            decay_run += 1
            if decay_run >= n_decay:
                break
        else:
            decay_run = 0
    return state, acc
