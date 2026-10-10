# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import torch
from torch.utils.checkpoint import checkpoint

from .stepper_tm import TMState, step_tm
from .stepper_te import TEState, step_te


def _nonfinite(state):
    """meep's divergence check, deferred rather than dropped.

    `src/step.cpp:138` reads ONE point every step -- the energy density at the
    cell centre -- and aborts: "simulation fields are NaN or Inf". Without it a
    diverged run returns NaN arrays and every flux, DFT and printed number
    downstream inherits them with nothing raising. meep and meep-torch diverge
    on the same inputs (SiO2's Sellmeier poles against the ADE stability limit
    at coarse dt, measured); only one of them says so.

    ponytail: checked at chunk boundaries and at run end, not per step -- a
    per-step `.item()` serialises the CUDA pipeline. The cost is that we name
    the chunk where it happened where meep names the step.
    """
    with torch.no_grad():
        probes = [x.detach().reshape(-1)[x.numel() // 2]
                  for x in state if torch.is_tensor(x) and x.numel()]
        if not probes:
            return False
        # every carried field, not just the first: a divergence can start in
        # the ADE polarizations or a CPML memory and reach E a few steps later
        return not bool(torch.isfinite(torch.stack(probes)).all())


def _check_finite(state, where=""):
    if _nonfinite(state):
        raise FloatingPointError(
            "simulation fields are NaN or Inf" + (f" ({where})" if where else "")
            + " -- the run diverged; meep aborts on the same condition "
              "(src/step.cpp:138)")


def _integrated(source):
    """True when this source is meep's is_integrated: a dipole, not a current.

    meep's `step_source(D)` defaults `including_integrated = false`
    (`meep.hpp:2261`), so an integrated source contributes NOTHING to D; it is
    subtracted as a polarization at the D-to-E step instead
    (`update_eh.cpp:130`). The two paths algebraically telescope to the same
    expression in a lossless cell and differ by `cndinv` wherever the source
    cell carries loss -- 45 % for a line into PML, measured.
    """
    prof = getattr(source, "profile", None)
    return bool(getattr(prof, "is_integrated", False))


def _step_source(step_fn, state, coeffs, source, t_idx):
    """One step with `source` injected the way its `is_integrated` says.

    `source.mask` is the Dz (TM) / Dy (TE) current and rides the second
    positional slot. `mask_x` is the TE Dx current, carried as a keyword and
    ONLY when the source has one: `step_tm` has no `src_mask_ex`, so passing
    it unconditionally would break every TM run.

    This is the single-source chunk path, which is the path nearly every run
    takes, and it had no route for Dx -- so an `mp.Source(component=mp.Ex)`
    in 2D injected the zeroed Dy mask and the whole cell stayed at zero,
    while `fields.step_source` (the multi-source phase loop) did carry it.
    """
    mask_x = getattr(source, "mask_x", None)
    kw = {"src_mask_ex": mask_x} if mask_x is not None else {}
    h_masks = getattr(source, "h_masks", None)
    if h_masks:
        return step_fn(state, coeffs, source.amplitude(t_idx), source.mask,
                       source.h_amplitude(t_idx), **h_masks, **kw)
    if _integrated(source):
        dip = source.profile.dipole((t_idx + 0.5) * source.dt).real
        return step_fn(state, coeffs, 0.0, source.mask, src_dipole_amp=dip,
                       **kw)
    return step_fn(state, coeffs, source.amplitude(t_idx), source.mask, **kw)


def _check_integrated(step_fn, source):
    import inspect
    if source is None or not _integrated(source):
        return
    if "src_dipole_amp" not in inspect.signature(step_fn).parameters:
        raise NotImplementedError(
            "is_integrated=True needs the source polarization term at the "
            "D-to-E step (meep update_eh.cpp:130); "
            f"{getattr(step_fn, '__name__', step_fn)} does not carry it yet")


def _run_chunk(step_fn, state, coeffs, step0: int, k: int, source, monitors):
    # sum-reduce monitors (DFT accumulators) fold in-place: storing every
    # step's sample first (old torch.stack path) held O(n_steps) copies --
    # tens of GB for long broadband runs -- before summing.
    partials = [None if mon.reduce == "sum" else [] for mon in monitors]
    for s in range(k):
        t_idx = step0 + s
        if source is not None:
            state = _step_source(step_fn, state, coeffs, source, t_idx)
        else:
            state = step_fn(state, coeffs)
        # A whole run is often ONE chunk, so a check only at the boundary is a
        # check at the end -- a diverged run would still burn its full time
        # where meep aborts at the step. 1024 bounds that to one sync per
        # thousand steps.
        if s and not s % 1024:
            _check_finite(state, f"steps {step0}..{t_idx}")
        for i, mon in enumerate(monitors):
            smp = mon.sample(state, t_idx)
            if mon.reduce == "sum":
                partials[i] = smp if partials[i] is None else partials[i] + smp
            else:
                partials[i].append(smp)
    _check_finite(state, f"steps {step0}..{step0 + k - 1}")
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
            _check_integrated(step_fn, source)
            state, outs = _run_chunk(step_fn, state, coeffs, 0, n_steps, source, monitors)
            absorb(outs)
            return state, acc

        # The autograd graph reaches ``coeffs`` through the chunk_fn closure
        # (coeffs is built from eps <- rho), NOT through the ``*state`` inputs
        # (init_tm_state -> grid.zeros: no requires_grad). With only ``*state``
        # passed to checkpoint, use_reentrant=False saw no requires_grad INPUT
        # and silently no-op'd: the full n_steps graph was retained regardless
        # of checkpoint_every (measured: CHKPT=64 ~12 GB/iter, NOT ~1.5 GB).
        # Fix: pass the grad-carrying coeffs leaves as explicit checkpoint args
        # so checkpoint sees a grad input and recomputes each chunk on backward.
        # The closure still supplies the values (same tensor objects -> the
        # graph links); the trailing leaves just tell checkpoint what to save
        # and recompute, collapsing saved-for-backward to chunk-boundary state.
        grad_leaves = tuple(
            t for t in coeffs if torch.is_tensor(t) and t.requires_grad
        )
        n_tensors = len(state)

        if not grad_leaves:
            # forward-only (no autograd graph): plain chunked run with no
            # checkpoint wrapper -- it would no-op without a grad input and
            # only add overhead. Monitor partials still fold per chunk.
            step0 = 0
            while step0 < n_steps:
                k = min(checkpoint_every, n_steps - step0)
                state, outs = _run_chunk(
                    step_fn, state, coeffs, step0, k, source, monitors)
                absorb(outs)
                step0 += k
            return state, acc

        def chunk_fn(step0, k, *tensors):
            # trailing grad_leaves are ignored here -- the closure ``coeffs``
            # holds the same tensor objects and supplies their values; the
            # leaves exist only as checkpoint args to engage recomputation.
            st, outs = _run_chunk(
                step_fn, state_cls(*tensors[:n_tensors]), coeffs,
                step0, k, source, monitors,
            )
            return (*st, *outs)

        step0 = 0
        while step0 < n_steps:
            k = min(checkpoint_every, n_steps - step0)
            # Monitor sampling runs inside the checkpointed region by design:
            # per-chunk partials are O(1)-sized outputs of the chunk, so memory
            # stays bounded; they are recomputed on backward (cheap vs. field state).
            flat = checkpoint(
                chunk_fn, step0, k, *state, *grad_leaves, use_reentrant=False)
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
    _check_integrated(step_fn, source)
    for t in range(max_steps):
        if source is not None:
            state = _step_source(step_fn, state, coeffs, source, t)
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
        # This loop already syncs every step for the decay test, so meep's
        # per-step check (src/step.cpp:138) costs nothing extra here. A NaN
        # probe also silently defeats the decay test itself: every comparison
        # against NaN is False, so decay_run never advances and the run grinds
        # to max_steps instead of stopping.
        if not math.isfinite(v):
            raise FloatingPointError(
                f"simulation fields are NaN or Inf at step {t} -- the run "
                "diverged; meep aborts on the same condition "
                "(src/step.cpp:138)")
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
