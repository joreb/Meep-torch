import torch
from torch.utils.checkpoint import checkpoint

from .stepper import TMState, step_tm
from .stepper_te import TEState, step_te


def _run_chunk(step_fn, state, coeffs, step0: int, k: int, source, monitors):
    partials = [[] for _ in monitors]
    for s in range(k):
        t_idx = step0 + s
        if source is not None:
            state = step_fn(state, coeffs, source.amplitude(t_idx), source.mask)
        else:
            state = step_fn(state, coeffs)
        for i, mon in enumerate(monitors):
            partials[i].append(mon.sample(state, t_idx))
    outs = []
    for mon, ps in zip(monitors, partials):
        stacked = torch.stack(ps)
        outs.append(stacked.sum(0) if mon.reduce == "sum" else stacked)
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
            flat = checkpoint(chunk_fn, step0, k, *state, use_reentrant=False)
            state = state_cls(*flat[:n_tensors])
            absorb(list(flat[n_tensors:]))
            step0 += k
        return state, acc

    return run


run_tm = _make_runner(step_tm, TMState)
run_te = _make_runner(step_te, TEState)
