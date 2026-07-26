import torch
from torch.utils.checkpoint import checkpoint

from .coeffs import TMCoeffs
from .stepper import TMState, step_tm


def _run_chunk(state: TMState, coeffs: TMCoeffs, step0: int, k: int, source, monitors):
    partials = [[] for _ in monitors]
    for s in range(k):
        t_idx = step0 + s
        if source is not None:
            state = step_tm(state, coeffs, source.amplitude(t_idx), source.mask)
        else:
            state = step_tm(state, coeffs)
        for i, mon in enumerate(monitors):
            partials[i].append(mon.sample(state, t_idx))
    outs = []
    for mon, ps in zip(monitors, partials):
        stacked = torch.stack(ps)
        outs.append(stacked.sum(0) if mon.reduce == "sum" else stacked)
    return state, outs


def run_tm(
    state: TMState,
    coeffs: TMCoeffs,
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
        state, outs = _run_chunk(state, coeffs, 0, n_steps, source, monitors)
        absorb(outs)
        return state, acc

    n_tensors = len(state)

    def chunk_fn(step0, k, *tensors):
        st, outs = _run_chunk(TMState(*tensors), coeffs, step0, k, source, monitors)
        return (*st, *outs)

    step0 = 0
    while step0 < n_steps:
        k = min(checkpoint_every, n_steps - step0)
        flat = checkpoint(chunk_fn, step0, k, *state, use_reentrant=False)
        state = TMState(*flat[:n_tensors])
        absorb(list(flat[n_tensors:]))
        step0 += k
    return state, acc
