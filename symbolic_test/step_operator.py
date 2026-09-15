# SPDX-License-Identifier: MIT
# The dense one-step operator of a linear stepper, extracted column by column
# from unit states. numpy and torch only: the stepper is a parameter, so this
# imports nothing from meep_torch and serves core's step_tm, stepper_sympy's
# step_tm_sympy, or anything else with a NamedTuple state.
#
# Why dense: N^2 memory, so small grids only (12x12 TM with PML is n ~ 900).
# Spectra, adjoint defects, symmetry commutators R A R^T - A, and bitwise
# comparison of two steppers all want the matrix, not the action.
import numpy as np
import torch


def extract(step, state, fields, tied=None, seed=0, rtol=1e-13):
    """``(A, pack, unpack, index)`` with ``A @ pack(s) == pack(step(s))``.

    ``fields``: state attribute names to pack, in order. ``tied``: attributes
    set equal to another packed field on unpack, e.g. ``{"Dz": "Ez"}`` in
    vacuum, and asserted to stay equal after a step so the packed state is
    closed under it. ``index`` maps name -> (offset, shape).

    Asserts, because each one has caught a wrong extraction before: the zero
    state stays zero (else the step is affine and a column is off by a
    constant), the tied fields stay tied, and ``A v`` reproduces ``step(v)``
    on a random ``v`` (else a field the step writes is not being packed).
    """
    tied = tied or {}
    index, n = {}, 0
    for name in fields:
        shape = tuple(getattr(state, name).shape)
        index[name] = (n, shape)
        n += int(np.prod(shape))
    dtype = getattr(state, fields[0]).dtype

    def pack(s):
        return np.concatenate([getattr(s, c).detach().cpu().numpy().ravel()
                               for c in fields])

    def unpack(v):
        f = {}
        for c, (off, shape) in index.items():
            f[c] = torch.tensor(v[off:off + int(np.prod(shape))].reshape(shape),
                                dtype=dtype)
        f.update({t: f[src].clone() for t, src in tied.items()})
        return state._replace(**f)

    assert np.abs(pack(step(unpack(np.zeros(n))))).max() == 0.0, \
        "the step moves the zero state: not linear"
    v = np.random.default_rng(seed).standard_normal(n)
    stepped = step(unpack(v))
    for t, src in tied.items():
        assert torch.equal(getattr(stepped, t), getattr(stepped, src)), \
            f"{t} left the {t} == {src} subspace"
    A = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        A[:, k] = pack(step(unpack(e)))
    want = pack(stepped)
    err = np.abs(A @ v - want).max() / max(np.abs(want).max(), 1e-300)
    assert err < rtol, f"A v != step(v): rel {err:.1e} (a written field is not packed?)"
    return A, pack, unpack, index
