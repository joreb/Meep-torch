"""A build-time-resolved generic 3D stepper, for the parity measurement.

The 2D answer was "genericity costs nothing provided the spec is resolved at
BUILD time" -- 0.95-0.99x hand at equal work. That was measured on 2D TM only,
and 3D is where the compute actually goes: more components per curl, a
different reduction shape. This is the controlled version of the same
comparison.

Controlled means the same physics, not merely the same problem: the first 2D
attempt compared 2D hand against 3D generic and could not tell code style from
arithmetic. So this mirrors `stepper_3d.step_3d`'s CPML path exactly -- same
phases, same field count, same coefficient tables -- and differs only in where
the slices come from. `check()` asserts bitwise equality before any timing.

Nothing here is typed per component. The twelve derivative terms fall out of
one cyclic rule: H_i takes curl E as (dE_k/dj - dE_j/dk) with (i, j, k)
cyclic, E_i takes curl H the same way, the pad axes of E_i are (j, k), and the
interior trim of each term is on the OTHER axis of that pair.
"""
import torch

from meep_torch.core.stepper_3d import _pad_interior, _pad_pmc, _trim_full
from meep_torch.core.grid import default_device

XYZ = "xyz"
CYCLIC = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]      # (i, j, k)


def _slices(axis, lo, hi):
    s = [slice(None)] * 3
    s[axis] = slice(lo, hi)
    return tuple(s)


def make_step_3d(c, hoist_eps=True):
    """Resolve the spec once; return a step function of ``state`` alone.

    ``hoist_eps=False`` reproduces hand's work exactly -- `step_3d` recomputes
    the three eps averages every step, though eps does not change. Hoisting is
    a genuine win the spec knows about and hand code forgets, so it is measured
    separately rather than folded into the genericity number.
    """
    dx, cp, pmc, damp = c.dx, c.cpml, c.pmc, c.damp
    assert cp is not None, "CPML path only"
    assert c.pols is None, "no poles in the parity measurement"

    lo = {a: _slices(a, None, -1) for a in range(3)}
    hi = {a: _slices(a, 1, None) for a in range(3)}

    # --- the twelve terms, derived once ------------------------------------
    hcurl, ecurl = [], []
    for i, j, k in CYCLIC:
        # H_i <- dE_k/dj - dE_j/dk ; coefficients keyed by the DERIVATIVE axis
        hcurl.append((
            f"H{XYZ[i]}", damp["h" + XYZ[i]],
            (f"psi_E{XYZ[k]}_{XYZ[j]}", f"E{XYZ[k]}", j, cp["h" + XYZ[j]]),
            (f"psi_E{XYZ[j]}_{XYZ[k]}", f"E{XYZ[j]}", k, cp["h" + XYZ[k]]),
        ))
        # E_i <- dH_k/dj - dH_j/dk, each trimmed on the other axis of (j, k)
        ecurl.append((
            f"E{XYZ[i]}", i, damp["e" + XYZ[i]], (j, k),
            (f"psi_H{XYZ[k]}_{XYZ[j]}", f"H{XYZ[k]}", j, cp["e" + XYZ[j]], k),
            (f"psi_H{XYZ[j]}_{XYZ[k]}", f"H{XYZ[j]}", k, cp["e" + XYZ[k]], j),
        ))

    eps_avg = None
    if hoist_eps:
        eps_avg = [_avg(c.eps, i) for i in range(3)]

    def step(s):
        out = {}
        H = {}
        for name, (d1, d2), t1, t2 in hcurl:
            terms = []
            for pname, fname, ax, (b, a, kap) in (t1, t2):
                F = getattr(s, fname)
                d = (F[hi[ax]] - F[lo[ax]]) / dx
                p = b * getattr(s, pname) + a * d
                out[pname] = p
                terms.append(d / kap + p)
            H[name] = d1 * getattr(s, name) - d2 * (terms[0] - terms[1])

        for name, i, (d1, d2), pad_ax, t1, t2 in ecurl:
            terms = []
            for pname, fname, ax, (b, a, kap), fullax in (t1, t2):
                F = H[fname]
                d = _trim_full(F[hi[ax]] - F[lo[ax]], pad_ax, fullax, pmc) / dx
                p = b * getattr(s, pname) + a * d
                out[pname] = p
                terms.append(_pad_pmc(d / kap + p, pad_ax, fullax, pmc))
            curl = _pad_interior(terms[0] - terms[1], pad_ax, pmc)
            e = eps_avg[i] if eps_avg is not None else _avg(c.eps, i)
            out[name] = (d1 * getattr(s, name) * e + d2 * curl) / e

        out.update(H)
        # _make, not _replace: _replace breaks dynamo (torch 2.9.1,
        # side_effects.codegen_save_tempvars) and falls back to eager.
        return type(s)._make([out.get(f, getattr(s, f)) for f in s._fields])

    return step


def _avg(v, axis):
    sl0, sl1 = _slices(axis, None, -1), _slices(axis, 1, None)
    return 0.5 * (v[sl0] + v[sl1])


def check(n=12, npml=3, steps=6, dtype=torch.float64, device=None):
    device = device or default_device()
    """Bitwise against step_3d before anything is timed."""
    from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                            step_3d)
    c = build_3d_coeffs(n, n, n, dx=0.05, npml=npml, dtype=dtype, device=device)
    a = init_3d_state(n, n, n, npml=npml, dtype=dtype, device=device)
    # a non-trivial state, so the psi slots are actually exercised
    torch.manual_seed(0)
    a = type(a)(**{f: (torch.randn_like(v) if v is not None else None)
                   for f, v in zip(a._fields, a)})
    b, fn = a, make_step_3d(c, hoist_eps=False)
    for _ in range(steps):
        a, b = step_3d(a, c), fn(b)
    bad = [f for f in a._fields
           if getattr(a, f) is not None and not torch.equal(getattr(a, f), getattr(b, f))]
    return bad
