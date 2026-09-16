"""The same generic gather, with the SPEC resolved at build time.

`generic_tm.step` was generic at RUN time: every call re-read the Yee table,
compared sympy Rationals to pick an axis, looked coefficients up by string, and
rebuilt slice tuples. That is spec work, and it belongs in a factory. Measured,
the run-time-generic version was ~1.5x (CUDA) / ~1.95x (CPU) slower than hand
slices after torch.compile -- and the question that number could not answer was
how much of it is genericity and how much is the spec being re-evaluated
2 million times.

So: `make_step(spec, m)` does the spec work ONCE and returns a closure over
plain ints, slice objects and bound tensors. The returned function contains
tensor operations and nothing else. This is partial evaluation -- the cheap
half of "emit specialised code" -- and it needs no codegen.
"""
import torch

from meep_torch.core.symbolic import HALF, YEE, PSI_NEW, STRETCHED

from .generic_tm import GState, init  # same state, same allocator


def _slices(ndim, axis, lo, hi):
    s = [slice(None)] * ndim
    s[axis] = slice(lo, hi)
    return tuple(s)


def make_step(m, ndim=2):
    """Resolve the spec once; return a step function of ``state`` alone."""
    dt, dx = m.dt, m.dx

    # --- spec work, done here and never again -------------------------------
    # which axis each H component differentiates Ez along: the one it is
    # half-offset on. A sympy comparison, evaluated at build time.
    b_axis = {c: next(a for a in range(ndim) if YEE[c][a] == HALF)
              for c in ("Hx", "Hy")}
    lo_b = {c: _slices(ndim, b_axis[c], None, -1) for c in b_axis}
    hi_b = {c: _slices(ndim, b_axis[c], 1, None) for c in b_axis}
    # D phase: Hy differenced along x, Hx along y, each trimmed on the other
    lo_x, hi_x = _slices(ndim, 0, None, -1), _slices(ndim, 0, 1, None)
    lo_y, hi_y = _slices(ndim, 1, None, -1), _slices(ndim, 1, 1, None)
    int_y, int_x = _slices(ndim, 1, 1, -1), _slices(ndim, 0, 1, -1)

    lo_hx, hi_hx = lo_b["Hx"], hi_b["Hx"]
    lo_hy, hi_hy = lo_b["Hy"], hi_b["Hy"]
    b_hy, a_hy, k_hy = m.b_hy, m.a_hy, m.k_hy
    b_hx, a_hx, k_hx = m.b_hx, m.a_hx, m.k_hx
    b_ex, a_ex, k_ex = m.b_ex, m.a_ex, m.k_ex
    b_ey, a_ey, k_ey = m.b_ey, m.a_ey, m.k_ey
    pad = (1, 1, 1, 1)

    def step(s):
        Ez, Hx, Hy = s.Ez, s.Hx, s.Hy
        dEy = (Ez[hi_hx] - Ez[lo_hx]) / dx
        p_ezy = PSI_NEW(dEy, s.psi_Ezy, b_hy, a_hy)
        Hx = Hx - dt * STRETCHED(dEy, p_ezy, k_hy)

        dEx = (Ez[hi_hy] - Ez[lo_hy]) / dx
        p_ezx = PSI_NEW(dEx, s.psi_Ezx, b_hx, a_hx)
        Hy = Hy + dt * STRETCHED(dEx, p_ezx, k_hx)

        dHy = ((Hy[hi_x] - Hy[lo_x]) / dx)[int_y]
        dHx = ((Hx[hi_y] - Hx[lo_y]) / dx)[int_x]
        p_hyx = PSI_NEW(dHy, s.psi_Hyx, b_ex, a_ex)
        p_hxy = PSI_NEW(dHx, s.psi_Hxy, b_ey, a_ey)
        curl = STRETCHED(dHy, p_hyx, k_ex) - dHx / k_ey - p_hxy
        Ez = Ez + dt * torch.nn.functional.pad(curl, pad)
        return GState._make([Ez, Hx, Hy, p_ezy, p_ezx, p_hyx, p_hxy])

    return step
