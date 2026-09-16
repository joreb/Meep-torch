"""A generic-gather TM stepper: same physics as core.stepper_tm, spec-driven.

This exists to answer one question the existing benchmarks could not. Comparing
`step_tm` (hand-sliced, 2D) against `step_3d` (axis-generic, 3D) confounds code
style with arithmetic: 3D evaluates two curl terms per component where TM
evaluates 4 differences across 3 fields, so 3D does ~1.5x the work per scalar
updated and the "Mscalars/s" normalisation does not correct for it. The measured
1.1-1.47 gap is therefore an upper bound on genericity that the physics alone
could fully explain.

So: same physics, same dimensionality, same field count, same CPML, same phase
order -- only the GATHER differs. Hand slices become axis-generic slicing driven
by the Yee offsets, which is the style a unified spec-driven stepper would emit.

Not a production stepper. No folds, no PMC, no poles, no Kerr, no sources: those
are orthogonal to the gather and would only add unmatched work to one side.
Correctness is established by agreeing with `step_tm` on the same inputs, which
`measure()` checks before it times anything.
"""
from typing import NamedTuple

import torch

from meep_torch.core.symbolic import HALF, YEE, PSI_NEW, STRETCHED

_AXES = ("x", "y")


class GState(NamedTuple):
    Ez: torch.Tensor
    Hx: torch.Tensor
    Hy: torch.Tensor
    psi_Ezy: torch.Tensor
    psi_Ezx: torch.Tensor
    psi_Hyx: torch.Tensor
    psi_Hxy: torch.Tensor


def _shape(comp, n, walled=(True, True)):
    """Field shape from the Yee offsets: short by one on a walled axis the
    component is half-offset along. The rule the allocator survey derived."""
    off = YEE[comp]
    return tuple(n[a] - 1 if (off[a] == HALF and walled[a]) else n[a]
                 for a in range(len(n)))


def init(nx, ny, dtype=torch.float64, device="cpu"):
    n = (nx, ny)
    z = lambda s: torch.zeros(*s, dtype=dtype, device=device)
    return GState(
        Ez=z(_shape("Ez", n)), Hx=z(_shape("Hx", n)), Hy=z(_shape("Hy", n)),
        psi_Ezy=z(_shape("Hx", n)), psi_Ezx=z(_shape("Hy", n)),
        psi_Hyx=z((nx - 2, ny - 2)), psi_Hxy=z((nx - 2, ny - 2)))


def _diff(v, axis, h):
    """Difference along ``axis``, axis-generic (cf. stepper_3d._d)."""
    lo = [slice(None)] * v.dim()
    hi = [slice(None)] * v.dim()
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    return (v[tuple(hi)] - v[tuple(lo)]) / h


def _interior(v, axis):
    sl = [slice(None)] * v.dim()
    sl[axis] = slice(1, -1)
    return v[tuple(sl)]


def _pad(v, axes):
    pad = [0] * (2 * v.dim())
    for a in axes:                       # torch pads last axis first
        i = 2 * (v.dim() - 1 - a)
        pad[i] = pad[i + 1] = 1
    return torch.nn.functional.pad(v, pad)


def step(s, m):
    """One full TM step, phases in meep's order, gathers driven by the offsets."""
    dt, dx = m.dt, m.dx

    # --- B phase: H from curl E. One term per H component; which axis each
    # differentiates along is the axis it is NOT offset on (Yee), so it is
    # read off the table rather than written down.
    curls, psis = {}, {}
    for comp, psi_name, coef in (("Hx", "psi_Ezy", ("b_hy", "a_hy", "k_hy")),
                                 ("Hy", "psi_Ezx", ("b_hx", "a_hx", "k_hx"))):
        axis = next(a for a in range(2) if YEE[comp][a] == HALF)
        d = _diff(s.Ez, axis, dx)
        p = PSI_NEW(d, getattr(s, psi_name), getattr(m, coef[0]), getattr(m, coef[1]))
        curls[comp], psis[psi_name] = STRETCHED(d, p, getattr(m, coef[2])), p
    Hx = s.Hx - dt * curls["Hx"]
    Hy = s.Hy + dt * curls["Hy"]

    # --- D phase: Ez from curl H. Two terms, each differentiating the
    # component along the axis Ez is collocated with it on.
    dHy = _interior(_diff(Hy, 0, dx), 1)
    dHx = _interior(_diff(Hx, 1, dx), 0)
    psi_Hyx = PSI_NEW(dHy, s.psi_Hyx, m.b_ex, m.a_ex)
    psi_Hxy = PSI_NEW(dHx, s.psi_Hxy, m.b_ey, m.a_ey)
    curl = STRETCHED(dHy, psi_Hyx, m.k_ex) - dHx / m.k_ey - psi_Hxy
    Ez = s.Ez + dt * _pad(curl, (0, 1))

    return GState._make([Ez, Hx, Hy, psis["psi_Ezy"], psis["psi_Ezx"],
                         psi_Hyx, psi_Hxy])
