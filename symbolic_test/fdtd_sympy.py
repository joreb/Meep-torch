# SPDX-License-Identifier: MIT
# 2D TMz FDTD whose update rule is DERIVED in sympy rather than written by
# hand, then compiled to TORCH by lambdify, run on the straight-waveguide
# scenario from scenario.py. The third symbolic backend next to fdtd_devito
# (symbolic PDE -> C/JIT) and fdtd_pylops (sparse LinearOperator).
#
# The torch backend is what makes this more than a readable oracle: sympy 1.14
# ships a TorchPrinter, so `lambdify(args, expr, "torch")` emits torch ops. The
# stepper therefore runs on the GPU AND is differentiable end to end -- grad
# flows through the lambdified expressions like any other torch graph, so
# unlike fdtd_pylops_2d and fdtd_devito_2d this backend needs NO hand-written
# adjoint and no autograd.Function at all. The loop below is deliberately
# out-of-place (F.pad instead of slice assignment) to keep that graph intact;
# the padded boundary rows are exactly the ones the scheme never writes, so it
# is equivalent, not an approximation.
#
# Why it exists: it is the stencil oracle. devito and pylops each hide their
# discretisation behind a layer (codegen / kron products), so when one of them
# diverges there is nothing to diff against. Here the finite differences come
# out of sympy's own `Derivative.as_finite_difference`, the update equations
# are printed-readable sympy expressions, and `lambdify` turns them into torch
# ops. Whatever the other two backends do, THIS is what the scheme is
# supposed to be.
#
# It earned that role immediately: the devito straight-waveguide diverged
# (max|Ez| 2.1e+82 after 400 steps) because its E update read H at time n
# instead of the n+1/2 the same step had just written. The Yee leapfrog below
# makes that ordering explicit and unmissable -- H is advanced, THEN E reads
# the advanced H.
#
# Scheme (mu=1, TMz), matching fdtd_devito's:
#   Hx^{n+1/2} = damp*Hx - dt * dEz/dy          forward difference in y
#   Hy^{n+1/2} = damp*Hy + dt * dEz/dx          forward difference in x
#   Ez^{n+1}   = damp*Ez + (dt/eps) * (dHy/dx - dHx/dy) - (dt/eps)*J
#                                               BACKWARD differences, on H^{n+1/2}
# PML: the same quadratic conductive absorber fdtd_devito uses (not meep's
# CPML recursion), so the gate against the meep_torch reference is field
# correlation, not bitwise.
import math

import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F

try:                       # package import (pytest, -m) ...
    from . import scenario as S
    from .fdtd_devito import _pml_sigma
except ImportError:        # ... or plain script run (path invocation)
    import scenario as S
    from fdtd_devito import _pml_sigma


def _stencils():
    """Derive the three update expressions with sympy.

    The differences are not typed out: sympy builds them from the derivative
    itself via `as_finite_difference`, choosing the sample points. Passing
    [v, v+h] yields the forward difference (f(v+h)-f(v))/h and [v-h, v] the
    backward one -- which is exactly the Yee stagger, stated as a choice of
    sample points rather than as an index shift buried in a slice.

    Returns three lambdified callables plus the sympy expressions (kept so the
    scheme can be printed and diffed against another backend's stencil)."""
    h, dt = sp.symbols("h dt", positive=True)
    eps, damp = sp.symbols("eps damp", positive=True)
    v = sp.Symbol("v")
    f = sp.Function("f")

    fwd = sp.Derivative(f(v), v).as_finite_difference([v, v + h])
    bwd = sp.Derivative(f(v), v).as_finite_difference([v - h, v])

    def at(expr, center, plus=None, minus=None):
        """Bind sympy's f(v), f(v+h), f(v-h) to named neighbour symbols."""
        sub = {f(v): center}
        if plus is not None:
            sub[f(v + h)] = plus
        if minus is not None:
            sub[f(v - h)] = minus
        return expr.subs(sub)

    # neighbour symbols: _yp / _xp are +1 in y / x, _ym / _xm are -1
    Ez, Ez_xp, Ez_yp = sp.symbols("Ez Ez_xp Ez_yp")
    Hx, Hx_ym, Hy, Hy_xm = sp.symbols("Hx Hx_ym Hy Hy_xm")
    J = sp.Symbol("J")

    dEz_dy = at(fwd, Ez, plus=Ez_yp)
    dEz_dx = at(fwd, Ez, plus=Ez_xp)
    dHy_dx = at(bwd, Hy, minus=Hy_xm)
    dHx_dy = at(bwd, Hx, minus=Hx_ym)

    hx_new = sp.simplify(damp * Hx - dt * dEz_dy)
    hy_new = sp.simplify(damp * Hy + dt * dEz_dx)
    # J enters as -(dt/eps)*J: meep's step_source SUBTRACTS the current
    # (dD/dt = curl H - J), the same sign scenario.source_amp assumes.
    ez_new = sp.simplify(damp * Ez + (dt / eps) * (dHy_dx - dHx_dy) - (dt / eps) * J)

    args_h = (Ez, Ez_xp, Ez_yp, Hx, Hy, damp, dt, h)
    args_e = (Ez, Hx, Hx_ym, Hy, Hy_xm, eps, damp, dt, h, J)
    # "torch" backend: sympy emits torch ops, so the compiled stencil runs on
    # whatever device the tensors live on and stays inside autograd's graph.
    return (
        sp.lambdify(args_h, hx_new, "torch"),
        sp.lambdify(args_h, hy_new, "torch"),
        sp.lambdify(args_e, ez_new, "torch"),
        {"Hx": hx_new, "Hy": hy_new, "Ez": ez_new},
    )


def step_sympy(ez, hx, hy, eps, damp, amp_mask, dt, dx, fns=None):
    """One Yee step with the sympy-compiled torch stencil. Pure function of the
    state: no slice assignment anywhere, so the whole trajectory stays
    differentiable w.r.t. eps (or anything else upstream).

    The F.pad calls restore the boundary entries the scheme never writes --
    hx's last column, hy's last row, ez's first row and column -- which the
    update leaves at zero for all time, so padding with zero is exact."""
    f_hx, f_hy, f_ez = fns if fns is not None else _stencils()[:3]
    # H first (n-1/2 -> n+1/2) from Ez at n
    hx = F.pad(f_hx(ez[:, :-1], 0.0, ez[:, 1:], hx[:, :-1], 0.0,
                    damp[:, :-1], dt, dx), (0, 1))
    hy = F.pad(f_hy(ez[:-1, :], ez[1:, :], 0.0, 0.0, hy[:-1, :],
                    damp[:-1, :], dt, dx), (0, 0, 0, 1))
    # then E at n+1, reading the H just advanced -- the leapfrog ordering the
    # devito version got wrong (see module docstring)
    ez = F.pad(f_ez(ez[1:, 1:], hx[1:, 1:], hx[1:, :-1],
                    hy[1:, 1:], hy[:-1, 1:],
                    eps[1:, 1:], damp[1:, 1:], dt, dx, amp_mask[1:, 1:]),
               (1, 0, 1, 0))
    return ez, hx, hy


def fields(device: str = "cpu", eps=None):
    """Grid constants for the scenario as torch tensors on `device`.

    `eps` may be passed in with requires_grad set, which is the point of the
    torch backend -- see test_sympy_is_differentiable."""
    nx, ny, dx = S.NX, S.NY, S.DX
    dt = dx * 0.5                      # scenario courant=0.5
    if eps is None:
        eps = S.eps_field("cpu").to(device)
    sx, sy = _pml_sigma(nx, S.NPML), _pml_sigma(ny, S.NPML)
    damp = torch.from_numpy(np.exp(-(sx[:, None] + sy[None, :]) * dt)).to(device)
    mask = torch.zeros(nx, ny, dtype=torch.float64, device=device)
    mask[int(round((S.SRC_X - S.X_LO) / dx)), ny // 2] = 1.0
    zeros = lambda: torch.zeros(nx, ny, dtype=torch.float64, device=device)
    return dict(ez=zeros(), hx=zeros(), hy=zeros(), eps=eps, damp=damp,
                mask=mask, dt=dt, dx=dx, n=S.n_steps(dt))


def run_sympy(gpu: bool = False, n_steps: int | None = None,
              eps=None) -> np.ndarray:
    """Run the straight-waveguide with the sympy-derived torch stencil.

    Returns Ez (NX, NY) as numpy, like run_devito / run_pylops. `gpu` selects
    the device -- unlike the other two backends this one needs no cupy and no
    nvc, it is just torch."""
    device = "cuda" if gpu else "cpu"
    g = fields(device, eps=eps)
    fns = _stencils()[:3]
    ez, hx, hy = g["ez"], g["hx"], g["hy"]
    for t in range(n_steps if n_steps is not None else g["n"]):
        amp = math.cos(2.0 * math.pi * S.FREQ * (t * g["dt"]))
        ez, hx, hy = step_sympy(ez, hx, hy, g["eps"], g["damp"],
                                amp * g["mask"], g["dt"], g["dx"], fns)
    return ez.detach().cpu().numpy()


def print_scheme() -> None:
    """Print the derived update equations -- the point of a symbolic backend."""
    *_, exprs = _stencils()
    for name, e in exprs.items():
        print(f"{name}^new = {sp.simplify(e)}")


if __name__ == "__main__":
    print_scheme()
    ez = run_sympy()
    print(f"max|Ez| = {np.max(np.abs(ez)):.6e}")
