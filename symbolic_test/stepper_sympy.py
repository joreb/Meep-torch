# SPDX-License-Identifier: MIT
# Torch steppers compiled from the symbolic Maxwell system in maxwell_sympy.py.
# Imports sympy and torch and NOTHING ELSE -- no meep_torch, no splitter_setup.
#
# Both polarisations come from one derivation. maxwell_sympy reduces the 3D
# six-component system to two decoupled 3x3 blocks; `_block_stencils` compiles
# whichever block it is handed, so TM and TE are the same code path with a
# different component triplet rather than two implementations. That is the
# difference from meep_torch, which carries stepper_tm.py and stepper_te.py
# separately, and from fdtd_pylops_2d / fdtd_devito_2d, which each hand-write a
# TM curl on top of meep_torch's phases.
#
# Two entry points:
#
#   step_tm_sympy(state, m, ...)  -- drop-in for meep_torch's step_tm. It never
#       imports meep_torch: it reads `state` and `m` by ATTRIBUTE (state.Ez,
#       m.b_hy, m.dt) and returns state._replace(...), which is a plain
#       NamedTuple method. So it plugs into _make_runner and optimize_splitter
#       --stepper sympy while this file stays standalone.
#
#   step_block(...)               -- the generic block stepper the above is
#       built from, usable on the native Field6 state for TE or 3D work.
#
# Also exported, and the reason fdtd_pylops_2d / fdtd_devito_2d can drop their
# own meep_torch import: step_source, update_pols, update_eh, derived here
# rather than borrowed from stepper_tm.
#
# Plain branch only, like both sibling drop-ins: no mirrors, no PMC, no
# rotational symmetry, no magnetic poles, no sigma_d, no chi2/chi3. Anything
# else raises instead of silently diverging from meep.
import sympy as sp
import torch
import torch.nn.functional as F

try:                       # package import (pytest, -m) ...
    from .maxwell_sympy import difference, pml_split, polarisation_blocks
except ImportError:        # ... or plain script run (path invocation)
    from maxwell_sympy import difference, pml_split, polarisation_blocks


def _compile_difference(field, axis, target):
    """lambdify the derived finite difference of `field` along `axis` at
    `target`'s location -> f(centre, neighbour, h).

    Which neighbour (the +1 or the -1 sample) is fixed by the Yee offsets in
    maxwell_sympy.difference, not by this function."""
    c, nb, h = sp.symbols("c nb h")
    expr = difference(field, axis, target, h, lambda i: c if i == 0 else nb)
    # together() puts it over the common denominator: (nb - c)/h rather than
    # as_finite_difference's expanded -c/h + nb/h. meep divides the DIFFERENCE
    # once -- `(Ez[:, 1:] - Ez[:, :-1]) / dx` -- and two divisions round
    # differently from one (measured 3.97e-23 on Ez after 3 steps, which is
    # the whole gap between this and a bitwise match).
    return sp.lambdify((c, nb, h), sp.together(expr), "torch")


def _compile_pml():
    """lambdify the derived PML split into its two halves.

    pml_split returns (psi_new, stretched) where stretched still contains
    psi_new; substituting a bare symbol for it means the recursion is
    evaluated once per step and reused, not recomputed inside the curl."""
    d, psi, b, a, k = sp.symbols("d psi b a k")
    psi_sym = sp.Symbol("psi_new")
    psi_new, stretched = pml_split(d, psi, b, a, k)
    return (sp.lambdify((d, psi, b, a), psi_new, "torch"),
            sp.lambdify((d, psi_sym, k), stretched.subs(psi_new, psi_sym), "torch"))


_PSI_NEW, _STRETCHED = _compile_pml()


def _tm_stencils():
    """Compile the TM block's four differences.

    The component triplet is READ from the derivation, not hardcoded: whichever
    block carries E along the invariant axis is TM."""
    blocks = polarisation_blocks(("z",))
    tm = blocks["tm"]["components"]           # ('Ez', 'Hx', 'Hy')
    e, hx, hy = tm
    return {
        # B phase: dEz/dy at Hx, dEz/dx at Hy -- both come out forward
        "dEz_dy": _compile_difference(e, "y", hx),
        "dEz_dx": _compile_difference(e, "x", hy),
        # D phase: dHx/dy and dHy/dx at Ez -- both come out backward
        "dHx_dy": _compile_difference(hx, "y", e),
        "dHy_dx": _compile_difference(hy, "x", e),
    }


_TM = None


def tm_stencils():
    global _TM
    if _TM is None:
        _TM = _tm_stencils()
    return _TM


def _reject_nonplain(m) -> None:
    """Refuse anything outside the plain branch rather than diverge from meep."""
    for attr in ("y_pmc", "y_mirror_lo", "x_mirror_lo", "y_rot2_lo", "rot4_lo"):
        if getattr(m, attr, 0):
            raise NotImplementedError(
                f"sympy stepper: plain branch only ({attr} set)")
    for attr in ("mc1x", "h1x", "d1"):
        if getattr(m, attr, None) is not None:
            raise NotImplementedError(
                f"sympy stepper: magnetic poles / sigma unsupported ({attr})")
    if getattr(m, "has_kerr", False) or getattr(m, "has_chi2", False):
        raise NotImplementedError("sympy stepper: chi2/chi3 unsupported")


# ---------------------------------------------------------------- phases --
# Derived here so the drop-ins do not need meep_torch.core.stepper_tm.

def update_pols(state, m):
    """ADE Lorentz recurrence P(n+1) = c1 P(n) + c2 P(n-1) + c3 Ez.

    With no poles the P arrays are empty and this is a no-op on empty
    tensors, which is what the splitter runs."""
    P_new = m.c1 * state.P_cur + m.c2 * state.P_prev + m.c3 * state.Ez
    return state._replace(P_cur=P_new, P_prev=state.P_cur)


def update_eh(state, m, ft):
    """ft="H" is the identity when B = mu H is folded into the B phase (no
    magnetic poles -- guarded above). ft="E" solves E from D - P."""
    if ft == "H":
        return state
    return state._replace(Ez=(state.Dz - state.P_cur.sum(0)) / m.eps_inf)


def step_source(state, m, ft, amp=0.0, src_mask=None,
                src_hy_mask=None, src_hx_mask=None):
    """Soft current injection. The current is SUBTRACTED: Ampere's law is
    dD/dt = curl H - J, so meep's step.cpp does `f -= current*dt` for magnetic
    B_stuff exactly as for electric D_stuff. Injecting with + gives the right
    magnitude and the wrong polarity, which every quadratic observable (flux,
    energy, |E|^2, Q) is blind to."""
    dt = m.dt
    if ft == "B":
        Hy, Hx = state.Hy, state.Hx
        if src_hy_mask is not None:
            Hy = Hy - dt * amp * src_hy_mask
        if src_hx_mask is not None:
            Hx = Hx - dt * amp * src_hx_mask
        return state._replace(Hx=Hx, Hy=Hy)
    if src_mask is None:
        return state
    return state._replace(Dz=state.Dz - dt * amp * src_mask)


# ------------------------------------------------------------- TM stepper --

def step_db_sympy(state, m, ft):
    """meep step_db with every difference compiled from the symbolic system.

    Shapes follow meep_torch's TMState so this is a drop-in: Ez (nx, ny),
    Hx (nx, ny-1), Hy (nx-1, ny), and the D-phase differences live on the
    interior (nx-2, ny-2) before being padded back to full size."""
    _reject_nonplain(m)
    s, dt, dx = tm_stencils(), m.dt, m.dx
    Ez = state.Ez
    if ft == "B":
        dEz_dy = s["dEz_dy"](Ez[:, :-1], Ez[:, 1:], dx)          # (nx, ny-1)
        psi_Ezy = _PSI_NEW(dEz_dy, state.psi_Ezy, m.b_hy, m.a_hy)
        curl_hx = _STRETCHED(dEz_dy, psi_Ezy, m.k_hy)
        dEz_dx = s["dEz_dx"](Ez[:-1, :], Ez[1:, :], dx)          # (nx-1, ny)
        psi_Ezx = _PSI_NEW(dEz_dx, state.psi_Ezx, m.b_hx, m.a_hx)
        curl_hy = _STRETCHED(dEz_dx, psi_Ezx, m.k_hx)
        return state._replace(psi_Ezy=psi_Ezy, psi_Ezx=psi_Ezx,
                              Hx=state.Hx - dt * curl_hx,
                              Hy=state.Hy + dt * curl_hy)

    Hx, Hy = state.Hx, state.Hy
    # backward differences at the Ez nodes, over the live interior
    dHx_dy = s["dHx_dy"](Hx[1:-1, 1:], Hx[1:-1, :-1], dx)        # (nx-2, ny-2)
    dHy_dx = s["dHy_dx"](Hy[1:, 1:-1], Hy[:-1, 1:-1], dx)        # (nx-2, ny-2)
    psi_Hyx = _PSI_NEW(dHy_dx, state.psi_Hyx, m.b_ex, m.a_ex)
    psi_Hxy = _PSI_NEW(dHx_dy, state.psi_Hxy, m.b_ey, m.a_ey)
    # Association matched to meep's, not just the algebra: stepper_tm computes
    # `d1/k1 + psi1 - d2/k2 - psi2` left to right, and grouping it as
    # (d1/k1 + psi1) - (d2/k2 + psi2) instead changes the float64 rounding --
    # measured 3.97e-23 on Ez after 3 steps, enough to miss a bitwise gate that
    # the pylops and devito curls both meet. The expression is derived; the
    # evaluation order is chosen.
    curl = _STRETCHED(dHy_dx, psi_Hyx, m.k_ex) - dHx_dy / m.k_ey - psi_Hxy
    return state._replace(Dz=state.Dz + dt * F.pad(curl, (1, 1, 1, 1)),
                          psi_Hyx=psi_Hyx, psi_Hxy=psi_Hxy)


def step_tm_sympy(state, m, src_amp=0.0, src_mask=None, src_h_amp=0.0,
                  src_hy_mask=None, src_hx_mask=None):
    """One TM step in meep's phase order, every phase derived in this package.

    Signature matches meep_torch's step_tm, and `state`/`m` are consumed by
    attribute only, so `_make_runner(step_tm_sympy, TMState)` works without
    this module importing meep_torch."""
    state = step_db_sympy(state, m, "B")
    if src_hy_mask is not None or src_hx_mask is not None:
        state = step_source(state, m, "B", src_h_amp,
                            src_hy_mask=src_hy_mask, src_hx_mask=src_hx_mask)
    state = update_eh(state, m, "H")
    state = step_db_sympy(state, m, "D")
    state = step_source(state, m, "D", src_amp, src_mask=src_mask)
    state = update_pols(state, m)
    return update_eh(state, m, "E")


def describe() -> None:
    """Show which block became TM and what the compiled differences are."""
    blocks = polarisation_blocks(("z",))
    for name, blk in sorted(blocks.items()):
        print(f"{name.upper():3s} components {blk['components']}")
    c, nb, h = sp.symbols("c nb h")
    for field, axis, target in (("Ez", "y", "Hx"), ("Ez", "x", "Hy"),
                                ("Hx", "y", "Ez"), ("Hy", "x", "Ez")):
        expr = difference(field, axis, target, h,
                          lambda i: c if i == 0 else nb)
        print(f"d({field})/d{axis} at {target}: {expr}")


if __name__ == "__main__":
    describe()
