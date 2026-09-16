"""The arithmetic that runs, proven equal to the arithmetic that was derived.

Two routes to the same expression, and the test is symbolic equality between
them -- not a numeric comparison of two implementations. The Python form is
pure operators, so evaluating it on sympy Symbols returns *the expression the
tensors compute*, which is what makes this a statement about the running code.

Why not just lambdify the derivation and run that? Because sympy does not carry
evaluation order: `UnevaluatedExpr` freezes the top-level association and sympy
still re-associates beneath it, turning meep's `a*(d/dx)` into `(a*d)/dx` and
costing bitwise parity. Derivation and bitwise parity are in tension in the
expression tree itself. Writing both and proving them equal keeps each where it
is strong.
"""
import sympy as sp
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid

from .derive_twice import b_phase, d_phase, derived_b_phase, derived_d_phase

_D_ARGS = ("Hy_lo", "Hy_hi", "Hx_lo", "Hx_hi", "psi_x", "psi_y",
           "b_x", "a_x", "k_x", "b_y", "a_y", "k_y", "dt", "dx", "Ez")
_B_ARGS = ("Ez_lo", "Ez_hi", "psi", "b", "a", "k", "dt", "dx", "H", "sign")


def test_d_phase_running_code_equals_the_derivation():
    actual, px, py = d_phase(*map(sp.Symbol, _D_ARGS))
    oracle, opx, opy = derived_d_phase()
    assert sp.simplify(oracle - actual) == 0
    assert sp.simplify(opx - px) == 0 and sp.simplify(opy - py) == 0


def test_b_phase_running_code_equals_the_derivation():
    actual, p = b_phase(*map(sp.Symbol, _B_ARGS))
    oracle, op = derived_b_phase("Ez", "y", "Hx")
    assert sp.simplify(oracle - actual) == 0
    assert sp.simplify(op - p) == 0


def test_the_proof_can_fail():
    """A symbolic-equality test that cannot fail proves nothing.

    Perturb one coefficient in the running form and the equality must break.
    Without this the two routes could have been the same function all along.
    """
    args = list(map(sp.Symbol, _D_ARGS))
    args[7] = sp.Symbol("a_x") * 2          # a_x -> 2 a_x
    wrong, _, _ = d_phase(*args)
    oracle, _, _ = derived_d_phase()
    assert sp.simplify(oracle - wrong) != 0


def test_the_running_form_is_bitwise_meep():
    """Symbolic equality is not enough: the point of the Python form is that
    its ASSOCIATION is meep's, which no expression tree records. This checks
    the property the derivation cannot carry."""
    from meep_torch.core.stepper_tm import init_tm_state, step_db, update_eh

    from .generic_tm import init as ginit
    from .generic_tm_hoisted import make_step

    g = Grid(nx=24, ny=20, dx=0.05)
    m = build_tm_coeffs(g, npml=0)          # no PML: meep's association exactly
    torch.manual_seed(0)
    ez = torch.randn(g.nx, g.ny, dtype=g.dtype)
    hand = init_tm_state(g)._replace(Ez=ez.clone())
    hand = hand._replace(Dz=hand.Ez.clone())
    gen = ginit(g.nx, g.ny)._replace(Ez=ez.clone())
    step = make_step(m)
    for _ in range(6):
        hand = update_eh(step_db(step_db(hand, m, "B"), m, "D"), m, "E")
        gen = step(gen)
    for n in ("Ez", "Hx", "Hy"):
        assert torch.equal(getattr(hand, n), getattr(gen, n)), n
