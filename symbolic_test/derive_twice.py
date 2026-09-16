"""Derive twice, execute once.

sympy cannot carry evaluation order -- `UnevaluatedExpr` freezes the top-level
association and sympy still re-associates the sub-expressions, so
`a*(d/dx)` comes back as `(a*d)/dx` and the result is one ulp from meep. Bitwise
parity is an evaluation-order property; an expression tree does not have one.
And the gather cannot live inside the expression either: sympy's TorchPrinter
raises on `Indexed`, and the numpy printer emits scalar indexing.

So neither "lambdify the whole step" nor "write it by hand" is right on its own.
Write it BOTH ways and prove them equal:

  * the PYTHON form runs. Its association is meep's, so it is bitwise, and
    inductor fuses it.
  * the SYMPY form is the oracle. It is association-agnostic and it is what
    `core.symbolic` derives from `curl` + `difference` + `pml_split`.

The proof is cheap because the Python form is pure arithmetic operators: feed it
sympy Symbols instead of tensors and it *returns the expression it computes*.
Then `simplify(oracle - actual) == 0` is a statement about the code that runs,
not about a parallel implementation of it.

That is the piece a closure factory cannot supply by itself -- a closure's
arithmetic is not a sympy object, so nothing can be proven about it.
"""
import sympy as sp

from meep_torch.core.symbolic import (AXES, PSI_NEW, STRETCHED, YEE,
                                      difference, pml_split, stencil_offsets)


# --- the arithmetic that RUNS. Operators only, so it evaluates on tensors and
# --- on sympy Symbols alike. The gather happens before it is called.
def b_phase(Ez_lo, Ez_hi, psi, b, a, k, dt, dx, H, sign):
    """One H component from one Ez difference. meep's association."""
    d = (Ez_hi - Ez_lo) / dx
    p = PSI_NEW(d, psi, b, a)
    return H + sign * dt * STRETCHED(d, p, k), p


def d_phase(Hy_lo, Hy_hi, Hx_lo, Hx_hi, psi_x, psi_y,
            b_x, a_x, k_x, b_y, a_y, k_y, dt, dx, Ez):
    """Ez from the curl of H, composed left to right exactly as meep does."""
    dHy = (Hy_hi - Hy_lo) / dx
    dHx = (Hx_hi - Hx_lo) / dx
    px = PSI_NEW(dHy, psi_x, b_x, a_x)
    py = PSI_NEW(dHx, psi_y, b_y, a_y)
    curl = STRETCHED(dHy, px, k_x) - dHx / k_y - py
    return Ez + dt * curl, px, py


# --- the same physics, DERIVED. Independent route: core.symbolic's difference
# --- (Fornberg, Yee-aware) and pml_split, assembled without reference to the
# --- functions above.
def _sym(name):
    return sp.Symbol(name)


def _namer(base, field, axis, target):
    """Map a stencil offset to the running code's lo/hi slice name.

    NOT by `k == 0`. `difference` numbers samples by their Yee offset, so a
    backward difference uses `k = -1, 0` and a forward one `k = 0, 1`: in the
    backward case `k = 0` is the HIGHER index, and naming it "lo" silently
    transposes the difference. That mistake produced a derived D phase whose
    psi terms carried the wrong sign, which is the same argument-order trap
    that already reached stepper_kz once through a compiled signature. The
    names come from the offsets themselves.
    """
    delta = (YEE[field][AXES.index(axis)] - YEE[target][AXES.index(axis)])
    ks = stencil_offsets(delta)
    lo = min(ks)
    return lambda k: _sym(f"{base}_{'lo' if k == lo else 'hi'}")


def derived_b_phase(field, axis, target):
    """``(H_new, psi_new)`` for one H component, from the derivation."""
    h, dt = _sym("dx"), _sym("dt")
    d = difference(field, axis, target, h, _namer("Ez", field, axis, target))
    psi, b, a, k = (_sym("psi"), _sym("b"), _sym("a"), _sym("k"))
    psi_new, stretched = pml_split(d, psi, b, a, k)
    sign = _sym("sign")
    return _sym("H") + sign * dt * stretched, psi_new


def derived_d_phase():
    """``Ez_new`` from the derivation, meep's association imposed by hand.

    The association is NOT derived and cannot be: it is the one thing the
    expression tree does not carry. It is written here the way the stepper
    writes it, and the test below is what ties the two together.
    """
    h, dt = _sym("dx"), _sym("dt")
    dHy = difference("Hy", "x", "Ez", h, _namer("Hy", "Hy", "x", "Ez"))
    dHx = difference("Hx", "y", "Ez", h, _namer("Hx", "Hx", "y", "Ez"))
    px, sx = pml_split(dHy, _sym("psi_x"), _sym("b_x"), _sym("a_x"), _sym("k_x"))
    py, _ = pml_split(dHx, _sym("psi_y"), _sym("b_y"), _sym("a_y"), _sym("k_y"))
    curl = sx - dHx / _sym("k_y") - py
    return _sym("Ez") + dt * curl, px, py
