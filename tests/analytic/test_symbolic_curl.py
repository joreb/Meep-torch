# SPDX-License-Identifier: MIT
"""The derivation in meep_torch.core.symbolic, checked against sympy itself.

symbolic.py writes the curl in an OPERATOR form -- symbols ``Dx``, ``Dy``,
``Dz`` multiplying plain field symbols -- because that is what lets
``system_matrix`` take a Jacobian and ``reduce_axes`` substitute ``Dz -> 0``.
``sympy.vector.curl`` cannot be used directly for that: it returns expressions
in ``Derivative(Ez(x, y, z), y)``, which do not factor into a coupling matrix
the same way.

That justifies the representation, not the typing. The nine signs in the curl
were still entered by hand, in a module whose whole premise is that hand-typed
stencils are where sign errors hide. This test closes that gap by realising the
operator form back into real derivatives and comparing against sympy's own
vector calculus.
"""
import sympy as sp
from sympy.vector import CoordSys3D, curl as vector_curl

from meep_torch.core.symbolic import (AXES, COMPONENTS, D, assert_decoupled,
                                      curl as operator_curl,
                                      polarisation_blocks)

_NAMES = ("Ex", "Ey", "Ez")


def _realise(expr, syms, fns, coord):
    """Turn the operator form back into honest derivatives: ``Du * f`` becomes
    ``d(f)/du``, so the result can be compared with sympy.vector."""
    out = sp.expand(expr)
    for axis in AXES:
        for name, fn in zip(_NAMES, fns):
            out = out.subs(D[axis] * syms[name], sp.Derivative(fn, coord[axis]))
    return sp.simplify(out)


def test_operator_curl_matches_sympy_vector():
    """All nine signs, against sympy.vector.curl on the same field."""
    C = CoordSys3D("C")
    fns = [sp.Function(n)(C.x, C.y, C.z) for n in _NAMES]
    syms = {n: sp.Symbol(n) for n in _NAMES}
    coord = {"x": C.x, "y": C.y, "z": C.z}

    field = fns[0] * C.i + fns[1] * C.j + fns[2] * C.k
    reference = vector_curl(field)

    mine = operator_curl((syms["Ex"], syms["Ey"], syms["Ez"]))
    for component, basis, axis in zip(mine, (C.i, C.j, C.k), AXES):
        got = _realise(component, syms, fns, coord)
        want = sp.simplify(reference.dot(basis))
        assert sp.simplify(got - want) == 0, (
            f"curl {axis}-component disagrees with sympy.vector:\n"
            f"  symbolic.py: {got}\n  sympy.vector: {want}")


def test_two_dimensional_reduction_decouples():
    """TM and TE exist because the reduced system is block-diagonal, not
    because anything declared them. assert_decoupled raises if it is not."""
    assert_decoupled(("z",))
    blocks = polarisation_blocks(("z",))
    assert blocks["tm"]["components"] == ("Ez", "Hx", "Hy")
    assert blocks["te"]["components"] == ("Hz", "Ex", "Ey")
    # the two blocks together account for all six components, exactly once
    members = blocks["tm"]["components"] + blocks["te"]["components"]
    assert sorted(members) == sorted(COMPONENTS)
