# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The derived Kottke tensor, and what the shipped diagonal form drops.

Nothing here is wired into the engine. The point is to have the algebra proven
before the smoothing layer needs it -- `stepper_fv2d` carries a full 3x3 eps
(the LC director's live `eps_yz`) and gets no smoothing at all today, because
`geometry/smoothing.py` only produces a diagonal pair.
"""
import math

import pytest
import sympy as sp

from .kottke import (blend_te_diagonal, dropped_offdiagonal, kottke_scalar,
                     kottke_tensor, projector, prove_routes_agree, tau,
                     tau_inv)

COV, EI, EO = sp.symbols("cov eps_in eps_out", positive=True)


def test_the_two_routes_are_the_same_physics():
    """Closed form vs Kottke's tau-map, for scalar constituents.

    This is the whole reason to derive rather than type: `h P + a (I - P)` and
    `tau_inv(<tau(eps)>)` are different-looking rules, and if they disagreed
    one of them would be a second hand-written convention wearing the same
    name.
    """
    assert prove_routes_agree() == sp.zeros(3, 3)


def test_tau_is_invertible():
    """`tau_inv . tau` is the identity, or the general route means nothing."""
    e = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"e{min(i,j)}{max(i,j)}"))
    assert sp.simplify(tau_inv(tau(e)) - e) == sp.zeros(3, 3)


@pytest.mark.parametrize("cov,expect", [(0, "out"), (1, "in")])
def test_a_full_or_empty_cell_is_the_bare_medium(cov, expect):
    """No interface, no smoothing: the tensor must collapse to a scalar."""
    n = (sp.Rational(3, 5), sp.Rational(4, 5), 0)     # a generic unit normal
    got = sp.simplify(kottke_scalar(n, cov, EI, EO))
    want = (EI if expect == "in" else EO) * sp.eye(3)
    assert sp.simplify(got - want) == sp.zeros(3, 3)


def test_the_diagonal_is_exactly_what_blend_te_ships():
    """The implementation is not wrong -- it is the diagonal of this."""
    th = sp.Symbol("theta", real=True)
    n = (sp.cos(th), sp.sin(th), 0)
    full = kottke_scalar(n, COV, EI, EO)
    xx, yy = blend_te_diagonal(n, COV, EI, EO)
    assert sp.simplify(full[0, 0] - xx) == 0
    assert sp.simplify(full[1, 1] - yy) == 0


def test_the_dropped_term_vanishes_only_for_axis_aligned_interfaces():
    """`(h - a) nx ny`: exact on axis, worst at 45 degrees.

    That is a statement about which geometries the shipped smoothing is right
    for, and it is not recoverable from the code -- the term is not commented
    out, it was never written.
    """
    th = sp.Symbol("theta", real=True)
    off = dropped_offdiagonal((sp.cos(th), sp.sin(th), 0), COV, EI, EO)

    for angle in (0, sp.pi / 2, sp.pi):               # axis-aligned
        assert sp.simplify(off.subs(th, angle)) == 0

    at45 = off.subs({th: sp.pi / 4, COV: sp.Rational(1, 2),
                     EI: 12, EO: 1})
    assert at45 != 0
    # and it is a real fraction of the diagonal, not a rounding-scale term
    xx = blend_te_diagonal((sp.cos(th), sp.sin(th), 0), COV, EI, EO)[0]
    ratio = abs(float(at45 / xx.subs({th: sp.pi / 4, COV: sp.Rational(1, 2),
                                      EI: 12, EO: 1})))
    assert ratio > 0.3, f"dropped term is only {ratio:.3f} of the diagonal"


def test_the_tensor_is_rotation_covariant():
    """Rotating the normal rotates the tensor.

    True by construction from `P = n n^T`, which is the reason to build it that
    way: a component-wise rule has to be checked for this, and the shipped
    diagonal form FAILS it -- rotating a diagonal-only tensor does not give the
    diagonal-only tensor of the rotated normal, which is the same fact as the
    dropped off-diagonal seen from the other side.
    """
    th, ph = sp.symbols("theta phi", real=True)
    R = sp.Matrix([[sp.cos(ph), -sp.sin(ph), 0],
                   [sp.sin(ph), sp.cos(ph), 0],
                   [0, 0, 1]])
    n = sp.Matrix([sp.cos(th), sp.sin(th), 0])
    lhs = sp.simplify(R * kottke_scalar(n, COV, EI, EO) * R.T)
    rhs = sp.simplify(kottke_scalar(R * n, COV, EI, EO))
    assert sp.simplify(lhs - rhs) == sp.zeros(3, 3)


def test_the_general_route_carries_an_anisotropic_constituent():
    """What fv2d needs and the diagonal rule cannot express.

    A uniaxial medium with a live `eps_yz` -- the liquid-crystal director of
    the polarization grating -- smoothed against vacuum. The result must keep
    the off-diagonal; a rule that returned a diagonal pair here would have
    thrown the grating's entire physics away.
    """
    phi = sp.pi / 6
    no2, ne2 = sp.Integer(sp.Rational(9, 4)), sp.Rational(289, 100)   # 1.5^2, 1.7^2
    lc = sp.Matrix([
        [no2, 0, 0],
        [0, no2 * sp.cos(phi) ** 2 + ne2 * sp.sin(phi) ** 2,
         (ne2 - no2) * sp.sin(phi) * sp.cos(phi)],
        [0, (ne2 - no2) * sp.sin(phi) * sp.cos(phi),
         no2 * sp.sin(phi) ** 2 + ne2 * sp.cos(phi) ** 2]])
    eff = kottke_tensor(sp.Rational(1, 2), lc, sp.eye(3))
    assert sp.simplify(eff[1, 2]) != 0, "the director coupling was smoothed away"
    assert sp.simplify(eff - eff.T) == sp.zeros(3, 3), "eps_eff must stay symmetric"


def test_projector_is_the_householder_building_block():
    """`I - 2P` is the mirror `core.symbolic` builds; same algebra, twice.

    Not decoration: it is why the smoothing tensor is derivable at all in a
    module that already knows how to reflect a frame.
    """
    from meep_torch.core.symbolic import mirror_matrix

    for n in ((1, 0, 0), (0, 1, 0), (0, 0, 1),
              (sp.Rational(3, 5), sp.Rational(4, 5), 0)):   # and an oblique one
        assert sp.simplify(sp.eye(3) - 2 * projector(n)
                           - mirror_matrix(n)) == sp.zeros(3, 3)
