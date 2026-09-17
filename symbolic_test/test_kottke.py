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
import torch

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


# ---------------------------------------------------------------------------
# The numeric implementation, gated against the algebra above.
#
# `geometry/smoothing.py::blend_tensor_tau` is what the engine calls; the
# module above is what proves it right. They are written independently -- one
# in torch over batched matrices, one in sympy over symbols -- so agreement is
# evidence rather than a restatement.
# ---------------------------------------------------------------------------

def _t(x, shape=(2, 3)):
    return torch.full(shape, float(x), dtype=torch.float64)


def _six(exx, exy, exz, eyy, eyz, ezz, shape=(2, 3)):
    return tuple(_t(v, shape) for v in (exx, exy, exz, eyy, eyz, ezz))


@pytest.mark.parametrize("axis,n", [(0, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1))])
def test_the_numeric_blend_reproduces_the_closed_form(axis, n):
    """Scalar constituents: torch must give what the sympy closed form gives.

    Harmonic along the blend axis, arithmetic across it. The blend AXIS is the
    interface normal -- averaging two adjacent cells onto the node between them
    is a two-phase cell whose interface is perpendicular to that axis, which is
    what makes Kottke applicable to a sampled epsilon at all.
    """
    from meep_torch.geometry.smoothing import blend_tensor_tau

    ei, eo = 12.0, 1.0
    got = blend_tensor_tau(_six(ei, 0, 0, ei, 0, ei),
                           _six(eo, 0, 0, eo, 0, eo), axis, 0.5)
    want_m = kottke_scalar(n, sp.Rational(1, 2), ei, eo)
    want = [want_m[i, j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    for g, w in zip(got, want):
        assert abs(float(g.flatten()[0]) - float(w)) < 1e-12


def test_the_numeric_blend_reproduces_the_tau_route_for_a_tensor():
    """Anisotropic constituents, where the closed form does not apply.

    This is the case that matters: `stepper_fv2d`'s liquid-crystal director has
    a live `eps_yz`, and a diagonal rule would throw the grating's physics away.
    """
    from meep_torch.geometry.smoothing import blend_tensor_tau

    no2, ne2, phi = 2.25, 2.89, math.pi / 6
    c, s = math.cos(phi), math.sin(phi)
    lc = (no2, 0.0, 0.0, no2 * c * c + ne2 * s * s,
          (ne2 - no2) * s * c, no2 * s * s + ne2 * c * c)
    vac = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    got = blend_tensor_tau(_six(*lc), _six(*vac), 0, 0.5)

    lc_m = sp.Matrix([[lc[0], lc[1], lc[2]], [lc[1], lc[3], lc[4]],
                      [lc[2], lc[4], lc[5]]])
    want_m = kottke_tensor(sp.Rational(1, 2), lc_m, sp.eye(3))
    want = [want_m[i, j] for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    for g, w in zip(got, want):
        assert abs(float(g.flatten()[0]) - float(w)) < 1e-10
    assert abs(float(got[4].flatten()[0])) > 1e-6, "the director coupling vanished"


@pytest.mark.parametrize("cov,which", [(0.0, 1), (1.0, 0)])
def test_the_numeric_blend_collapses_with_no_interface(cov, which):
    """A full or empty cell is the bare medium, whatever the axis."""
    from meep_torch.geometry.smoothing import blend_tensor_tau

    a, b = (7.0, 0.0, 0.0, 7.0, 0.0, 7.0), (2.0, 0.0, 0.0, 2.0, 0.0, 2.0)
    got = blend_tensor_tau(_six(*a), _six(*b), 1, cov)
    want = (a, b)[which][0]
    for k, g in zip((0, 3, 5), (got[0], got[3], got[5])):
        assert abs(float(g.flatten()[0]) - want) < 1e-12
    for k in (1, 2, 4):
        assert abs(float(got[k].flatten()[0])) < 1e-12


def test_the_numeric_blend_is_not_an_arithmetic_average():
    """The check that would have caught the fix that made things worse.

    Moving eps onto a node with a plain arithmetic mean is FIRST order across
    an interface, and measured worse on `stepper_fv2d` than not moving it at
    all. If `blend_tensor_tau` ever silently degraded to that, every test above
    would still pass for the off-diagonal-free cases, so assert the difference
    directly.
    """
    from meep_torch.geometry.smoothing import blend_tensor_tau

    ei, eo = 12.0, 1.0
    got = blend_tensor_tau(_six(ei, 0, 0, ei, 0, ei),
                           _six(eo, 0, 0, eo, 0, eo), 0, 0.5)
    arithmetic = 0.5 * (ei + eo)
    harmonic = 1.0 / (0.5 / ei + 0.5 / eo)
    assert abs(float(got[0].flatten()[0]) - harmonic) < 1e-12   # normal
    assert abs(float(got[3].flatten()[0]) - arithmetic) < 1e-12  # parallel
    assert abs(float(got[0].flatten()[0]) - arithmetic) > 1.0, (
        "the normal component took the arithmetic mean")
