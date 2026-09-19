# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The mode transforms as identities, not as literals.

`modes/transforms.py` types two Jacobians and `modes/fd2d.py` types the RESULT
of contracting one of them. These prove the typed forms instead.
"""
import sympy as sp

from .transforms import (angled_closed_form, bend_dwdz_assumes_positive_u,
                        bend_jacobian, bend_offdiagonal_under_arclength,
                        contract, prove_bend_collapses_to_scalings,
                        prove_bend_det_is_dwdz, shear_jacobian)

R, x, theta, phi = (sp.Symbol("R", positive=True), sp.Symbol("x", real=True),
                    sp.Symbol("theta", real=True), sp.Symbol("phi", real=True))


def test_the_bend_jacobian_is_derived_not_typed():
    """`diag(1, 1, R/u)` falls out of the embedding, including `dwdz`."""
    J, dwdz = bend_jacobian()
    assert J == sp.diag(1, 1, R / (R + x))
    assert sp.simplify(dwdz - R / (R + x)) == 0
    assert J[0, 2] == 0 and J[1, 2] == 0     # the diagonal claim itself


def test_det_of_the_bend_jacobian_is_dwdz():
    """The docstring says so and `fd2d` divides by it. Two copies, one fact."""
    assert prove_bend_det_is_dwdz() == 0


def test_the_bend_contraction_collapses_to_the_six_typed_lines():
    """THE point of this module.

    `fd2d.py:429-434` hand-expands `J eps J^T / det(J)` for a diagonal J into
    `eps_xx /= dwdz`, `eps_yy /= dwdz`, `eps_zz *= dwdz` -- and the same three
    for mu. That collapse existed only as a sentence in a docstring.
    """
    assert prove_bend_collapses_to_scalings() == sp.zeros(3, 3)


def test_the_collapse_is_not_vacuous():
    """The check has to be able to fail, or it is decoration.

    Scale eps_zz the wrong way and the difference must stop being zero.
    """
    a, b, c = sp.symbols("a_xx a_yy a_zz", positive=True)
    J, dwdz = bend_jacobian()
    wrong = sp.diag(a / dwdz, b / dwdz, c / dwdz)      # /dwdz, not *dwdz
    assert sp.simplify(contract(J, sp.diag(a, b, c)) - wrong) != sp.zeros(3, 3)


def test_the_shear_jacobian_is_derived_not_typed():
    """One `Matrix.jacobian` of the map reproduces the numpy literal."""
    J = shear_jacobian()
    assert J == sp.Matrix([[1, 0, -sp.tan(theta) * sp.cos(phi)],
                           [0, 1, -sp.tan(theta) * sp.sin(phi)],
                           [0, 0, 1]])
    assert sp.simplify(J.det() - 1) == 0      # the `det = 1` the code asserts


def test_the_angled_tensor_has_a_closed_form():
    """Four expressions, against two einsums over a (3, 3, N) array."""
    e = sp.Symbol("e", positive=True)
    t = sp.tan(theta)
    got = angled_closed_form()
    assert sp.simplify(got[0, 2] + e * t * sp.cos(phi)) == 0
    assert sp.simplify(got[1, 2] + e * t * sp.sin(phi)) == 0
    assert sp.simplify(got[2, 2] - e) == 0
    assert sp.simplify(got[0, 0] - e * (1 + (t * sp.cos(phi)) ** 2)) == 0
    assert sp.simplify(got[0, 1] - e * t ** 2 * sp.cos(phi) * sp.sin(phi)) == 0
    # theta = 0 must give back plain isotropic eps
    assert sp.simplify(got.subs(theta, 0) - sp.diag(e, e, e)) == sp.zeros(3, 3)


def test_the_bend_offdiagonal_vanishes_only_for_this_propagation_coordinate():
    """"The bend Jacobian is diagonal" is about the COORDINATE, not the bend.

    Measure the propagation coordinate as physical arc length rather than at
    the reference radius and the off-diagonal returns. Worth a test because a
    reader who takes the diagonal property as a fact about bends will reach for
    exactly that parametrisation.
    """
    assert bend_offdiagonal_under_arclength() != 0


def test_the_typed_dwdz_assumes_the_guide_is_inside_the_bend():
    """`R/|R + x|` derived, `R/u` typed: equal only while `u > 0`."""
    derived, typed = bend_dwdz_assumes_positive_u()
    assert sp.simplify(derived - typed) != 0                  # not identical
    assert sp.simplify((derived - typed).subs(x, R / 2)) == 0  # agree inside
    assert sp.simplify((derived - typed).subs(x, -2 * R)) != 0  # differ outside


def test_the_collapse_for_a_general_symmetric_tensor():
    """The TRUE statement: the diagonal scalings, plus `eps_xy / dwdz`.

    Found by a peer review of the diagonal proof above. "Reduces to
    per-component scaling" is true of the diagonal alone; the full contraction
    also scales the xy entry, and leaves xz and yz alone because `J_xx = 1` and
    `J_zz/det(J) = 1`.
    """
    from .transforms import bend_collapse_full_tensor

    assert bend_collapse_full_tensor() == sp.zeros(3, 3)


def test_the_diagonal_rule_is_incomplete_and_this_is_where():
    """`fd2d`'s six lines touch [0,0], [1,1], [2,2] and nothing else.

    Harmless today -- `solve_2d` is a diagonal solver, builds its tensor from
    `_yee_eps` and reads only those three entries. It stops being harmless the
    moment a bend meets a tensorial cross-section, which is what the angled
    transform and Kottke smoothing both produce by construction. The omission
    is exactly `eps_xy (1/dwdz - 1)`, so it vanishes only as the bend
    straightens.
    """
    from .transforms import bend_jacobian, what_the_diagonal_rule_omits

    _, dwdz = bend_jacobian()
    gap = what_the_diagonal_rule_omits()
    xy = sp.Symbol("a_xy", real=True)

    assert gap != sp.zeros(3, 3)
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 2), (2, 1)):
        assert sp.simplify(gap[i, j]) == 0, (i, j)
    for i, j in ((0, 1), (1, 0)):
        assert sp.simplify(gap[i, j] - xy * (1 / dwdz - 1)) == 0, (i, j)
    # and it closes as the bend straightens
    assert sp.simplify(gap[0, 1].subs(sp.Symbol("x", real=True), 0)) == 0
