# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The mode solver's coordinate transforms, derived rather than typed.

`meep_torch/modes/transforms.py` carries two Jacobians as numpy literals,
ported from tidy3d. A Jacobian is not a literal: it is the derivative of a
coordinate map, and sympy will take that derivative. What makes this worth
deriving is not the two matrices but the THIRD thing, which has no executable
form anywhere in the engine:

    the bend's contraction `eps' = J eps J^T / det(J)` collapses, for a
    DIAGONAL J, into per-component scalings

and those scalings are then hand-expanded into six lines of
`modes/fd2d.py` (`eps_xx /= dwdz`, `eps_yy /= dwdz`, `eps_zz *= dwdz`, and mu
identically). The collapse itself lives only in a docstring sentence. That is
the same shape as the source-injection sign: a rule stated in prose, typed out
by hand somewhere else, with nothing able to contradict it -- and the
`apply_bend` double-scaling WARNING in that module is what it looks like when
one contraction lives in two places.

`prove_bend_collapses_to_scalings` turns the sentence into a check.

THE TWO MAPS.

* BEND, a waveguide curved in the x-z plane. The computational coordinates are
  `(x, y, w)` with `x` the offset from the bend radius and `w = R phi` an arc
  length measured at the REFERENCE radius. Embedding the straightened guide
  back into physical space,

      P(x, y, w) = ((R + x) cos(w/R),  y,  (R + x) sin(w/R))

  and sympy takes `Lambda = dP/d(x, y, w)` from there. Its columns are
  orthogonal with norms `(1, 1, u/R)`, `u = R + x`, so in the orthonormal frame
  the map is `diag(1, 1, u/R)` and the one we want -- physical TO computational
  -- is its inverse, `diag(1, 1, R/u)`. That derives both the typed `J` and the
  typed `dwdz = R/u`.

  WHY THE OFF-DIAGONAL VANISHES, since this is the part worth getting right:
  `w = R phi` depends on `phi` alone and `x` is an independent coordinate, so
  `dw/dx = 0`. Parametrise the same bend by PHYSICAL arc length `z = u phi`
  instead and the off-diagonal is back, `dw/dx = -R z / u^2`. The two
  parametrisations are not equally good: only the first is translationally
  invariant in the propagation coordinate, which is the property a mode solver
  needs -- a cross-section that changes with `z` has no `e^{i beta w}` mode to
  find. tidy3d takes the first, and this is why.

* ANGLED, a straight guide tilted by `theta` about an azimuth `phi`. The map
  shears the propagation coordinate out of the cross-section,

      u = x - tan(theta) cos(phi) z,   v = y - tan(theta) sin(phi) z,   w = z

  whose Jacobian is the typed literal exactly, with `det = 1`.

`meep_torch.modes` does NOT import this module. sympy resolving a coordinate
map per cell is the step-time mistake in different clothes; the derivation is
the oracle and the tests are where the two meet.
"""
import sympy as sp

__all__ = ["bend_embedding", "bend_jacobian", "shear_jacobian", "contract",
           "prove_bend_collapses_to_scalings", "prove_bend_det_is_dwdz",
           "angled_closed_form", "bend_offdiagonal_under_arclength",
           "bend_dwdz_assumes_positive_u", "bend_collapse_full_tensor",
           "what_the_diagonal_rule_omits"]

x, y, w = sp.symbols("x y w", real=True)
R = sp.Symbol("R", positive=True)
theta, phi = sp.symbols("theta phi", real=True)


def bend_embedding():
    """Where a straightened bend sits in physical space, as a position vector."""
    return sp.Matrix([(R + x) * sp.cos(w / R), y, (R + x) * sp.sin(w / R)])


def bend_jacobian():
    """``diag(1, 1, dwdz)`` with ``dwdz = R/u``, DERIVED from the embedding.

    Returns ``(J, dwdz)``. The columns of ``dP/d(x, y, w)`` are orthogonal, so
    the orthonormal-frame map is the diagonal of their norms; physical to
    computational is its inverse, which is what the contraction wants.
    """
    lam = bend_embedding().jacobian([x, y, w])
    cols = [lam[:, i] for i in range(3)]
    for i in range(3):                       # the frame really is orthogonal
        for j in range(i + 1, 3):
            assert sp.simplify(cols[i].dot(cols[j])) == 0, (i, j)
    scale = [sp.simplify(sp.sqrt(c.dot(c))) for c in cols]
    forward = sp.diag(*scale)                # computational -> physical
    J = sp.simplify(forward.inv())           # physical -> computational
    # The norm hands back `Abs(R + x)`, and dropping it is an ASSUMPTION the
    # engine makes silently: `bend_dwdz` computes `radius / u` outright, which
    # is the same number only while `u > 0`. It is negative for a cross-section
    # that reaches past the bend centre, i.e. a radius smaller than the guide's
    # half-width -- physically degenerate, but the typed form gives a negative
    # eps scaling there rather than an error. Stated here so the assumption is
    # somewhere.
    J = sp.refine(J, sp.Q.positive(R + x))
    return J, sp.simplify(J[2, 2])


def shear_jacobian():
    """The angled map's Jacobian, derived from the map in one line."""
    xs, ys, zs = sp.symbols("x_s y_s z_s", real=True)
    u = xs - sp.tan(theta) * sp.cos(phi) * zs
    v = ys - sp.tan(theta) * sp.sin(phi) * zs
    return sp.simplify(sp.Matrix([u, v, zs]).jacobian([xs, ys, zs]))


def contract(J, eps):
    """``eps' = J eps J^T / det(J)``, the transformation-optics rule.

    The same expression serves mu, which is why `fd2d` applies the identical
    six lines to both -- nothing in it knows which material it holds.
    """
    return sp.simplify(J * eps * J.T / J.det())


def prove_bend_det_is_dwdz():
    """``det(J) == dwdz``. Returns the simplified difference, which must be 0.

    `modes/transforms.py` says so in a docstring and `modes/fd2d.py` divides by
    `dwdz` on that basis. Two typed copies of one fact, and this is the only
    place they could silently disagree.
    """
    J, dwdz = bend_jacobian()
    return sp.simplify(J.det() - dwdz)


def prove_bend_collapses_to_scalings():
    """The identity `modes/fd2d.py` hand-expands into six lines.

    Returns the simplified difference, which must be the zero matrix. Works for
    eps and mu alike: the symbols are just the three diagonal entries, and the
    contraction never asks which material they belong to.
    """
    a, b, c = sp.symbols("a_xx a_yy a_zz", positive=True)
    J, dwdz = bend_jacobian()
    got = contract(J, sp.diag(a, b, c))
    want = sp.diag(a / dwdz, b / dwdz, c * dwdz)
    return sp.simplify(got - want)


def bend_collapse_full_tensor():
    """The collapse for a GENERAL SYMMETRIC eps, which is the true statement.

    Returns the simplified difference against

        diag part : eps_xx/dwdz, eps_yy/dwdz, eps_zz*dwdz
        eps_xy    : eps_xy / dwdz          <- scaled, and the docstring omits it
        eps_xz    : eps_xz                 <- unchanged
        eps_yz    : eps_yz                 <- unchanged

    which must be the zero matrix. "Reduces to per-component scaling" is true
    of the DIAGONAL only; the full rule also scales xy, and leaves xz and yz
    alone because `J_xx = 1` and `J_zz / det(J) = 1`. The distinction is
    invisible while the cross-section is diagonal -- which is all `fd2d.solve_2d`
    accepts today, since it builds its tensor from `_yee_eps` and reads only the
    three diagonal entries -- and becomes load-bearing the moment a bend meets a
    tensorial cross-section: an angled transform makes eps_xz nonzero by
    construction, and a Kottke-smoothed interface makes the whole 3x3 nonzero by
    construction. `fd2d_tensor` carries no `bend_radius` yet, so this is the
    rule to have written down BEFORE it does.
    """
    J, dwdz = bend_jacobian()
    xx, yy, zz, xy, xz, yz = sp.symbols(
        "a_xx a_yy a_zz a_xy a_xz a_yz", real=True)
    eps = sp.Matrix([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    want = sp.Matrix([[xx / dwdz, xy / dwdz, xz],
                      [xy / dwdz, yy / dwdz, yz],
                      [xz,        yz,        zz * dwdz]])
    return sp.simplify(contract(J, eps) - want)


def what_the_diagonal_rule_omits():
    """What the six typed lines leave untouched on a general tensor.

    Returns ``contract(J, eps) - (eps with only its diagonal scaled)``. Nonzero
    exactly in the xy entries, by `eps_xy (1/dwdz - 1)`, so it vanishes only in
    the straight limit `dwdz -> 1`. Kept as the executable form of "the
    diagonal rule is incomplete", rather than a sentence saying so.
    """
    J, dwdz = bend_jacobian()
    xx, yy, zz, xy, xz, yz = sp.symbols(
        "a_xx a_yy a_zz a_xy a_xz a_yz", real=True)
    eps = sp.Matrix([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    diagonal_only = sp.Matrix([[xx / dwdz, xy, xz],
                               [xy, yy / dwdz, yz],
                               [xz, yz, zz * dwdz]])
    return sp.simplify(contract(J, eps) - diagonal_only)


def angled_closed_form():
    """The angled tensor for ISOTROPIC eps, in closed form.

    `modes/transforms.py` builds this with two einsums over a (3, 3, N) array;
    here it is four expressions, which is what the numpy path can be gated
    against entry by entry. With ``a = tan(theta) cos(phi)``,
    ``b = tan(theta) sin(phi)`` and ``det(J) = 1``:

        eps_xx = e (1 + a^2)    eps_xy = e a b    eps_xz = -e a
        eps_yy = e (1 + b^2)    eps_zz = e        eps_yz = -e b
    """
    e = sp.Symbol("e", positive=True)
    return sp.simplify(contract(shear_jacobian(), sp.diag(e, e, e)))


def bend_dwdz_assumes_positive_u():
    """The `Abs` the engine drops. Returns ``(derived, typed)`` for ``dwdz``.

    Deriving the scale factor from a column norm gives ``R/|R + x|``;
    `modes/transforms.py` types ``radius / u``. Equal while ``u > 0``, opposite
    in sign when the cross-section reaches past the bend centre. Not a bug --
    a radius below the guide's half-width is degenerate -- but the typed form
    returns a NEGATIVE eps scaling there instead of refusing, which is the kind
    of thing that is only ever found by writing the derivation down.
    """
    lam = bend_embedding().jacobian([x, y, w])
    c = lam[:, 2]
    return sp.simplify(1 / sp.sqrt(c.dot(c))), R / (R + x)


def bend_offdiagonal_under_arclength():
    """What the SAME bend gives if parametrised by physical arc length.

    `z = u phi` instead of `w = R phi`. Returns ``dw/dx``, which is NOT zero --
    it is `-R z / u^2`. Kept because "the off-diagonal vanishes" is a statement
    about the choice of propagation coordinate, not about bends, and a reader
    who assumes it is about bends will reach for this parametrisation and get a
    cross-section that changes along the guide.
    """
    z_arc = sp.Symbol("z", real=True)
    u = R + x
    return sp.simplify(sp.diff(R * z_arc / u, x))


if __name__ == "__main__":
    J, dwdz = bend_jacobian()
    print("bend J (derived):"); sp.pprint(J)
    print("\ndwdz =", dwdz, "   det(J) - dwdz =", prove_bend_det_is_dwdz())
    print("\nthe collapse fd2d.py hand-expands (must be the zero matrix):")
    sp.pprint(prove_bend_collapses_to_scalings())
    print("\nshear J (derived):"); sp.pprint(shear_jacobian())
    print("\nangled tensor, isotropic eps:"); sp.pprint(angled_closed_form())
    print("\ndw/dx under arc-length parametrisation:",
          bend_offdiagonal_under_arclength())
