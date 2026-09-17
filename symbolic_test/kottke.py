# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The Kottke subpixel-smoothing tensor, derived rather than typed.

Kottke, Farjadpour and Johnson (Phys. Rev. E 77, 036611, 2008) smooth a
dielectric interface by averaging the field components the way each one's
continuity condition demands: the components PARALLEL to the interface are
continuous so eps averages arithmetically, the component NORMAL to it has
continuous D so eps averages harmonically. Both statements are about a frame
aligned with the interface normal, which makes the whole construction frame
algebra -- the same kind `core.symbolic` already does for the symmetry
operators, where a mirror is the Householder `I - 2 n n^T`.

TWO ROUTES, and the point of deriving is that they must agree.

* The CLOSED FORM, valid when both constituents are scalar. With the rank-1
  projector `P = n n^T` onto the normal,

      eps_eff = h P + a (I - P)

  where `a` is the arithmetic and `h` the harmonic average over the cell. It
  is manifestly rotation-covariant: nothing picks a coordinate axis.

* The GENERAL ROUTE, valid for anisotropic constituents, which is what the
  polarization grating needs. Kottke defines an invertible map `tau` that
  sends eps to a matrix whose entries average LINEARLY, averages that, and
  maps back:

      eps_eff = tau_inv( <tau(eps)> )

  For scalar eps this collapses to the closed form, and `prove_routes_agree`
  checks that symbolically rather than by assertion.

WHAT THIS EXPOSED. `meep_torch/geometry/smoothing.py::blend_te` used to keep
only the DIAGONAL of the closed form:

    eps_xx = nx^2 h + (1 - nx^2) a        eps_yy = ny^2 h + (1 - ny^2) a

which is exactly `diag(h P + a (I - P))`. The off-diagonal it dropped is

    eps_xy = (h - a) nx ny

so the hand-written form was EXACT for an axis-aligned interface (`nx ny = 0`)
and worst at 45 degrees, where `nx ny = 1/2`. That is a statement about which
geometries that smoothing was right for, and it was not recoverable from the
code -- the term was not commented out, it was never written. `blend_te`
returns it now; `blend_te_supersampled` still projects it away for the mpb
eigensolver, which carries a diagonal eps.

The derivation itself is not wired into the engine -- it is symbolic and
independent, which is what makes it evidence when `test_kottke.py` gates the
torch implementation against it. `blend_tensor_tau`, the general route below,
is what `stepper_fv2d` calls under `eps_at_nodes=True`.
"""
import sympy as sp

__all__ = ["projector", "kottke_scalar", "tau", "tau_inv", "kottke_tensor",
           "prove_routes_agree", "blend_te_diagonal", "dropped_offdiagonal"]


def projector(n):
    """`P = n n^T`, the rank-1 projector onto the interface normal.

    The same object a Householder mirror is built from (`I - 2 P`), which is
    why the symmetry module and the smoothing module are algebraically the
    same subject seen twice.
    """
    n = sp.Matrix(3, 1, list(n))
    return n * n.T


def kottke_scalar(n, cov, eps_in, eps_out):
    """`h P + a (I - P)` for scalar constituents.

    ``cov`` is the fill fraction of ``eps_in`` in the cell. No coordinate axis
    appears, so the result rotates with ``n`` by construction.
    """
    a = cov * eps_in + (1 - cov) * eps_out
    h = 1 / (cov / eps_in + (1 - cov) / eps_out)
    P = projector(n)
    return sp.simplify(h * P + a * (sp.eye(3) - P))


def tau(e):
    """Kottke's map, in the frame whose FIRST axis is the interface normal.

    The entries of `tau(eps)` are the quantities that average linearly across
    the interface: `-1/eps_11` because D_normal is continuous, the parallel
    block because E_parallel is, and the cross terms scaled so the map stays
    invertible.
    """
    e = sp.Matrix(e)
    e11 = e[0, 0]
    return sp.Matrix([
        [-1 / e11,        e[0, 1] / e11,                    e[0, 2] / e11],
        [e[1, 0] / e11,   e[1, 1] - e[1, 0] * e[0, 1] / e11, e[1, 2] - e[1, 0] * e[0, 2] / e11],
        [e[2, 0] / e11,   e[2, 1] - e[2, 0] * e[0, 1] / e11, e[2, 2] - e[2, 0] * e[0, 2] / e11],
    ])


def tau_inv(t):
    """The inverse map. `tau` is an involution up to the sign of its 1,1 entry."""
    t = sp.Matrix(t)
    t11 = t[0, 0]
    return sp.Matrix([
        [-1 / t11,       -t[0, 1] / t11,                   -t[0, 2] / t11],
        [-t[1, 0] / t11,  t[1, 1] - t[1, 0] * t[0, 1] / t11, t[1, 2] - t[1, 0] * t[0, 2] / t11],
        [-t[2, 0] / t11,  t[2, 1] - t[2, 0] * t[0, 1] / t11, t[2, 2] - t[2, 0] * t[0, 2] / t11],
    ])


def kottke_tensor(cov, eps_in, eps_out):
    """General anisotropic Kottke, in the normal-aligned frame.

    ``eps_in``/``eps_out`` are 3x3 matrices already expressed in the frame
    whose first axis is the interface normal; rotate them there and rotate the
    result back. This is the form `stepper_fv2d` would need, since its eps has
    a live off-diagonal (the LC director's `eps_yz`) that no diagonal rule can
    carry.
    """
    avg = cov * tau(eps_in) + (1 - cov) * tau(eps_out)
    return sp.simplify(tau_inv(avg))


def prove_routes_agree():
    """The two routes must give the same tensor for scalar constituents.

    Returns the simplified difference, which must be the zero matrix. This is
    the check that the general map is the same physics as the closed form and
    not a second, independently-typed rule.
    """
    cov, ei, eo = sp.symbols("cov eps_in eps_out", positive=True)
    general = kottke_tensor(cov, ei * sp.eye(3), eo * sp.eye(3))
    closed = kottke_scalar((1, 0, 0), cov, ei, eo)     # normal-aligned frame
    return sp.simplify(general - closed)


def blend_te_diagonal(n, cov, eps_in, eps_out):
    """The DIAGONAL of what `geometry/smoothing.py::blend_te` computes, `(xx, yy)`."""
    a = cov * eps_in + (1 - cov) * eps_out
    h = 1 / (cov / eps_in + (1 - cov) / eps_out)
    nx2, ny2 = n[0] ** 2, n[1] ** 2
    return (sp.simplify(nx2 * h + (1 - nx2) * a),
            sp.simplify(ny2 * h + (1 - ny2) * a))


def dropped_offdiagonal(n, cov, eps_in, eps_out):
    """The `eps_xy` term the diagonal-only implementation never had.
    (`blend_te` ships it now; `blend_te_supersampled` still projects it away.)

    `(h - a) nx ny`: zero for an axis-aligned interface, maximal at 45 degrees.
    """
    return sp.simplify(kottke_scalar(n, cov, eps_in, eps_out)[0, 1])


if __name__ == "__main__":
    cov, ei, eo = sp.symbols("cov eps_in eps_out", positive=True)
    th = sp.Symbol("theta", real=True)
    n = (sp.cos(th), sp.sin(th), 0)

    print("routes agree (must be the zero matrix):")
    sp.pprint(prove_routes_agree())
    print("\nclosed form at a general in-plane normal:")
    sp.pprint(kottke_scalar(n, cov, ei, eo))
    print("\nthe off-diagonal blend_te drops:")
    sp.pprint(dropped_offdiagonal(n, cov, ei, eo))
