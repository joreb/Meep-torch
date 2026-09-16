# SPDX-License-Identifier: MIT
# The independent oracle (maxwell_sympy, typed textbook + Lagrange
# interpolation) against meep_torch.core.symbolic (sympy.vector + Fornberg).
# This file may import both; the oracle module imports nothing from
# meep_torch. Two routes to one object: disagreement here is a finding about
# one of them, not a test to be tuned into agreement.
import numpy as np
import pytest
import sympy as sp
import torch
from sympy.vector import CoordSys3D, curl as _vcurl

from meep_torch.core import symbolic as core

from . import maxwell_sympy as oracle
from .step_operator import extract

_PAIRS = (("Ez", "y", "Hx"), ("Ez", "x", "Hy"), ("Hx", "y", "Ez"),
          ("Hy", "x", "Ez"), ("Ey", "x", "Hz"), ("Hz", "y", "Ex"))


def _plain_r(expr):
    """core leaves the cylindrical radius as a sympy.vector BaseScalar; the
    oracle uses a plain symbol. Map one onto the other for comparison."""
    return expr.subs({s: oracle.R for s in expr.free_symbols if str(s) == "Cy.r"})


@pytest.mark.parametrize("system", ["cartesian", "cylindrical"])
def test_curl_typed_textbook_matches_sympy_vector(system):
    v = sp.symbols("V1 V2 V3")
    for i, (got, want) in enumerate(zip(core.curl(v, system), oracle.curl(v, system))):
        assert sp.simplify(_plain_r(got) - want) == 0, (
            f"{system} curl component {i}: core {got} vs typed {want}")


def test_reduced_system_matrices_agree():
    assert core.system_matrix() == oracle.system_matrix()
    assert core.system_matrix(("z",)) == oracle.system_matrix(("z",))


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("field,axis,target", _PAIRS)
def test_fornberg_matches_lagrange(field, axis, target, order):
    h = sp.Symbol("h", positive=True)
    syms = {k: sp.Symbol(f"f{k}") for k in range(-3, 4)}
    got = core.difference(field, axis, target, h, syms.__getitem__, order=order)
    want = oracle.difference(field, axis, target, h, syms.__getitem__, order=order)
    assert sp.simplify(got - want) == 0, f"order {order}: core {got} vs oracle {want}"


def test_uniform_difference_argument_order():
    """The pinned (centre, neighbour, h) signature, numerically, against the
    oracle's Yee-derived forward and backward differences."""
    c, nb, h = sp.symbols("c nb h")
    for direction, (field, axis, target) in (("forward", ("Ez", "y", "Hx")),
                                             ("backward", ("Hx", "y", "Ez"))):
        want = sp.lambdify((c, nb, h), oracle.difference(
            field, axis, target, h, lambda k: c if k == 0 else nb))
        got = core.compiled_uniform_difference(direction)
        args = (torch.tensor(1.0), torch.tensor(3.0), torch.tensor(0.5))
        assert got(*args).item() == want(*args).item(), direction


def _whole_curl_pair():
    fn, names = core.compiled_stretched_curl(
        (sp.Symbol("Hx"), sp.Symbol("Hy"), sp.Symbol("Hz")), "Ez")
    h = sp.Symbol("h")
    want = oracle.stretched_curl(
        (sp.Symbol("Hx"), sp.Symbol("Hy"), sp.Symbol("Hz")), "Ez", h,
        sym=lambda n, k: sp.Symbol(n if k == 0 else f"{n}_{'p' if k > 0 else 'm'}{abs(k)}"),
        psi=lambda n, a: sp.Symbol(f"psi_{n}{a}"),
        coef=lambda a: tuple(sp.Symbol(f"{c}_{a}") for c in "bak"))
    return fn, names, want


def test_whole_curl_expression_matches_typed_assembly():
    fn, names, want = _whole_curl_pair()
    got = fn(*[sp.Symbol(n) for n in names])
    assert sp.simplify(got - want) == 0, f"core {got}\noracle {want}"
    assert set(map(str, want.free_symbols)) == set(names)


def test_whole_curl_numeric_agreement():
    fn, names, want = _whole_curl_pair()
    oracle_fn = sp.lambdify([sp.Symbol(n) for n in names], want)
    g = torch.Generator().manual_seed(0)
    vals = {n: torch.rand(7, 5, dtype=torch.float64, generator=g) + 0.5 for n in names}
    a, b = fn(*[vals[n] for n in names]), oracle_fn(*[vals[n] for n in names])
    torch.testing.assert_close(a, b, rtol=1e-14, atol=0)


def test_symmetry_table_matches_derived_transform():
    for kind, rows in oracle.PARITY.items():
        assert sp.simplify(core.SYMMETRIES[kind] - oracle.SYMMETRY_MATRICES[kind]) \
            == sp.zeros(3, 3), kind
        for comp, want in rows.items():
            got = {k: int(v) for k, v in core.component_transform(kind, comp).items()}
            assert got == want, f"{kind} {comp}: core {got} vs typed {want}"


# ------------------------------------------------ the D phase, both forms --
# stepper_sympy composes the derived pieces in meep's left-to-right order.
# core's default assembles the whole curl. A random initial state puts psi
# live in the PML from step one, which is what the 3-steps-from-zero gate in
# test_splitter_2d never does, so this is the check of the default path.

_FIELDS = ("Ez", "Hx", "Hy", "psi_Ezy", "psi_Ezx", "psi_Hyx", "psi_Hxy")


def _operators(N=10, npml=3):
    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.grid import Grid
    from meep_torch.core.stepper_tm import init_tm_state, step_tm

    from .stepper_sympy import step_tm_sympy

    g = Grid(nx=N, ny=N, dx=0.05)
    m = build_tm_coeffs(g, npml=npml)
    s0 = init_tm_state(g)
    a_core, _, _, index = extract(lambda s: step_tm(s, m), s0, _FIELDS, tied={"Dz": "Ez"})
    a_oracle = extract(lambda s: step_tm_sympy(s, m), s0, _FIELDS, tied={"Dz": "Ez"})[0]
    ez = slice(index["Ez"][0], index["Ez"][0] + N * N)
    psi = slice(index["psi_Hyx"][0], None)
    assert np.abs(a_core[ez, psi]).max() > 0, "psi memory is not live in the D phase"
    return a_core, a_oracle


def test_core_and_oracle_steppers_are_bitwise():
    """The core stepper and the standalone sympy oracle agree bit for bit.

    Both compose the derived per-difference pieces in meep's left-to-right
    order, so this is a real bitwise claim and not a tolerance. It used to be
    conditional on MEEP_TORCH_WHOLE_CURL=0; that switch is gone, because the
    one-piece derived curl cannot be what executes -- an expression tree does
    not carry an association, and bitwise parity is an association property.
    The derived form is now the oracle only (see derive_twice.py).

    The dense operator is extracted rather than a few steps compared, so the
    psi memory is live by construction -- asserted in _operators, because a
    gate that cannot reach the PML is bitwise for any implementation.
    """
    a_core, a_oracle = _operators()
    assert np.array_equal(a_core, a_oracle), (
        f"max |core - oracle| = {np.abs(a_core - a_oracle).max():.3e}, not bitwise")


# ------------------------------------------ the axial factor, from the curl --
# Three routes to the 24-entry component table. Route 1, oracle.PARITY, is a
# table typed from the textbook rule "H picks up det(R)". Route 2, core's
# field_transform, is that same rule as a formula, R.det() * R. Two routes that
# share a source are one route (the Fornberg lesson), so route 3 here uses no
# rule and no table: it transforms a generic POLAR field, E'(x) = R E(R^T x),
# pushes sympy.vector's curl through the map, and reads off the scalar s in
#     curl E' = s * R * (curl E)(R^T x).
# Whatever s is, that is how a curl of a polar field transforms, and H is a
# curl of E (Faraday). One typed fact remains on every route -- E is polar,
# because the force on a charge anchors it -- and it cannot be removed:
# deriving the family from maxwell_rhs is circular (E = curl H / eps gives
# s*s = 1, consistent with either assignment and anchoring neither).

def _curl_pushforward_factor(R):
    """``s`` with ``curl(R E(R^T x)) == s * R * (curl E)(R^T x)``, sympy.vector only."""
    C = CoordSys3D("C")
    x, y, z = C.base_scalars()
    B = C.base_vectors()
    back = dict(zip((x, y, z), R.T * sp.Matrix([x, y, z])))
    E = [sp.Function(n)(x, y, z) for n in ("Ex", "Ey", "Ez")]
    E_at_back = sp.Matrix([e.subs(back, simultaneous=True) for e in E])
    E_prime = R * E_at_back
    as_vec = lambda comps: sum((c * b for c, b in zip(comps, B)), sp.vector.Vector.zero)
    curl_E = sp.Matrix([_vcurl(as_vec(E)).dot(b) for b in B]).subs(back, simultaneous=True)
    curl_Ep = sp.Matrix([_vcurl(as_vec(E_prime)).dot(b) for b in B])
    s = sp.Symbol("s")
    residual = curl_Ep - s * R * curl_E
    (sol,) = sp.solve(next(r for r in residual if r != 0), s)
    assert all(sp.simplify(r.subs(s, sol)) == 0 for r in residual), (R, sol)
    return sol


def test_axial_factor_is_derived_from_the_curl():
    factors = {}
    for kind, R in oracle.SYMMETRY_MATRICES.items():
        s = factors[kind] = _curl_pushforward_factor(R)
        for fam, Mat in (("E", R), ("H", s * R)):
            for i, a in enumerate("xyz"):
                comp = f"{fam}{a}"
                derived = {f"{fam}{b}": int(Mat[i, j]) for j, b in enumerate("xyz") if Mat[i, j] != 0}
                assert derived == oracle.PARITY[kind][comp], (kind, comp, "typed table")
                assert derived == {k: int(v) for k, v in core.component_transform(kind, comp).items()}, (kind, comp, "core")
    # The signal lives in the improper elements and nowhere else. Rotations
    # have s = +1, so for them polar and axial coincide and the test cannot
    # tell R from det(R) R; a symmetry module that got every mirror wrong
    # would pass every rotation case. Deleting the mirror cases as "redundant
    # with the rotations" would make this test vacuous, so say so here.
    assert {k for k, s in factors.items() if s == -1} == {"mirror_x", "mirror_y"}, factors
    assert {k for k, s in factors.items() if s == +1} == {"rot2_z", "rot4_z"}, factors
