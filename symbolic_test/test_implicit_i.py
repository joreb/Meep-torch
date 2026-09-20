# SPDX-License-Identifier: MIT
# The implicit-i derivation (typed textbook system) against the rule the
# factory applies to a phased term. The oracle module imports nothing from
# meep_torch; this file may import both.
import sympy as sp

from meep_torch.core import symbolic as core

from . import implicit_i as oracle


def test_the_beta_system_is_complex_and_couples_the_polarisations():
    M = oracle.beta_system()
    assert not oracle.is_real(M)
    # every beta entry crosses the TM/TE line
    for (t, s), c in oracle.beta_couplings(imaginary=()).items():
        assert (t in oracle.TM) != (s in oracle.TM), (t, s)


def test_storing_exactly_one_polarisation_as_imaginary_makes_it_real():
    assert oracle.is_real(oracle.stored_system(oracle.TM))
    assert oracle.is_real(oracle.stored_system(oracle.TE))
    assert not oracle.is_real(oracle.stored_system(()))
    assert not oracle.is_real(oracle.stored_system(oracle.TM + oracle.TE))
    assert not oracle.is_real(oracle.stored_system(("Ez",)))     # half a block


def test_the_real_couplings_are_meeps_beta_terms():
    """Four terms, Dx<-Hy, Dy<-Hx, Bx<-Ey, By<-Ex, and no z-component."""
    got = oracle.beta_couplings(oracle.TM)
    assert set(got) == {("Ex", "Hy"), ("Ey", "Hx"), ("Hx", "Ey"), ("Hy", "Ex")}
    # the pair on one direction has opposite signs, X against Y, as
    # step_db.cpp's `(d_c == X ? +1 : -1)` has it
    assert got[("Ex", "Hy")] == -got[("Ey", "Hx")]
    assert got[("Hx", "Ey")] == -got[("Hy", "Ex")]


def test_the_factory_rule_reproduces_the_derivation():
    """``sign * k * i^(a-b+1)``, ``a``/``b`` = target/source stored as ``i*``.

    This is the rule `_phased_term` applies; here it is checked against the
    conjugated system rather than typed twice.
    """
    for imaginary in (oracle.TM, oracle.TE):
        want = oracle.beta_couplings(imaginary)
        got = {}
        for target in core.COMPONENTS:
            for src, axis, sign in core.curl_terms(target):
                if axis != "z":
                    continue
                a, b = target in imaginary, src in imaginary
                assert a != b, (target, src)
                got[(target, src)] = sign * (1 if b else -1)
        assert got == want, (imaginary, got, want)


def test_the_1d_reduction_keeps_stepper_kz_three_components_closed():
    """The task page's open question: does `stepper_kz` (Ex, Ez, Hy; 1D in x)
    drop a coupling meep's six-component 2D-with-beta block keeps? With
    `d/dy = 0` the beta system splits into {Ex, Ez, Hy} and {Ey, Hx, Hz} with
    no cross terms, and the first block is exactly the stepper's docstring:
    `dDx/dt = -i kz Hy`, `dDz/dt = dHy/dx`, `dBy/dt = -i kz Ex + dEz/dx`.
    So the 1D stepper is a legitimate reduction, and the six-component block
    is needed only where `d/dy != 0`."""
    from .maxwell_sympy import COMPONENTS, D, EPS
    M = oracle.beta_system().subs(D["y"], 0)
    idx = {c: i for i, c in enumerate(COMPONENTS)}
    a, b = ("Ex", "Ez", "Hy"), ("Ey", "Hx", "Hz")
    for t in a:
        for s in b:
            assert M[idx[t], idx[s]] == 0 and M[idx[s], idx[t]] == 0, (t, s)
    assert M[idx["Ex"], idx["Hy"]] == -sp.I * oracle.BETA / EPS
    assert M[idx["Ez"], idx["Hy"]] == D["x"] / EPS
    assert M[idx["Hy"], idx["Ex"]] == -sp.I * oracle.BETA
    assert M[idx["Hy"], idx["Ez"]] == D["x"]
