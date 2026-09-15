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


def test_operator_curl_matches_the_typed_curl():
    """All nine signs, against a curl TYPED OUT here from the definition.

    The independent side has to be typed. When this test was written,
    symbolic.py entered the nine signs by hand and ``sympy.vector.curl`` was
    the independent check. symbolic.py now *is* ``sympy.vector.curl``, so
    comparing against it again would compare sympy.vector with sympy.vector
    and pass no matter what the nine signs were -- the test would still be
    green while checking only the operatorise/realise round trip.

    So the reference below is written out longhand. It is the thing being
    trusted, and it is deliberately the most boring code in this file.
    """
    C = CoordSys3D("C")
    fns = [sp.Function(n)(C.x, C.y, C.z) for n in _NAMES]
    syms = {n: sp.Symbol(n) for n in _NAMES}
    coord = {"x": C.x, "y": C.y, "z": C.z}
    d = lambda f, u: sp.Derivative(f, coord[u])
    Ex, Ey, Ez = fns

    # curl F = (dFz/dy - dFy/dz,  dFx/dz - dFz/dx,  dFy/dx - dFx/dy)
    typed = (
        d(Ez, "y") - d(Ey, "z"),
        d(Ex, "z") - d(Ez, "x"),
        d(Ey, "x") - d(Ex, "y"),
    )

    mine = operator_curl((syms["Ex"], syms["Ey"], syms["Ez"]))
    for component, want, axis in zip(mine, typed, AXES):
        got = _realise(component, syms, fns, coord)
        assert sp.simplify(got - want) == 0, (
            f"curl {axis}-component disagrees with the typed curl:\n"
            f"  symbolic.py: {got}\n  typed: {want}")

    # and the typed reference itself agrees with sympy.vector, so a mistake in
    # the longhand above cannot quietly become the accepted answer
    field = fns[0] * C.i + fns[1] * C.j + fns[2] * C.k
    reference = vector_curl(field)
    for want, basis, axis in zip(typed, (C.i, C.j, C.k), AXES):
        assert sp.simplify(want - reference.dot(basis)) == 0, (
            f"the typed reference is itself wrong on {axis}")


def test_two_dimensional_reduction_decouples():
    """TM and TE exist because the reduced system is block-diagonal, not
    because anything declared them.

    The membership is now read off the coupling graph rather than checked
    against a typed partition, so the blocks come back in COMPONENTS order.
    Sets, not tuples, are what this test cares about.
    """
    assert_decoupled(("z",))
    blocks = polarisation_blocks(("z",))
    assert set(blocks["tm"]["components"]) == {"Ez", "Hx", "Hy"}
    assert set(blocks["te"]["components"]) == {"Hz", "Ex", "Ey"}
    # the two blocks together account for all six components, exactly once
    members = blocks["tm"]["components"] + blocks["te"]["components"]
    assert sorted(members) == sorted(COMPONENTS)


def test_the_partition_is_derived_not_assumed():
    """The z case agreeing with the typed partition is not the claim.

    The claim is that membership comes from the matrix, and the way to show
    that is a reduction the typed partition gets WRONG. Under d/dx = 0 the
    system decouples just as cleanly, into different blocks; a check against a
    z-shaped partition called that a failure to decouple. And 1D is four
    blocks, not a TM/TE pair, which the two-way answer could only misreport.
    """
    from meep_torch.core.symbolic import decoupled_blocks

    assert decoupled_blocks(("x",)) == (("Ex", "Hy", "Hz"), ("Ey", "Ez", "Hx"))
    assert decoupled_blocks(("y",)) == (("Ex", "Ez", "Hy"), ("Ey", "Hx", "Hz"))
    assert decoupled_blocks(("y", "z")) == (("Ex",), ("Ey", "Hz"),
                                            ("Ez", "Hy"), ("Hx",))
    # unreduced Maxwell is fully coupled: one block, and no polarisations
    assert len(decoupled_blocks(())) == 1
    for bad in ((), ):
        try:
            assert_decoupled(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"assert_decoupled({bad}) should refuse")
    try:
        polarisation_blocks(("y", "z"))
    except ValueError:
        pass
    else:
        raise AssertionError("1D has four blocks; TM/TE must not be forced")


def _run_tm(steps, npml):
    """A short TM run with PML live, so the CPML stretch and psi memory are
    actually exercised -- without a PML both curl forms reduce to the same
    unstretched difference and the comparison is vacuous."""
    import torch

    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.grid import Grid
    from meep_torch.core.stepper_tm import init_tm_state, step_tm

    g = Grid(nx=60, ny=60, dx=0.05)
    m = build_tm_coeffs(g, npml=npml)
    s = init_tm_state(g)
    torch.manual_seed(0)
    ez = torch.zeros(g.nx, g.ny, dtype=g.dtype, device=g.device)
    ez[g.nx // 2, g.ny // 2] = 1.0
    s = s._replace(Ez=ez, Dz=ez.clone())
    for _ in range(steps):
        s = step_tm(s, m)
    return s.Ez


def test_whole_curl_and_meep_association_agree(monkeypatch):
    """Both sides of the gate, exercised.

    The whole-curl form is the default; MEEP_TORCH_WHOLE_CURL=0 restores meep's
    bitwise left-to-right association. Whichever is the non-default, it is an
    unreached branch -- and an unreached branch is where the
    compiled_uniform_difference argument-order bug hid in stepper_kz while
    reporting 0.000e+00. So run both. Deliberately time-domain: solve_cw's
    BiCGSTAB(2) has a residual floor just above 1e-10 on the whole-curl form,
    which is a conditioning property and would read here as a field error.
    """
    import torch

    from meep_torch.core import symbolic

    monkeypatch.delenv("MEEP_TORCH_WHOLE_CURL", raising=False)
    assert symbolic.whole_curl_enabled(), "whole curl is the default"
    whole = _run_tm(steps=40, npml=8)

    monkeypatch.setenv("MEEP_TORCH_WHOLE_CURL", "0")
    assert not symbolic.whole_curl_enabled()
    meep_order = _run_tm(steps=40, npml=8)

    # Not bitwise: sympy assembles its own canonical Add rather than meep's
    # left-to-right association. One ulp apart, accumulated over 40 steps.
    assert meep_order.abs().max() > 0
    rel = (torch.linalg.vector_norm(whole - meep_order)
           / torch.linalg.vector_norm(meep_order))
    assert rel < 1e-13, f"the two associations disagree by {rel:.3e}"
    assert not torch.equal(whole, meep_order), (
        "the two forms came out bitwise-equal -- the gate is not switching the "
        "code path, or the PML is not live")


def test_symmetry_signs_come_from_the_transform_not_a_table():
    """The mirror/rotation signs each field component picks up.

    These were a typed table in mp.simulation._comp_parity. They are a
    consequence of two facts: E is a polar vector and transforms as ``R E``,
    H is axial and transforms as ``det(R) R H``. Everything below follows
    from that, and the det factor is the sign most easily typed wrong -- it
    is why Hy flips about the x cut where Ez does not.
    """
    from meep_torch.core.symbolic import component_sign, component_transform

    # the table _comp_parity used to carry, reproduced exactly
    assert component_sign("mirror_x", "Ez") == +1
    assert component_sign("mirror_x", "Hx") == +1
    assert component_sign("mirror_x", "Hy") == -1
    assert component_sign("mirror_y", "Ez") == +1
    assert component_sign("mirror_y", "Hx") == -1
    assert component_sign("mirror_y", "Hy") == +1
    assert component_sign("rot2_z", "Ez") == +1
    assert component_sign("rot2_z", "Hx") == -1
    assert component_sign("rot2_z", "Hy") == -1

    # E and H differ by det(R) under a REFLECTION and agree under a ROTATION
    for axis in ("x", "y", "z"):
        for kind in ("mirror_x", "mirror_y"):
            assert (component_sign(kind, f"E{axis}")
                    == -component_sign(kind, f"H{axis}"))
        assert (component_sign("rot2_z", f"E{axis}")
                == component_sign("rot2_z", f"H{axis}"))

    # a quarter turn MIXES the in-plane pair, so no scalar sign exists --
    # component_sign must refuse rather than invent one
    assert component_transform("rot4_z", "Hx") == {"Hy": -1}
    assert component_transform("rot4_z", "Hy") == {"Hx": 1}
    assert component_transform("rot4_z", "Ez") == {"Ez": 1}
    assert component_sign("rot4_z", "Ez") == +1
    for comp in ("Hx", "Hy", "Ex", "Ey"):
        try:
            component_sign("rot4_z", comp)
        except ValueError:
            pass
        else:
            raise AssertionError(f"component_sign('rot4_z', {comp!r}) "
                                 f"should refuse: a quarter turn mixes it")

    # the symmetries are orthogonal and are their own group elements:
    # a mirror and a half turn are involutions, a quarter turn has order 4
    from meep_torch.core.symbolic import SYMMETRIES
    import sympy as sp
    for kind, R in SYMMETRIES.items():
        assert sp.simplify(R.T * R - sp.eye(3)) == sp.zeros(3, 3), kind
    assert SYMMETRIES["rot4_z"] ** 2 == SYMMETRIES["rot2_z"]
    assert SYMMETRIES["rot4_z"] ** 4 == sp.eye(3)
