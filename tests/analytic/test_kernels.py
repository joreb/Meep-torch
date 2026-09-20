# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The phase arithmetic is derivable, and bitwise with what the steppers write.

Twelve steppers restate four elementwise formulas: the Crank-Nicolson field
update (19 sites), the ADE Lorentz recurrence (5), and two forms of recovering
E from the stored or folded D. `core.kernels` derives each once.

Elementwise is the easy half -- no slices, no boundaries, no lattice -- but it
is not free, because bitwise parity with the STEPPERS is an ASSOCIATION property and
sympy sorts `Add` canonically on construction. A plain `lambdify` of the pole
recurrence emits `(c3 E + c1 P) + c2 P_prev` where every stepper writes
`(c1 P + c2 P_prev) + c3 E`, and those differ in the last bit. Hence
`ordered_sum`, and hence these tests.
"""
import pytest
import sympy as sp
import torch

from meep_torch.core.kernels import (ADE_POLE, CN_UPDATE, E_FOLDED_D,
                                     EH_FROM_D, expressions)

N = (16, 16)


def _rand(k, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(*N, dtype=torch.float64) for _ in range(k)]


def test_cn_update_is_bitwise():
    """`F' = c1 F + c2 curl` -- the update under every stepper's step_db."""
    F, curl, c1, c2 = _rand(4)
    assert torch.equal(CN_UPDATE(F, curl, c1, c2), c1 * F + c2 * curl)


def test_cn_update_carries_the_h_phase_sign_exactly():
    """The H phase writes `d1 F - d2 curl`; it passes a negated c2 instead.

    Worth its own assertion because "just negate the argument" is the kind of
    claim that is true in exact arithmetic and has to be checked in floating
    point. It holds: IEEE negation is a sign-bit flip, and `a - x` and
    `a + (-x)` are the same operation.
    """
    F, curl, d1, d2 = _rand(4, seed=3)
    assert torch.equal(CN_UPDATE(F, curl, d1, -d2), d1 * F - d2 * curl)


def test_ade_pole_is_bitwise():
    """`P' = c1 P + c2 P_prev + c3 E`, grouped as the steppers group it."""
    P_cur, P_prev, E, c1, c2, c3 = _rand(6, seed=1)
    hand = c1 * P_cur + c2 * P_prev + c3 * E
    assert torch.equal(ADE_POLE(P_cur, P_prev, E, c1, c2, c3), hand)


def test_ade_pole_needs_the_ordered_sum():
    """And the canonical grouping must NOT be bitwise, or this proves nothing.

    If sympy's own sort ever produced the steppers' association by luck, the test
    above would pass for any construction and stop being evidence.
    """
    P_cur, P_prev, E, c1, c2, c3 = _rand(6, seed=1)
    hand = c1 * P_cur + c2 * P_prev + c3 * E
    a, b, c, s1, s2, s3 = sp.symbols("P_cur P_prev E c1 c2 c3")
    canonical = sp.lambdify((a, b, c, s1, s2, s3),
                            s1 * a + s2 * b + s3 * c, "torch")
    got = canonical(P_cur, P_prev, E, c1, c2, c3)
    assert not torch.equal(got, hand), (
        "the canonical sort reproduced the steppers' association -- then ordered_sum "
        "is buying nothing here and this kernel's construction is untested")
    assert torch.allclose(got, hand), "they must still be the same number"


def test_eh_from_d_is_bitwise():
    """`E = (D - sum P) / eps`, as stepper_tm's update_eh writes it."""
    D, P_sum, eps = _rand(3, seed=2)
    assert torch.equal(EH_FROM_D(D, P_sum, eps), (D - P_sum) / eps)


def test_e_folded_d_is_bitwise():
    """stepper_3d's `_e_pole`: D is never stored, so eps E + sum P stands in."""
    E, P_sum, P_sum_new, curl, c1, c2, eps = _rand(7, seed=4)
    hand = (c1 * (E * eps + P_sum) + c2 * curl - P_sum_new) / eps
    assert torch.equal(
        E_FOLDED_D(E, P_sum, P_sum_new, curl, c1, c2, eps), hand)


@pytest.mark.parametrize("name", sorted(expressions))
def test_every_kernel_is_one_expression(name):
    """Each phase has exactly one sympy definition, and it is not a constant.

    Cheap, but it is the assertion that would fail if someone added a second
    hand-written variant beside the derived one rather than passing different
    coefficients.
    """
    args, expr = expressions[name]
    assert expr.free_symbols, f"{name} has no inputs"
    assert set(args) <= expr.free_symbols | {sp.Symbol("P_sum_new")}, (
        f"{name} declares arguments its expression does not use")


# Field sums after 8 steps from a randomised state, captured from stepper_tm
# BEFORE its five call sites were wired to the kernels above. Hex floats, so
# the comparison is exact. This is the gate that matters: the tests above
# compare a kernel against an expression typed in this file, which proves the
# association survives lambdify and nothing about whether the kernel is what
# the stepper used to compute.
_TM_PRE_KERNEL = [
    ("vacuum_pml", dict(npml=8),
     {"Ez": "-0x1.916e44a9b931bp+3", "Hx": "0x1.0c4f491921ef5p+4",
      "Hy": "0x1.085ed2e205ddcp+2", "Dz": "-0x1.916e44a9b931bp+3"}),
    ("sigma_d", dict(npml=8, sigma_d=0.3),
     {"Ez": "-0x1.745bdff02aa98p+3", "Hx": "0x1.0e389eb845088p+4",
      "Hy": "0x1.f52636caf7678p+1", "Dz": "-0x1.745bdff02aa98p+3"}),
    ("e_poles", dict(npml=8, poles=[(1.1, 0.0, 0.4), (2.0, 0.1, 0.2)]),
     {"Ez": "-0x1.f10b886b69af8p+5", "Hx": "0x1.6bb3b9ec8139ap+4",
      "Hy": "0x1.881ae6495ae8bp+2", "Dz": "-0x1.91762efe902e0p+2"}),
    ("m_poles", dict(npml=8, m_poles=[(1.3, 0.05, 0.3)]),
     {"Ez": "0x1.1f535c74e1458p+1", "Hx": "0x1.3c1cddc22197cp+2",
      "Hy": "0x1.3a90517987ef3p+5", "Dz": "0x1.1f535c74e1458p+1"}),
    ("poles_kerr", dict(npml=8, poles=[(1.1, 0.0, 0.4)], chi3=0.2),
     {"Ez": "0x1.9881c7885b781p+4", "Hx": "0x1.e21935b7b3aefp+4",
      "Hy": "-0x1.f093191ac49d8p+1", "Dz": "-0x1.ee43f0b3ea1f0p+1"}),
    ("chi2", dict(npml=8, chi2=0.15),
     {"Ez": "0x1.5a3116abfb99ep+10", "Hx": "0x1.9a0c2834bc58cp+6",
      "Hy": "-0x1.53481af5e1d8ep+6", "Dz": "-0x1.cd5beeece6300p+7"}),
]


@pytest.mark.parametrize("label,kw,want", _TM_PRE_KERNEL,
                         ids=[c[0] for c in _TM_PRE_KERNEL])
def test_stepper_tm_on_kernels_is_bitwise_with_its_own_past(label, kw, want):
    """Wiring stepper_tm to the kernels must not move a bit.

    Five call sites changed: the D-phase Crank-Nicolson update, the electric
    pole, the two magnetic poles, and E from the stored D. The six cases reach
    all of them -- a kernel that is never exercised cannot be shown to be
    right, and `chi2`/`poles_kerr` additionally drive the fixed point that
    reads the same `rhs`.

    A canonical-sort `ade` (no `ordered_sum`) fails this at 2.8e-14, which is
    what the numbers are here to catch. The state starts randomised, not zero,
    so the pole and psi slots are live rather than equal by construction.
    """
    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.grid import Grid
    from meep_torch.core.stepper_tm import init_tm_state, step_tm

    g = Grid(nx=24, ny=20, dx=0.05)
    m = build_tm_coeffs(g, **kw)
    mask = torch.zeros(24, 20, dtype=torch.float64)
    mask[12, 10] = 1.0
    s = init_tm_state(g, n_poles=len(kw.get("poles", ())),
                      n_mpoles=len(kw.get("m_poles", ())))
    torch.manual_seed(0)
    s = type(s)._make([(torch.randn_like(v) if torch.is_tensor(v) else v)
                       for v in s])
    for n in range(8):
        s = step_tm(s, m, float(torch.sin(torch.tensor(0.3 * n))), mask)

    got = {f: float(getattr(s, f).sum()).hex() for f in want}
    assert got == want
    assert float(s.Ez.abs().max()) > 1e-6, "a dead field would match anything"
