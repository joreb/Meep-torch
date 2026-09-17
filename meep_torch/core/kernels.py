# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""The pointwise arithmetic of meep's step phases, defined once.

Twelve steppers restate the same four formulas. They are elementwise -- they
act on tensors the stencil layer has already gathered -- so unlike the stencils
they carry no slicing, no boundary and no lattice, and one definition covers
every reduction. The stencil half lives in :mod:`meep_torch.core.symbolic`
(`compiled_difference`, `compiled_curl_composition`); this is the other half.

The CPML recursion already had exactly one definition, as `PSI_NEW` and
`STRETCHED` in `symbolic`. It is re-exported here so a stepper has one import
for the phase arithmetic rather than two.

ASSOCIATION, and it bit immediately. A plain `lambdify` of the three-term pole
recurrence emits `E*c3 + P_cur*c1 + P_prev*c2` -- sympy's canonical sort -- and
that groups as `(c3 E + c1 P) + c2 P_prev` where every stepper writes
`(c1 P + c2 P_prev) + c3 E`. Reordering the factors of a `Mul` is exact, and so
is reordering the two arguments of one `Add`; regrouping three terms is not. So
every sum of more than two terms here is built with `ordered_sum`, the same
nested-binary construction `compiled_curl_composition` needs, and
`tests/analytic/test_kernels.py` asserts bitwise against the hand form rather
than assuming it.

SIGNS are the caller's. `cn_update` is `c1 F + c2 curl`; the H phase passes a
negated `c2` rather than getting a second kernel, which is exact -- IEEE
negation and `a - x == a + (-x)` are both bit-preserving -- and gated.
"""
import sympy as sp

from .symbolic import PSI_NEW, STRETCHED, ordered_sum  # noqa: F401

__all__ = ["PSI_NEW", "STRETCHED", "CN_UPDATE", "ADE_POLE", "EH_FROM_D",
           "E_FOLDED_D", "expressions"]


def _compile():
    F, curl, c1, c2, c3 = sp.symbols("F curl c1 c2 c3")
    P_cur, P_prev, E, D, P_sum, P_sum_new, eps = sp.symbols(
        "P_cur P_prev E D P_sum P_sum_new eps")

    # Crank-Nicolson field update: the conductivity-free case is c1=1, c2=dt.
    cn = c1 * F + c2 * curl
    # ADE Lorentz pole, meep's update_pols, read off E before this step's
    # update_eh (the ordering is the stepper's business, not the kernel's).
    # ordered_sum, not `+`: see the association note above.
    ade = ordered_sum([c1 * P_cur, c2 * P_prev, c3 * E])
    # E from the stored D with the polarisation removed.
    eh = (D - P_sum) / eps
    # Folded-D form: D is never stored, so eps_inf E + sum P stands in for it.
    # E(n+1) = (c1 (eps E^n + sum P^n) + c2 curl - sum P^(n+1)) / eps
    e_folded = ordered_sum([c1 * (E * eps + P_sum), c2 * curl,
                            -P_sum_new]) / eps

    return {
        "cn_update": ((F, curl, c1, c2), cn),
        "ade_pole": ((P_cur, P_prev, E, c1, c2, c3), ade),
        "eh_from_d": ((D, P_sum, eps), eh),
        "e_folded_d": ((E, P_sum, P_sum_new, curl, c1, c2, eps), e_folded),
    }


expressions = _compile()
_FNS = {k: sp.lambdify(args, expr, "torch")
        for k, (args, expr) in expressions.items()}

#: ``cn_update(F, curl, c1, c2)`` -- ``c1 F + c2 curl``.
CN_UPDATE = _FNS["cn_update"]
#: ``ade_pole(P_cur, P_prev, E, c1, c2, c3)`` -- the Lorentz recurrence.
ADE_POLE = _FNS["ade_pole"]
#: ``eh_from_d(D, P_sum, eps)`` -- ``(D - sum P) / eps``.
EH_FROM_D = _FNS["eh_from_d"]
#: ``e_folded_d(E, P_sum, P_sum_new, curl, c1, c2, eps)`` -- the D-free form.
E_FOLDED_D = _FNS["e_folded_d"]
