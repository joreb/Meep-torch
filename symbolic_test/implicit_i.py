# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep's implicit-i trick, derived from the typed Maxwell system.

A 2D cell with an ``e^{i beta z}`` dependence turns ``d/dz`` into ``i beta``
and couples the two polarisations through an ``i beta zhat x`` term. meep
(``src/step_db.cpp:147-160``) keeps REAL arrays anyway by storing ``i*(TM
fields)`` in place of the TM fields, so the ``i``'s cancel in the update.

This module derives that from :mod:`maxwell_sympy` alone: substitute
``D_z -> i beta`` in the six-by-six system, conjugate by the diagonal
``S = diag(i on the imaginary-stored components, 1 elsewhere)``, and read off
which choices of stored-as-imaginary components leave a real matrix. Imports
nothing from ``meep_torch``; the core rule is checked against this in
``test_implicit_i.py``.
"""
import sympy as sp

from .maxwell_sympy import COMPONENTS, D, system_matrix

BETA = sp.Symbol("beta", real=True)

#: meep's choice: the Ez polarisation is stored as ``i *`` itself.
TM = ("Ez", "Hx", "Hy")
TE = ("Ex", "Ey", "Hz")


def beta_system():
    """``dF/dt = M F`` for a 2D cross-section with ``d/dz = i beta``."""
    return system_matrix().subs(D["z"], sp.I * BETA)


def stored_system(imaginary=TM):
    """The system in the STORED variables ``F' = S F``, ``S = diag(i or 1)``.

    ``dF'/dt = S M S^{-1} F'``. Real iff the ``i`` factors cancel on every
    coupling, which is the whole trick.
    """
    s = sp.diag(*[sp.I if c in imaginary else 1 for c in COMPONENTS])
    return sp.simplify(s * beta_system() * s.inv())


def is_real(M):
    """No explicit ``i`` survives in any entry. The D operators and ``eps``
    are plain symbols, so ``sp.im`` cannot decide this; the trick is about
    the factor ``I``, and that is what is looked for."""
    return not any(sp.expand(e).has(sp.I) for e in M)


def beta_couplings(imaginary=TM):
    """``{(target, source): real coefficient of beta}`` in the stored system.

    Only the entries that carry ``beta``: the in-plane differences are
    untouched by the trick and stay where they were.
    """
    M = stored_system(imaginary)
    out = {}
    for i, target in enumerate(COMPONENTS):
        for j, source in enumerate(COMPONENTS):
            e = sp.expand(M[i, j])
            c = e.coeff(BETA)
            if c != 0:
                assert sp.simplify(e - c * BETA) == 0, (target, source, e)
                out[(target, source)] = sp.nsimplify(c.subs(sp.Symbol("eps", positive=True), 1))
    return out
