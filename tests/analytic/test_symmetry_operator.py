# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""A symmetry of the discrete system is ``R A R^dag = A``, and it is measurable.

The steppers implement a symmetry as a domain fold: keep half the cell, supply
a ghost beyond the cut, de-mirror on output. That is the symmetry APPLIED. The
symmetry ITSELF is the statement that the one-step update operator ``A``
commutes with the symmetry operator ``R``,

    R A R^dag = A        equivalently        [A, R] = 0

where ``R`` factors into two pieces that are usually discussed separately:

    the VALUE  -- what each field component becomes. An orthogonal matrix on
                  the component index, with E a polar vector (``R E``) and H an
                  axial one (``det(R) R H``). Derived in core.symbolic.
    the ADDRESS -- which grid sample is the partner. A permutation of grid
                  points; `flip` and `transpose` are that permutation applied
                  in O(N) rather than materialised.

``tests/mp_parity/test_symmetry_parity.py`` already compares a folded run
against a full one at 1e-15, so this file is not here to DETECT a broken fold.
It is here because the commutator is *intrinsic* -- it needs no reference run,
and its residual is supported on the specific cells and components that break,
which a whole-field comparison cannot tell you.

Forming ``A`` and ``R`` is an O(N^2) thing to do and belongs nowhere near a
real run. At test scale it is 408x408.
"""
import numpy as np
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.core.symbolic import component_sign

N_CELL = 12
_COMPONENTS = ("Ez", "Hx", "Hy")


@pytest.fixture(scope="module")
def system():
    """``(A, shapes, sizes)`` for one TM step on a small vacuum cell.

    No PML: a CPML is not mirror-symmetric cell-for-cell (its low-edge profile
    sits half a cell off its high-edge mirror -- see the note in
    test_symmetry_parity), so with one the operator genuinely does not commute
    and the test would be measuring the absorber, not the fold.
    """
    g = Grid(nx=N_CELL, ny=N_CELL, dx=0.05)
    m = build_tm_coeffs(g, npml=0)
    s0 = init_tm_state(g)
    shapes = {c: tuple(getattr(s0, c).shape) for c in _COMPONENTS}
    sizes = {c: int(np.prod(shapes[c])) for c in _COMPONENTS}
    n = sum(sizes.values())

    def pack(s):
        return np.concatenate([getattr(s, c).numpy().ravel() for c in _COMPONENTS])

    def unpack(v):
        off, fields = 0, {}
        for c in _COMPONENTS:
            fields[c] = torch.tensor(v[off:off + sizes[c]].reshape(shapes[c]),
                                     dtype=g.dtype)
            off += sizes[c]
        # Dz = Ez holds in vacuum with no dispersion, and the step preserves
        # it -- asserted below rather than assumed, because a state variable
        # quietly leaving the subspace would make every column of A wrong.
        return s0._replace(Dz=fields["Ez"].clone(), **fields)

    probe = unpack(np.zeros(n))
    stepped = step_tm(probe, m)
    assert np.abs(pack(stepped)).max() == 0.0, (
        "the step is not linear: it moves the zero state, so extracting A "
        "column by column would drop a constant term")

    rng = np.random.default_rng(0)
    stepped = step_tm(unpack(rng.standard_normal(n)), m)
    assert torch.equal(stepped.Dz, stepped.Ez), (
        "Dz left the Dz == Ez subspace, so the packed state is not closed "
        "under the step")
    for psi in ("psi_Ezy", "psi_Ezx", "psi_Hyx", "psi_Hxy"):
        assert torch.equal(getattr(stepped, psi),
                           torch.zeros_like(getattr(stepped, psi))), (
            f"{psi} became nonzero without a PML")

    a = np.zeros((n, n))
    for col in range(n):
        e = np.zeros(n)
        e[col] = 1.0
        a[:, col] = pack(step_tm(unpack(e), m))
    return a, shapes, sizes


def reflection_operator(axis, shapes, sizes, signs=None, shift=0):
    """``R`` for the mirror about ``axis``, as a dense matrix.

    ``signs`` overrides the derived component signs and ``shift`` offsets the
    permutation -- both exist so the test can prove the commutator actually
    responds to a wrong ``R``.
    """
    if signs is None:
        signs = {c: component_sign(f"mirror_{axis}", c) for c in _COMPONENTS}
    n = sum(sizes.values())
    r = np.zeros((n, n))
    off = 0
    for comp in _COMPONENTS:
        nx, ny = shapes[comp]
        extent = nx if axis == "x" else ny
        for i in range(nx):
            for j in range(ny):
                mirrored = (extent - 1 - (i if axis == "x" else j) + shift) % extent
                src = off + i * ny + j
                dst = off + (mirrored * ny + j if axis == "x"
                             else i * ny + mirrored)
                r[dst, src] = signs[comp]
        off += sizes[comp]
    return r


@pytest.mark.parametrize("axis", ["x", "y"])
def test_step_operator_commutes_with_the_mirror(axis, system):
    """``R A R^dag = A``, exactly."""
    a, shapes, sizes = system
    r = reflection_operator(axis, shapes, sizes)
    np.testing.assert_allclose(r @ r.T, np.eye(sum(sizes.values())), atol=0)
    residual = np.abs(r @ a @ r.T - a).max()
    assert residual == 0.0, (
        f"the {axis}-mirror does not commute with the step: "
        f"||R A R^dag - A||_max = {residual:.3e}")


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_commutator_responds_to_a_wrong_operator(axis, system):
    """A gate that cannot fail is not a gate.

    Every entry below is a mistake someone could plausibly make writing the
    symmetry by hand -- dropping the ``det(R)`` that distinguishes E from H,
    getting one component's parity backwards, or landing the reflected index
    one cell off. Each must show up in the commutator.
    """
    a, shapes, sizes = system
    derived = {c: component_sign(f"mirror_{axis}", c) for c in _COMPONENTS}
    wrong = {
        "det(R) dropped (H treated as polar)": dict(Ez=1, Hx=1, Hy=1),
        "Hx parity flipped": {**derived, "Hx": -derived["Hx"]},
        "Hy parity flipped": {**derived, "Hy": -derived["Hy"]},
    }
    for label, signs in wrong.items():
        r = reflection_operator(axis, shapes, sizes, signs=signs)
        assert np.abs(r @ a @ r.T - a).max() > 0.1, (
            f"{axis}-mirror, {label}: commutator stayed zero, so this test "
            f"would not catch the mistake")

    r = reflection_operator(axis, shapes, sizes, shift=1)
    assert np.abs(r @ a @ r.T - a).max() > 0.1, (
        f"{axis}-mirror with the reflection one cell off: commutator stayed "
        f"zero")


@pytest.mark.parametrize("axis", ["x", "y"])
def test_the_phase_selects_the_sector_not_the_symmetry(axis, system):
    """``Mirror(X, phase=-1)`` is not part of the symmetry statement.

    The phase appears in both ``R`` and ``R^dag`` and cancels, so even and odd
    give the same commutator -- the fold is a property of the operator, and
    the phase picks WHICH eigenspace of R the solution is confined to
    (projector ``(I + p R)/2``). Confusing the two is how a parity ends up
    applied twice or not at all.
    """
    a, shapes, sizes = system
    derived = {c: component_sign(f"mirror_{axis}", c) for c in _COMPONENTS}
    even = reflection_operator(axis, shapes, sizes, signs=derived)
    odd = reflection_operator(axis, shapes, sizes,
                              signs={c: -v for c, v in derived.items()})
    np.testing.assert_allclose(odd, -even, atol=0)
    np.testing.assert_allclose(odd @ a @ odd.T, even @ a @ even.T, atol=0)
