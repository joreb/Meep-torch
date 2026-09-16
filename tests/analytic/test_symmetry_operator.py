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
from meep_torch.core.symbolic import component_transform

N_CELL = 12
_COMPONENTS = ("Ez", "Hx", "Hy")
_KINDS = ("mirror_x", "mirror_y", "rot2_z", "rot4_z")

# (i, j) in a source grid of shape (nx, ny) -> its index in the image grid.
#
# Every TM grid is symmetric about the cell centre in its OWN index space:
# Ez sits at (i, j), Hx at (i, j+1/2), Hy at (i+1/2, j), and each spans a range
# whose midpoint is the same physical point, so `extent - 1 - index` is the
# reflection for all three despite the half-offsets.
#
# The quarter turn is the one that does not stay inside a component. It sends a
# grid of shape (nx, ny) onto one of shape (ny, nx) -- which is exactly how the
# Hx grid (n, n-1) lands on the Hy grid (n-1, n). A scalar-per-component parity
# cannot express that, and not merely because the number would be wrong: the
# image has the wrong SHAPE.
_POINT_MAP = {
    "mirror_x": lambda i, j, nx, ny: (nx - 1 - i, j),
    "mirror_y": lambda i, j, nx, ny: (i, ny - 1 - j),
    "rot2_z": lambda i, j, nx, ny: (nx - 1 - i, ny - 1 - j),
    "rot4_z": lambda i, j, nx, ny: (ny - 1 - j, i),
}


@pytest.fixture(scope="module")
def system():
    """``(A, shapes, offsets, n)`` for one TM step on a small vacuum cell.

    Square, because a quarter turn needs it. No PML: a CPML is not
    mirror-symmetric cell-for-cell (its low-edge profile sits half a cell off
    its high-edge mirror -- see the note in test_symmetry_parity), so with one
    the operator genuinely does not commute and the test would be measuring
    the absorber, not the fold.
    """
    g = Grid(nx=N_CELL, ny=N_CELL, dx=0.05)
    m = build_tm_coeffs(g, npml=0)
    s0 = init_tm_state(g)
    shapes = {c: tuple(getattr(s0, c).shape) for c in _COMPONENTS}
    offsets, off = {}, 0
    for c in _COMPONENTS:
        offsets[c] = off
        off += int(np.prod(shapes[c]))
    n = off

    def pack(s):
        return np.concatenate([getattr(s, c).numpy().ravel() for c in _COMPONENTS])

    def unpack(v):
        fields = {}
        for c in _COMPONENTS:
            size = int(np.prod(shapes[c]))
            fields[c] = torch.tensor(
                v[offsets[c]:offsets[c] + size].reshape(shapes[c]), dtype=g.dtype)
        # Dz = Ez holds in vacuum with no dispersion, and the step preserves
        # it -- asserted below rather than assumed, because a state variable
        # quietly leaving the subspace would make every column of A wrong.
        return s0._replace(Dz=fields["Ez"].clone(), **fields)

    assert np.abs(pack(step_tm(unpack(np.zeros(n)), m))).max() == 0.0, (
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
    return a, shapes, offsets, n


def symmetry_operator(kind, shapes, offsets, n, rows=None, point_map=None):
    """``R`` for ``kind`` as a dense matrix: component transform x grid permutation.

    ``rows`` (the component transform) and ``point_map`` (the grid
    permutation) can be overridden so a test can prove the commutator responds
    to a wrong ``R``. Returns ``None`` when the override sends a sample out of
    its destination grid -- a shape mismatch, which is its own kind of answer.
    """
    if rows is None:
        rows = {c: component_transform(kind, c) for c in _COMPONENTS}
    if point_map is None:
        point_map = _POINT_MAP[kind]
    r = np.zeros((n, n))
    for dest in _COMPONENTS:
        dnx, dny = shapes[dest]
        for src, coeff in rows[dest].items():
            nx, ny = shapes[src]
            for i in range(nx):
                for j in range(ny):
                    di, dj = point_map(i, j, nx, ny)
                    if not (0 <= di < dnx and 0 <= dj < dny):
                        return None
                    r[offsets[dest] + di * dny + dj,
                      offsets[src] + i * ny + j] = float(coeff)
    return r


@pytest.mark.parametrize("kind", _KINDS)
def test_step_operator_commutes_with_the_symmetry(kind, system):
    """``R A R^dag = A``, exactly, for every symmetry the engine supports."""
    a, shapes, offsets, n = system
    r = symmetry_operator(kind, shapes, offsets, n)
    np.testing.assert_allclose(r @ r.T, np.eye(n), atol=0)
    residual = np.abs(r @ a @ r.T - a).max()
    assert residual == 0.0, (
        f"{kind} does not commute with the step: "
        f"||R A R^dag - A||_max = {residual:.3e}")


def test_the_grid_representation_reproduces_the_group(system):
    """C4 squares to C2 and closes at four, as 408x408 grid operators.

    Not just as 3x3 component matrices -- that would only test sympy. This is
    the statement that the ADDRESS half composes correctly too: a quarter turn
    applied twice must land every staggered sample exactly where the half turn
    puts it, Hx via Hy and back.
    """
    _, shapes, offsets, n = system
    c4 = symmetry_operator("rot4_z", shapes, offsets, n)
    c2 = symmetry_operator("rot2_z", shapes, offsets, n)
    np.testing.assert_array_equal(c4 @ c4, c2)
    np.testing.assert_array_equal(np.linalg.matrix_power(c4, 4), np.eye(n))
    for kind in ("mirror_x", "mirror_y", "rot2_z"):
        r = symmetry_operator(kind, shapes, offsets, n)
        np.testing.assert_array_equal(r @ r, np.eye(n))


@pytest.mark.parametrize("kind", ["mirror_x", "mirror_y", "rot2_z"])
def test_the_commutator_responds_to_wrong_signs(kind, system):
    """A gate that cannot fail is not a gate.

    Each entry is a mistake someone could plausibly make writing the symmetry
    by hand: dropping the ``det(R)`` that distinguishes E from H, getting one
    component's parity backwards, or landing the reflected index one cell off.
    """
    a, shapes, offsets, n = system
    derived = {c: component_transform(kind, c) for c in _COMPONENTS}
    wrong = {
        "det(R) dropped (H treated as polar)":
            {c: {c: 1} for c in _COMPONENTS},
        "Hx parity flipped":
            {**derived, "Hx": {"Hx": -derived["Hx"]["Hx"]}},
        "Hy parity flipped":
            {**derived, "Hy": {"Hy": -derived["Hy"]["Hy"]}},
    }
    for label, rows in wrong.items():
        r = symmetry_operator(kind, shapes, offsets, n, rows=rows)
        assert np.abs(r @ a @ r.T - a).max() > 0.1, (
            f"{kind}, {label}: commutator stayed zero, so this test would not "
            f"catch the mistake")

    shifted = symmetry_operator(
        kind, shapes, offsets, n,
        point_map=lambda i, j, nx, ny: tuple(
            (v + 1) % ext for v, ext in zip(_POINT_MAP[kind](i, j, nx, ny),
                                            (nx, ny))))
    assert shifted is None or np.abs(shifted @ a @ shifted.T - a).max() > 0.1, (
        f"{kind} with the permutation one cell off: commutator stayed zero")


def test_the_quarter_turn_cannot_be_written_as_a_scalar_parity(system):
    """The C4 case that motivated deriving the transform rather than typing it.

    ``Hx -> -Hy`` is off-diagonal, and a per-component sign cannot hold it.
    The failure is not subtle arithmetic: assuming C4 keeps each component to
    itself sends the Hx grid (n, n-1) onto a grid of shape (n-1, n), which is
    not Hx. The shape check catches it before any number is compared.
    """
    a, shapes, offsets, n = system
    diagonal = symmetry_operator(
        "rot4_z", shapes, offsets, n,
        rows={c: {c: 1} for c in _COMPONENTS})
    assert diagonal is None, (
        "a diagonal C4 built without error -- the staggered grids no longer "
        "distinguish it from the correct off-diagonal form")

    good = {c: component_transform("rot4_z", c) for c in _COMPONENTS}
    wrong = {
        "Hx<->Hy coupling both +1": {**good, "Hx": {"Hy": 1}},
        "transpose without the flip": None,
        "rotated the other way": None,
    }
    maps = {
        "transpose without the flip": lambda i, j, nx, ny: (j, i),
        "rotated the other way": lambda i, j, nx, ny: (j, nx - 1 - i),
    }
    for label, rows in wrong.items():
        r = symmetry_operator("rot4_z", shapes, offsets, n,
                              rows=rows or good, point_map=maps.get(label))
        assert r is not None, label
        # orthogonal, and still wrong -- being a valid permutation with unit
        # entries is not the same as being THIS symmetry
        np.testing.assert_allclose(r @ r.T, np.eye(n), atol=0)
        assert np.abs(r @ a @ r.T - a).max() > 0.1, (
            f"rot4_z, {label}: commutator stayed zero")


@pytest.mark.parametrize("kind", _KINDS)
def test_the_phase_selects_the_sector_not_the_symmetry(kind, system):
    """``Mirror(X, phase=-1)`` is not part of the symmetry statement.

    The phase appears in both ``R`` and ``R^dag`` and cancels, so even and odd
    give the same commutator -- the fold is a property of the operator, and
    the phase picks WHICH eigenspace of R the solution is confined to
    (projector ``(I + p R)/2``). Confusing the two is how a parity ends up
    applied twice or not at all.
    """
    a, shapes, offsets, n = system
    even = symmetry_operator(kind, shapes, offsets, n)
    odd = symmetry_operator(
        kind, shapes, offsets, n,
        rows={c: {s: -v for s, v in component_transform(kind, c).items()}
              for c in _COMPONENTS})
    np.testing.assert_array_equal(odd, -even)
    np.testing.assert_array_equal(odd @ a @ odd.T, even @ a @ even.T)


def test_the_residual_localises_the_mistake(system):
    """The reason to prefer the commutator over a folded-vs-full comparison.

    A whole-field comparison says "wrong". The commutator says WHICH coupling.
    Dropping det(R) under the y-mirror changes exactly one derived sign --
    Hx's -- so the residual must be supported on the two blocks coupling Ez to
    Hx and be exactly zero everywhere else, Ez<->Hy included.
    """
    a, shapes, offsets, n = system
    polar = {c: {c: 1} for c in _COMPONENTS}     # det(R) dropped
    r = symmetry_operator("mirror_y", shapes, offsets, n, rows=polar)
    residual = r @ a @ r.T - a

    def block(dest, src):
        d0, s0 = offsets[dest], offsets[src]
        d1 = d0 + int(np.prod(shapes[dest]))
        s1 = s0 + int(np.prod(shapes[src]))
        return np.abs(residual[d0:d1, s0:s1]).max()

    guilty = {("Ez", "Hx"), ("Hx", "Ez")}
    for dest in _COMPONENTS:
        for src in _COMPONENTS:
            got = block(dest, src)
            if (dest, src) in guilty:
                assert got > 0.1, (
                    f"{dest}<-{src} should carry the residual, got {got:.3e}")
            else:
                assert got == 0.0, (
                    f"{dest}<-{src} should be clean, got {got:.3e} -- the "
                    f"residual is not localising the mistake")


# --- the cut-plane ghost, derived from R rather than typed per case ---------

_FOLDS = {
    "mirror_y_odd": dict(y_mirror=-1),
    "mirror_x_odd": dict(x_mirror=-1),
    "rot2_even": dict(y_rot2=+1),
    "rot2_odd": dict(y_rot2=-1),
    "rot4_even": dict(rot4=+1),
    "rot4_odd": dict(rot4=-1),
}


def _folded_run(steps=6, **fold):
    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.grid import Grid
    from meep_torch.core.stepper_tm import init_tm_state, step_tm

    g = Grid(nx=16, ny=16, dx=0.05)
    m = build_tm_coeffs(g, npml=3, **fold)
    s = init_tm_state(g, **fold)
    torch.manual_seed(0)
    s = s._replace(Ez=torch.randn(g.nx, g.ny, dtype=g.dtype))
    s = s._replace(Dz=s.Ez.clone())
    for _ in range(steps):
        s = step_tm(s, m)
    return s


@pytest.mark.parametrize("name", sorted(_FOLDS))
def test_the_derived_ghost_reproduces_every_fold(name):
    """The ghost comes from ``R f = lam f``; it used to be three typed branches.

    The eigen-relation holds pointwise at every node including ghosts, so the
    ghost is the phase times whichever of ``P`` / ``P^-1`` reaches inside the
    fold. This asserts the folds still RUN and stay finite; the bitwise claim
    is the one below, which is what the substitution actually has to preserve.
    """
    s = _folded_run(**_FOLDS[name])
    assert torch.isfinite(s.Ez).all(), f"{name}: Ez went non-finite"
    assert s.Ez.abs().max() > 0, f"{name}: field is identically zero"


def test_the_two_cut_rays_are_one_ray():
    """``Hy0 = conj(lam) Hx0`` -- the fold is not a strict fundamental domain.

    Under C4 the two cut-plane rays are images of each other, so storing both
    is redundant and they must agree exactly. The quarter turn carries the Hx0
    node onto the Hy0 node at the SAME fold index, and
    (M_H)_yx = +1 for the +90-degree convention, so with a real phase the
    relation is Hy0 = lam Hx0. It holds bitwise once the ghosts are derived,
    because both updates then come from the same expression.
    """
    for phase in (+1, -1):
        s = _folded_run(rot4=phase)
        assert torch.equal(s.Hy0, phase * s.Hx0), (
            f"rot4={phase:+d}: the two rays disagree, "
            f"max {(s.Hy0 - phase * s.Hx0).abs().max():.3e}")
        # and it is not vacuous: the rays are nonzero, and the OPPOSITE sign
        # does not also satisfy it
        assert s.Hx0.abs().max() > 0
        assert not torch.equal(s.Hy0, -phase * s.Hx0)


def test_the_component_transform_supplies_the_ray_sign():
    """(M_H)_yx = +1 is read off the transform, not chosen to fit.

    The test above asserts a sign; this is where that sign comes from, so the
    two cannot drift apart silently.
    """
    assert component_transform("rot4_z", "Hy") == {"Hx": 1}
    assert component_transform("rot4_z", "Hx") == {"Hy": -1}
