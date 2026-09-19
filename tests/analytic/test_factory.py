# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""A curl built from the spec, against the one stepper_tm writes by hand.

`make_curl` resolves four separate derivations at build time and nothing at
step time:

    curl_terms                 which (source, axis, sign) terms exist
    compiled_difference        which neighbour is the +1 sample
    psi_extent                 what shape every term lands on
    compiled_curl_composition  which order they combine in

If the result is bitwise with `stepper_tm.step_db("D")` then all four agree
with code that was written by hand, and the remaining difference between a
unified stepper and twelve hand-written ones is plumbing rather than physics.

Bitwise and not merely close: the composition is meep's association, and this
is exactly the property that a canonical `Add` sort would break at 1e-14 while
every tolerance-based check still passed.
"""
import pytest
import torch
import torch.nn.functional as F

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.factory import curl_recipe, make_curl
from meep_torch.core.grid import Grid
from meep_torch.core.kernels import CN_UPDATE
from meep_torch.core.stepper_tm import init_tm_state, step_db
from meep_torch.core.symbolic import FOLD, WALL, BlockSpec

NX, NY = 24, 20

_TM = ("Ez", "Hx", "Hy")


def _spec(n=(NX, NY), bnd=(WALL, WALL), **kw):
    from meep_torch.core.factory import TM_DAMPING
    kw.setdefault("damping", TM_DAMPING)
    return BlockSpec(_TM, n, bnd, **kw)

_CASES = [("vacuum_pml", dict(npml=6)),
          ("sigma_d", dict(npml=6, sigma_d=0.3)),
          ("no_pml", dict(npml=0))]


def _seeded(**kw):
    """A randomised state, so psi and the CPML terms are live.

    From zero the psi slots stay zero and both associations agree bitwise by
    construction -- the mistake `test_sympy_stepper_matches_step_tm` made, and
    the reason that gate proved nothing for a while.
    """
    g = Grid(nx=NX, ny=NY, dx=0.05)
    m = build_tm_coeffs(g, **kw)
    st = init_tm_state(g)
    torch.manual_seed(0)
    st = type(st)._make([(torch.randn_like(v) if torch.is_tensor(v) else v)
                         for v in st])
    return g, m, st


@pytest.mark.parametrize("label,kw", _CASES, ids=[c[0] for c in _CASES])
def test_the_built_curl_is_bitwise_with_stepper_tm(label, kw):
    _, m, st = _seeded(**kw)
    ref = step_db(st, m, "D")
    curl, psis = make_curl(_spec(), "Ez")(st, m, m.dx)

    padded = F.pad(curl, (1, 1, 1, 1))
    Dz = (CN_UPDATE(st.Dz, padded, m.d1, m.d2) if m.d1 is not None
          else st.Dz + m.dt * padded)
    assert torch.equal(Dz, ref.Dz)
    assert torch.equal(psis["psi_Hyx"], ref.psi_Hyx)
    assert torch.equal(psis["psi_Hxy"], ref.psi_Hxy)


def test_the_recipe_lands_on_the_psi_extent():
    """Every term of a curl has the target's psi shape -- that is the trim rule.

    It is what lets ONE rule slice both terms: the differenced axis loses its
    row to the difference, the others are trimmed to the target's own extent.
    A per-term hand slicing is what this replaces.
    """
    terms, want = curl_recipe(_spec(), "Ez")
    assert want == (NX - 2, NY - 2)
    _, m, st = _seeded(npml=6)
    for t in terms:
        f = getattr(st, t.source)
        assert f[t.centre].shape == f[t.neighbour].shape
        assert tuple(t.diff(f[t.centre], f[t.neighbour], m.dx).shape) == want


def test_the_terms_match_what_stepper_tm_pairs():
    """Names too: psi_Hyx takes b_ex, psi_Hxy takes b_ey, as the stepper does.

    The coefficient family follows the TARGET and the letter follows the
    DIFFERENCE AXIS. Getting that pairing backwards is silent -- both arrays
    exist and broadcast.
    """
    terms, _ = curl_recipe(_spec(), "Ez")
    got = {t.psi: (t.coeff, t.sign) for t in terms}
    assert got == {"psi_Hyx": ("ex", +1), "psi_Hxy": ("ey", -1)}


def test_swapping_centre_and_neighbour_is_caught():
    """The order `compiled_difference` wants is not guessable, so assert it.

    `stepper_tm` passes `(Hy[1:], Hy[:-1])` and the naive reading is
    `(lo, hi)`. It is the other way round for a positively-offset source, and
    getting it wrong NEGATES the derivative -- which does not raise, it
    diverges quietly. `stencil_offsets` decides, and this is the check that it
    was consulted.
    """
    _, m, st = _seeded(npml=6)
    terms, _ = curl_recipe(_spec(), "Ez")
    for t in terms:
        f = getattr(st, t.source)
        right = t.diff(f[t.centre], f[t.neighbour], m.dx)
        wrong = t.diff(f[t.neighbour], f[t.centre], m.dx)
        assert torch.allclose(wrong, -right), "swap should negate"
        assert not torch.equal(wrong, right)


@pytest.mark.parametrize("label,kw", _CASES, ids=[c[0] for c in _CASES])
def test_the_built_B_phase_is_bitwise_with_stepper_tm(label, kw):
    """Hx and Hy from curl E -- the other half of step_db, and a DIFFERENT shape.

    A 2D TM block leaves each H component ONE live curl term: `curl_terms("Hx")`
    carries an Ey term differenced along z, and the reduction removes that axis.
    So there is no association to preserve here and no composition to derive --
    which is why the one-term path exists rather than being forced through a
    two-term compose with a zero.
    """
    _, m, st = _seeded(**kw)
    ref = step_db(st, m, "B")
    for tgt, h1n, h2n in (("Hx", "h1x", "h2x"), ("Hy", "h1y", "h2y")):
        rhs, psis = make_curl(_spec(), tgt)(st, m, m.dx)
        h1, h2 = getattr(m, h1n), getattr(m, h2n)
        new = (h1 * getattr(st, tgt) + h2 * rhs if h1 is not None
               else getattr(st, tgt) + m.dt * rhs)
        assert torch.equal(new, getattr(ref, tgt)), tgt
        for k, v in psis.items():
            assert torch.equal(v, getattr(ref, k)), k


def test_a_reduced_H_curl_has_exactly_one_term():
    """And it is the Ez one: the Ey term is differenced along the missing axis.

    Worth asserting rather than assuming, because "two terms" was hard-coded
    here at first and a 2D H phase quietly has one.
    """
    for tgt, src, axis, sign in (("Hx", "Ez", "y", -1), ("Hy", "Ez", "x", +1)):
        terms, want = curl_recipe(_spec(), tgt)
        assert len(terms) == 1
        t = terms[0]
        assert (t.source, t.axis, t.sign) == (src, axis, sign)
    # and in 3D both terms survive, same rule
    assert len(curl_recipe(BlockSpec(("Hx",), (8, 8, 8), (WALL,) * 3), "Hx")[0]) == 2


def test_a_curl_with_no_live_terms_is_refused():
    """A 1D block removes both of Ez's sources; building half a curl would run."""
    with pytest.raises(ValueError, match="no live curl terms"):
        curl_recipe(BlockSpec(("Hx",), (NX,), (WALL,)), "Hx")


# ---------------------------------------------------------------------------
# Folds. A symmetry cut replaces ONE of the two walls, so the trim goes
# asymmetric and the differenced axis gains a row at the cut.
# ---------------------------------------------------------------------------

_FOLDS = [
    ("y_odd",   dict(y_mirror=-1), (WALL, FOLD), (0, 1, 1, 1), False),
    ("y_even",  dict(y_mirror=+1), (WALL, FOLD), (0, 1, 1, 1), False),
    ("x_odd",   dict(x_mirror=-1), (FOLD, WALL), (1, 1, 0, 1), False),
    ("x_even",  dict(x_mirror=+1), (FOLD, WALL), (1, 1, 0, 1), False),
    ("y_rot2",  dict(y_rot2=1),    (WALL, FOLD), (0, 1, 1, 1), True),
    ("rot4",    dict(rot4=1),      (FOLD, FOLD), (0, 1, 0, 1), True),
]


def _ghosts(st, kw, bnd):
    """The cut-plane row each folded axis needs, as stepper_tm builds it.

    Even mirror: H is zero beyond the cut, so a zero row. Odd mirror, C2, C4:
    the plane node is LIVE and carried separately as Hx0/Hy0. Constructing it
    is symmetry algebra and deliberately stays outside the factory -- see
    `make_curl`. This mirrors it only to drive the comparison.
    """
    out = {}
    if bnd[1] is FOLD:
        out["Hx"] = (F.pad(st.Hx, (1, 0)) if kw.get("y_mirror") == 1
                     else torch.cat([st.Hx0[:, None], st.Hx], dim=1))
    if bnd[0] is FOLD:
        out["Hy"] = (F.pad(st.Hy, (0, 0, 1, 0)) if kw.get("x_mirror") == 1
                     else torch.cat([st.Hy0[None, :], st.Hy], dim=0))
    return out


@pytest.mark.parametrize("label,kw,bnd,pad,square", _FOLDS,
                         ids=[c[0] for c in _FOLDS])
def test_the_built_curl_is_bitwise_under_a_fold(label, kw, bnd, pad, square):
    n = (24, 24) if square else (NX, NY)          # rot4 needs a square grid
    g = Grid(nx=n[0], ny=n[1], dx=0.05)
    m = build_tm_coeffs(g, npml=6, **kw)
    st = init_tm_state(g, **kw)
    torch.manual_seed(0)
    st = type(st)._make([(torch.randn_like(v) if torch.is_tensor(v) else v)
                         for v in st])
    ref = step_db(st, m, "D")

    curl, psis = make_curl(_spec(n=n, bnd=bnd), "Ez")(
        st, m, m.dx, fields=_ghosts(st, kw, bnd))
    assert torch.equal(st.Dz + m.dt * F.pad(curl, pad), ref.Dz)
    for k, v in psis.items():
        assert torch.equal(v, getattr(ref, k)), k


def test_a_fold_trims_one_end_and_a_wall_trims_two():
    """The rule a cut changes, stated as slices rather than as a shape.

    Two walls remove a row at each end; a cut replaces one of them and leaves
    that end live. Getting this symmetric would be silent -- the shapes still
    broadcast, the fields just come from the wrong rows.
    """
    walled, _ = curl_recipe(_spec(), "Ez")
    folded, _ = curl_recipe(_spec(bnd=(WALL, FOLD)), "Ez")
    by_src = lambda ts: {t.source: t for t in ts}

    # dHy/dx is trimmed on y, which is the axis whose boundary changed
    assert by_src(walled)["Hy"].centre[1] == slice(1, -1)
    assert by_src(folded)["Hy"].centre[1] == slice(None, -1)
    # dHx/dy is differenced along the folded axis, so it needs the cut row
    assert by_src(walled)["Hx"].ghost == ""
    assert by_src(folded)["Hx"].ghost == "y"
    # and the untouched axis keeps its symmetric trim
    assert by_src(folded)["Hx"].centre[0] == slice(1, -1)


# ---------------------------------------------------------------------------
# Phase composition: a whole TM step, in meep's order.
# ---------------------------------------------------------------------------

_STEP_CASES = [
    ("vacuum_pml", dict(npml=6), 0.7),
    ("sigma_d", dict(npml=6, sigma_d=0.3), 0.7),
    ("sigma_b", dict(npml=6, sigma_b=0.2), 0.7),
    ("both_sigmas", dict(npml=6, sigma_d=0.3, sigma_b=0.2), 0.7),
    ("no_pml", dict(npml=0), 0.7),
    ("no_source", dict(npml=6), 0.0),
]


@pytest.mark.parametrize("label,kw,amp", _STEP_CASES,
                         ids=[c[0] for c in _STEP_CASES])
def test_a_composed_step_is_bitwise_with_step_tm(label, kw, amp):
    """Every field of a whole step, not just the curl.

    This is the claim the unification rests on: `step_tm` written out by hand
    and `make_step` built from a spec produce the SAME BITS, including the
    source injection and the D -> E constitutive division.
    """
    from meep_torch.core.factory import make_step
    from meep_torch.core.stepper_tm import step_tm

    _, m, st = _seeded(**kw)
    mask = torch.zeros(NX, NY, dtype=torch.float64)
    mask[12, 10] = 1.0
    ref = step_tm(st, m, amp, mask)
    got = make_step(_spec())(
        st, m, m.dx, src_amp=amp, src_mask=mask)

    for f, a, b in zip(st._fields, got, ref):
        if torch.is_tensor(a):
            assert torch.equal(a, b), f


def test_the_phase_order_is_load_bearing():
    """H from the OLD E, E from the NEW H. Reversing it still converges.

    That is the danger: a stepper with its phases in the wrong order does not
    blow up, it produces a plausible field that is wrong at O(dt). Advancing E
    from the stale H is the mistake, so drive one step from a state where the
    two differ and check the result is not what the correct order gives.
    """
    from meep_torch.core.factory import make_step
    from meep_torch.core.stepper_tm import step_db, update_eh

    _, m, st = _seeded(npml=6)
    step = make_step(_spec())
    right = step(st, m, m.dx)

    # E advanced from the OLD H: run the D phase before the B phase
    wrong = update_eh(step_db(step_db(st, m, "D"), m, "B"), m, "E")
    assert not torch.equal(right.Ez, wrong.Ez), (
        "swapping the phase order changed nothing -- then this state cannot "
        "tell the orders apart and the test is vacuous")


def test_a_damping_name_is_supplied_or_the_component_is_undamped():
    """Two cases, and the builder must not confuse them.

    The names are a per-stepper convention -- TM reads `d1`/`d2` for Ez but
    `h1x`/`h2x` for Hx, TE reads `d1x`/`d2x` and `hz1`/`hz2`. OMITTING a
    component means undamped on purpose. MISSPELLING one used to mean the same
    thing, because a `getattr(..., None)` fallback swallowed it: the step then
    ran, converged, and was wrong only once a conductivity was on.
    """
    from meep_torch.core.factory import TM_DAMPING, make_step

    assert dict((e[0], e[1:]) for e in TM_DAMPING)["Ez"] == ("d1", "d2")
    _, m, st = _seeded(npml=6, sigma_d=0.3)
    mask = torch.zeros(NX, NY, dtype=torch.float64)

    # a name the coefficients do not carry raises, rather than running undamped
    typo = make_step(_spec(damping=(("Ez", "d1", "d2_"),
                                    ("Hx", "h1x", "h2x"),
                                    ("Hy", "h1y", "h2y"))))
    with pytest.raises(AttributeError):
        typo(st, m, m.dx)

    # and omission really is the undamped branch, visibly different under sigma
    undamped = make_step(_spec(damping=(("Hx", "h1x", "h2x"),
                                        ("Hy", "h1y", "h2y"))))
    assert not torch.equal(undamped(st, m, m.dx).Ez,
                           make_step(_spec())(st, m, m.dx).Ez)


def test_the_allocator_and_the_curl_name_the_same_psi_arrays():
    """Two independent derivations of one name set, compared as SETS.

    `BlockSpec.psi_shapes` gets them from Yee-offset inequality; the built
    curl gets them from `curl_terms`. Comparing shapes on the keys they share
    would pass while one side allocated an array nothing writes -- which is
    exactly the `stepper_3d_lean` over-allocation the parity gate could not
    see, because a psi that is never read is never wrong.
    """
    from meep_torch.core.factory import make_step

    spec = _spec()
    step = make_step(spec)
    built = {t.psi for c in spec.components for t in step.curls[c].terms}
    assert set(spec.psi_shapes()) == built

    _, m, st = _seeded(npml=6)
    for name, shape in spec.psi_shapes().items():
        assert getattr(st, name).shape == torch.Size(shape), name
