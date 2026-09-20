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

Bitwise and not merely close: the composition is the steppers' association, and this
is exactly the property that a canonical `Add` sort would break at 1e-14 while
every tolerance-based check still passed.
"""
from typing import NamedTuple

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
    kw.setdefault("entry", "B")                       # TM enters at the B half
    kw.setdefault("sources", ("Ez",))
    kw.setdefault("constitutive", (("Ez", "eps_inf"),))
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
# Phase composition: a whole TM step, in the steppers' phase order.
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
    got = make_step(_spec())(st, m, m.dx, currents={"Ez": (amp, mask)})

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


# ---------------------------------------------------------------------------
# The reduction rule: an axis a block drops, against one it carries as a phase.
# ---------------------------------------------------------------------------

_KZ = ("Ex", "Ez", "Hy")


class _KzShim(NamedTuple):
    """`stepper_kz`'s state under the factory's own names.

    The names are the gap, not the physics: kz calls its CPML memory `psi_dEz`
    and `psi_dHy` while the factory derives `psi_Ezx` and `psi_Hyx` from
    (source, axis). Renaming eleven steppers to suit a builder is the tail
    wagging the dog, so the shim carries the translation and the gate stays
    about the REDUCTION RULE.
    """
    Ex: torch.Tensor
    Ez: torch.Tensor
    Hy: torch.Tensor
    psi_Ezx: torch.Tensor
    psi_Hyx: torch.Tensor


class _KzCoeffShim(NamedTuple):
    dt: float
    kz: float
    b_hx: torch.Tensor
    a_hx: torch.Tensor
    k_hx: torch.Tensor
    b_ex: torch.Tensor
    a_ex: torch.Tensor
    k_ex: torch.Tensor


def _kz_seeded(nx=24, npml=6):
    from meep_torch.core.stepper_kz import build_kz_coeffs, init_kz_state

    c = build_kz_coeffs(nx=nx, dx=0.05, eps=1.0, kz=0.8, npml=npml)
    st = init_kz_state(nx, npml=npml)
    torch.manual_seed(0)
    rand = lambda v: torch.randn_like(v.real) + 1j * torch.randn_like(v.real)
    st = type(st)._make([(rand(v) if torch.is_tensor(v) else v) for v in st])
    bH, aH, kH, bE, aE, kE = c.cpml
    shim = _KzShim(st.Ex, st.Ez, st.Hy, st.psi_dEz, st.psi_dHy)
    cs = _KzCoeffShim(dt=c.dt, kz=c.kz, b_hx=bH, a_hx=aH, k_hx=kH,
                      b_ex=bE, a_ex=aE, k_ex=kE)
    return c, st, shim, cs


def _kz_spec(nx=24):
    return BlockSpec(_KZ, (nx,), (WALL,), phases=(("z", "kz"),))


def test_a_phased_axis_survives_the_reduction_bitwise():
    """`stepper_kz` carries z as `e^{i k_z z}`, so d/dz is `i k_z`, not zero.

    Three components, each a different shape of the same rule: Hy mixes a
    phased term with a PML-stretched difference, Ex is phased ALONE (its other
    driver is differenced along the missing axis and really does vanish), and
    Ez has no phased term at all. If all three are bitwise then the factory
    tells the two kinds of reduction apart.
    """
    from meep_torch.core.factory import make_curl
    from meep_torch.core.stepper_kz import step_kz

    c, st, shim, cs = _kz_seeded()
    ref = step_kz(st, c)
    spec = _kz_spec()

    # Hy: one phased term (-i k_z Ex) and one stretched difference (dEz/dx)
    curl, psis = make_curl(spec, "Hy")(shim, cs, c.dx)
    assert torch.equal(c.h1 * st.Hy + c.h2 * curl, ref.Hy)
    assert torch.equal(psis["psi_Ezx"], ref.psi_dEz)

    # Ex: phased alone, from the NEW Hy, and no CPML memory at all
    mid = shim._replace(Hy=ref.Hy)
    curl, psis = make_curl(spec, "Ex")(mid, cs, c.dx)
    assert psis == {}
    assert torch.equal((c.dx1 * st.Ex * c.eps_x + c.dx2 * curl) / c.eps_x,
                       ref.Ex)

    # Ez: the dHx/dy term is differenced along the DROPPED axis and vanishes
    curl, psis = make_curl(spec, "Ez")(mid, cs, c.dx)
    assert torch.equal((c.de1 * st.Ez * c.eps_z
                        + c.de2 * F.pad(curl, (1, 1))) / c.eps_z, ref.Ez)
    assert torch.equal(psis["psi_Hyx"], ref.psi_dHy)


def test_dropping_and_phasing_are_different_answers():
    """The gap was real: without the rule the phased term is silently gone.

    A block that omits `phases` still builds, still runs and still converges --
    it is solving `k_z = 0` instead. That is the failure the eighth gap named,
    and it has no symptom other than the answer.
    """
    from meep_torch.core.factory import curl_recipe, make_curl

    _, _, shim, cs = _kz_seeded()
    dropped = BlockSpec(_KZ, (24,), (WALL,))

    assert len(curl_recipe(_kz_spec(), "Hy")[0]) == 2
    assert len(curl_recipe(dropped, "Hy")[0]) == 1        # the phase is gone
    assert not torch.equal(make_curl(_kz_spec(), "Hy")(shim, cs, 0.05)[0],
                           make_curl(dropped, "Hy")(shim, cs, 0.05)[0])

    # and Ex loses its ONLY term, which at least raises rather than running
    with pytest.raises(ValueError, match="no live curl terms"):
        curl_recipe(dropped, "Ex")


def test_an_axis_is_differenced_or_phased_but_never_both():
    """A phase on an axis the block CARRIES would be silently ignored.

    `a >= ndim` is false for such an axis, so it is differenced and the phase
    never fires: the spec says one thing, the step does another, and the run
    converges on whichever physics the grid happened to give. That is the same
    lie a misspelled damping name used to tell.
    """
    from meep_torch.core.factory import curl_recipe

    with pytest.raises(ValueError, match="never both"):
        curl_recipe(BlockSpec(_KZ, (24,), (WALL,), phases=(("x", "kz"),)), "Hy")


# ---------------------------------------------------------------------------
# TE through the same factory: the genericity claim, tested rather than argued.
# ---------------------------------------------------------------------------

_TE = ("Hz", "Ex", "Ey")


def _te_seeded(**kw):
    from meep_torch.core.coeffs import build_te_coeffs
    from meep_torch.core.stepper_te import init_te_state

    g = Grid(nx=NX, ny=NY, dx=0.05)
    m = build_te_coeffs(g, **kw)
    st = init_te_state(g)
    torch.manual_seed(0)
    st = type(st)._make([(torch.randn_like(v) if torch.is_tensor(v) else v)
                         for v in st])
    return g, m, st


def _te_spec(**kw):
    kw.setdefault("damping", (("Ex", "d1x", "d2x"), ("Ey", "d1y", "d2y"),
                              ("Hz", "hz1", "hz2")))
    kw.setdefault("entry", "D")                  # TE enters at the D half
    kw.setdefault("sources", ("Ey", "Hz"))       # Dy alone, plus magnetic Hz
    kw.setdefault("constitutive", (("Ex", "eps_x"), ("Ey", "eps_y")))
    return BlockSpec(_TE, (NX, NY), (WALL, WALL), **kw)


@pytest.mark.parametrize("label,kw", _CASES, ids=[c[0] for c in _CASES])
def test_the_built_curl_is_bitwise_with_stepper_te(label, kw):
    """A different polarisation, a different lattice, the same factory.

    TM and TE are the two blocks of one 6x6 system, but they are NOT the same
    discrete problem: TE's Hz sits on the dual lattice at `(nx-1, ny-1)`, its
    psi arrays are named for different sources, and its coefficients are
    `b_ey`/`b_ex` where TM reads `b_ex`/`b_ey`. Nothing in `curl_recipe` was
    changed for any of that -- the names come out of `_family(target) + axis`
    and the shapes out of `psi_extent`, which is the claim.
    """
    from meep_torch.core.stepper_te import step_db

    _, m, st = _te_seeded(**kw)
    spec = _te_spec()
    refD, refB = step_db(st, m, "D"), step_db(st, m, "B")

    for tgt, ref, base, pad, fld, damp in (
            ("Ex", refD, st.Dx, (1, 1), "Dx", ("d1x", "d2x")),
            ("Ey", refD, st.Dy, (0, 0, 1, 1), "Dy", ("d1y", "d2y")),
            ("Hz", refB, st.Hz, (0, 0, 0, 0), "Hz", ("hz1", "hz2"))):
        curl, psis = make_curl(spec, tgt)(st, m, m.dx)
        padded = F.pad(curl, pad)
        c1, c2 = (getattr(m, n) for n in damp)
        got = CN_UPDATE(base, padded, c1, c2) if c1 is not None \
            else base + m.dt * padded
        assert torch.equal(got, getattr(ref, fld)), tgt
        for k, v in psis.items():
            assert torch.equal(v, getattr(ref, k)), (tgt, k)


def test_the_curl_is_composed_in_the_curls_own_sign_order():
    """Which term is stretched first comes from the CURL, not the rhs.

    `curl_terms` folds the minus of `dB/dt = -curl E` into its signs; meep does
    not -- it composes `curl_z = dEy/dx - dEx/dy` and applies the minus at the
    field update. Same four numbers, different grouping, and grouping is what
    bitwise parity is. Ordering an H target by the rhs sign was measured one
    ulp out (8.882e-16 on TE's Hz) with both psi arrays already exact, which is
    the failure this ordering exists to prevent.
    """
    from meep_torch.core.symbolic import curl_signs, curl_terms, family_sign

    spec = _te_spec()
    terms, _ = curl_recipe(spec, "Hz")
    by_src = {t.source: t for t in terms}

    # the rhs and the curl disagree on BOTH terms, by exactly the family sign
    assert family_sign("Hz") == -1 and family_sign("Ez") == +1
    for t in terms:
        assert t.curl_sign == -t.sign, t.source
    # and the stretched-first term is dEy/dx, which is what stepper_te composes
    assert by_src["Ey"].curl_sign > 0 and by_src["Ex"].curl_sign < 0

    # an E target has nothing to distinguish: rhs and curl agree
    assert curl_signs("Ez") == curl_terms("Ez")


def test_te_and_tm_share_every_derived_name():
    """The coefficient and psi names are derived, and TE is the proof.

    TM reads `b_ex` for the Ez update differenced along x; TE reads `b_ex` for
    the Ey update differenced along x. Same rule -- family of the TARGET, letter
    of the DIFFERENCE AXIS -- and it was written against TM alone.
    """
    te = {t.source: t for t in curl_recipe(_te_spec(), "Hz")[0]}
    assert (te["Ex"].coeff, te["Ex"].psi) == ("hy", "psi_Exy")
    assert (te["Ey"].coeff, te["Ey"].psi) == ("hx", "psi_Eyx")

    ex, = curl_recipe(_te_spec(), "Ex")[0]
    ey, = curl_recipe(_te_spec(), "Ey")[0]
    assert (ex.coeff, ex.psi) == ("ey", "psi_Hzy")
    assert (ey.coeff, ey.psi) == ("ex", "psi_Hzx")


_TE_STEP = [("vacuum_pml", dict(npml=6), 0.7, 0.0),
            ("sigma_d", dict(npml=6, sigma_d=0.3), 0.7, 0.0),
            ("sigma_b", dict(npml=6, sigma_b=0.2), 0.7, 0.0),
            ("both_sigmas", dict(npml=6, sigma_d=0.3, sigma_b=0.2), 0.7, 0.0),
            ("no_pml", dict(npml=0), 0.7, 0.0),
            ("magnetic_src", dict(npml=6), 0.0, 0.45),
            ("both_srcs", dict(npml=6), 0.7, 0.45),
            ("no_source", dict(npml=6), 0.0, 0.0)]


@pytest.mark.parametrize("label,kw,amp,hamp", _TE_STEP,
                         ids=[c[0] for c in _TE_STEP])
def test_a_composed_te_step_is_bitwise_with_step_te(label, kw, amp, hamp):
    """The whole TE step, in TE's own phase order.

    Three gaps close here at once and each was a TM constant until now: TE
    enters meep's cycle at the D HALF (TM enters at B), it injects the electric
    current into `Dy` ALONE, and it puts a MAGNETIC current into Hz -- which
    `make_step` had no concept of. Plus `eps_x`/`eps_y` per component where TM
    has one `eps_inf`.
    """
    from meep_torch.core.factory import make_step
    from meep_torch.core.stepper_te import step_te

    _, m, st = _te_seeded(**kw)
    ey_mask = torch.zeros(NX, NY - 1, dtype=torch.float64)
    ey_mask[12, 10] = 1.0
    hz_mask = torch.zeros(NX - 1, NY - 1, dtype=torch.float64)
    hz_mask[11, 9] = 1.0

    ref = step_te(st, m, amp, ey_mask if amp else None,
                  src_h_amp=hamp, src_hz_mask=hz_mask if hamp else None)
    got = make_step(_te_spec())(st, m, m.dx, currents={
        "Ey": (amp, ey_mask), "Hz": (hamp, hz_mask)})

    for f, a, b in zip(st._fields, got, ref):
        if torch.is_tensor(a):
            assert torch.equal(a, b), f


def test_the_entry_half_is_a_spec_field_and_it_changes_the_answer():
    """TM enters at B, TE at D. A rotation of one cycle, and not derivable.

    It decides which fields are old at each phase, so it decides the bits. The
    check that it is load-bearing: run TE's own spec from the wrong half and
    the result must differ -- otherwise this state cannot tell the two apart
    and the gate is vacuous.
    """
    from meep_torch.core.factory import make_step

    _, m, st = _te_seeded(npml=6)
    right = make_step(_te_spec())(st, m, m.dx)
    wrong = make_step(_te_spec(entry="B"))(st, m, m.dx)
    assert not torch.equal(right.Hz, wrong.Hz)
    with pytest.raises(ValueError, match="entry must be"):
        make_step(_te_spec(entry="E"))


def test_a_source_only_drives_the_components_the_spec_declares():
    """TE injects into Dy ALONE; driving every E component is a TM assumption.

    `stepper_te` has one electric source target and it is Ey. A builder that
    fanned the same mask across Ex and Ey would be wrong for TE and invisible
    to every TM gate, since TM has exactly one E component.
    """
    from meep_torch.core.factory import make_step

    _, m, st = _te_seeded(npml=6)
    mask_x = torch.ones(NX - 1, NY, dtype=torch.float64)
    step = make_step(_te_spec())

    # Ex is not a declared source, so a current aimed at it does nothing
    assert torch.equal(step(st, m, m.dx, currents={"Ex": (1.0, mask_x)}).Ex,
                       step(st, m, m.dx).Ex)
    # and a component this block does not carry raises rather than passing
    with pytest.raises(ValueError, match="not in this block"):
        step(st, m, m.dx, currents={"Ez": (1.0, mask_x)})
