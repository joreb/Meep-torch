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
from meep_torch.core.symbolic import WALL

NX, NY = 24, 20

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
    curl, psis = make_curl("Ez", (NX, NY), (WALL, WALL), ndim=2)(st, m, m.dx)

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
    terms, want = curl_recipe("Ez", (NX, NY), (WALL, WALL), ndim=2)
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
    terms, _ = curl_recipe("Ez", (NX, NY), (WALL, WALL), ndim=2)
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
    terms, _ = curl_recipe("Ez", (NX, NY), (WALL, WALL), ndim=2)
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
        rhs, psis = make_curl(tgt, (NX, NY), (WALL, WALL), ndim=2)(st, m, m.dx)
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
        terms, want = curl_recipe(tgt, (NX, NY), (WALL, WALL), ndim=2)
        assert len(terms) == 1
        t = terms[0]
        assert (t.source, t.axis, t.sign) == (src, axis, sign)
    # and in 3D both terms survive, same rule
    assert len(curl_recipe("Hx", (8, 8, 8), (WALL,) * 3, ndim=3)[0]) == 2


def test_a_curl_with_no_live_terms_is_refused():
    """A 1D block removes both of Ez's sources; building half a curl would run."""
    with pytest.raises(ValueError, match="no live curl terms"):
        curl_recipe("Hx", (NX,), (WALL,), ndim=1)
