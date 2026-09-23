# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Build a curl from a spec, once, and execute it many times.

Twelve steppers hand-write the same assembly: difference each source along its
axis, trim the other axes to the target's interior, run the CPML recursion,
compose in the steppers' order, and hand the result to the field update. None
is a choice --

* WHICH terms, from :func:`~meep_torch.core.symbolic.curl_terms`: the Jacobian
  of Maxwell's right-hand side, giving ``(source, axis, sign)``.
* WHICH neighbour is the ``+1`` sample, from
  :func:`~meep_torch.core.symbolic.compiled_difference`, fixed by the Yee
  offsets of source and target.
* WHAT SHAPE the result is, from
  :func:`~meep_torch.core.symbolic.psi_extent`: every term of a curl lands on
  the target's psi extent, which is why one trim rule covers them all.
* WHICH ORDER they combine in, from
  :func:`~meep_torch.core.symbolic.compiled_curl_composition`, because bitwise
  parity with the hand-written steppers is an association property. (Parity
  with MEEP is a tolerance property -- see symbolic's module docstring.)

THE MEASURED CONSTRAINT, and the reason this is a factory rather than a
generic function: resolve the spec at BUILD time. A version that re-read the
Yee table, compared sympy Rationals and looked coefficients up by string on
every step measured 1.5-2x slower, and that 1.5-2x was mistaken for the cost of
genericity by two sessions. The heavy work -- which terms, which neighbour,
what shape, what order -- is hoisted into a closure over plain ints, slice
objects and bound functions. No codegen; a closure.

WHAT IS NOT HOISTED, measured 2026-09-20 on this code rather than inherited
from the earlier generic stepper: `_one` still builds ``f"b_{t.coeff}"`` and
friends and calls ``getattr`` per term per step. That costs 222 ns per term
against 37 ns for a build-time ``attrgetter``, so 0.74 us per 4-term step --
**0.45% of a 100x100 TM step and 0.029% of a 400x400 one**. An A/B against
`step_tm` cannot resolve it: the difference sits about two orders of magnitude
below the thread-scheduling noise at these sizes, and a run that tried it
produced ratios from 0.69x to 1.04x, i.e. noise with the factory "winning"
twice, which is impossible for strictly more Python work.

So it is left alone deliberately. Precomputing the names onto `CurlTerm` and
binding `attrgetter` at build time is about five lines and would keep the
argument-not-capture property, but 0.03% is not a reason to touch a hot path
that is bitwise-gated against two steppers. Revisit if a profile ever shows
per-step Python on the critical path -- many small blocks, or an eager 3D run
with tiny tiles, are where it would first appear.

``coeffs`` is an ARGUMENT to the returned function, not a capture, so the same
built step serves every rebuild of the coefficients an adjoint produces and
stays a pure function of two pytrees.

SCOPE. Cartesian only, and no symmetry folds yet. `_reject_noncartesian`
guards the cylindrical frame upstream (the bare ``E_phi / r`` term has no
difference to attach to), and a fold changes which rows are live rather than
which terms exist -- see ``cut_plane_row`` and ``tasks/fdtd_sympy-stepper-kernels``.

STATUS, 2026-09-23: NOTHING IN THE ENGINE IMPORTS THIS YET. The only importer
in the tree is ``tests/analytic/test_factory.py``, which builds steppers here
and compares them BITWISE against the hand-written ``step_tm`` / ``step_te`` /
``step_kz`` / ``step_1d`` / ``step_3d``. So today this module is a
verification oracle: it proves the twelve hand-written steppers are what the
spec says they are. Every PHYSICS gate -- meep parity, the analytic cavity and
Fresnel solutions -- is still held by those hand-written steppers, and the
factory's gates only compare it against them.

Stated here because an audit had to grep for importers to find it out, and
because the distinction matters when reading a failure: a red in
``test_factory`` means the derived form and the hand form disagree, NOT that
the engine is wrong. The hand form is what ran.

THE INTENT IS TO REPLACE THE HAND-WRITTEN STEPPERS WITH THIS, not to retire
it. The module stays in ``core/`` for that reason rather than moving to
``symbolic_test/`` -- it is unlanded engine, not a test fixture that wandered
in. What landing needs is the five open tasks:
``fdtd_factory-3d-step-composition``, ``fdtd_factory-bloch-phase-on-the-wrap``,
``fdtd_factory-kz-through-the-phased-axis``,
``fdtd_factory-cylindrical-yee-table`` and
``fdtd_factory-poles-and-kerr-in-the-phase-layer``. Until they land, a
hand-written stepper is NOT dead weight duplicating this one: it is the thing
under test.
"""
from typing import NamedTuple

import torch

from .kernels import EH_FROM_D, PSI_NEW, STRETCHED
from .symbolic import (AXES, FOLD, PERIODIC, BlockSpec, compiled_curl_composition,
                       compiled_difference, compiled_phase_factor, curl_signs,
                       curl_terms, family_sign, field_shape, psi_extent,
                       stencil_offsets, yee_table)

#: ``f, k -> i k f``, derived once. See :func:`compiled_phase_factor`.
PHASE = compiled_phase_factor()

__all__ = ["CurlTerm", "curl_recipe", "make_curl", "make_step", "TM_DAMPING"]


class CurlTerm(NamedTuple):
    """One difference of a curl, fully resolved at build time."""
    source: str                  # component being differenced
    axis: str                    # axis it is differenced along
    sign: int                    # +1 or -1: the RHS sign, `family_sign(target)
                                 # * curl_sign` -- DERIVED, not an independent
                                 # source of truth (the identity is asserted in
                                 # curl_recipe). It is what the three branches
                                 # with nothing to associate read directly.
    diff: object                 # compiled_difference(source, axis, target)
    centre: tuple                # slice for the field's OWN node (k = 0)
    neighbour: tuple             # slice for the other sample
    psi: str                     # state field holding this term's CPML memory
    coeff: str                   # coefficient suffix, e.g. "ex" -> m.b_ex
    ghost: str                   # axis needing a cut-plane row before the
                                 # difference, or "" -- see make_curl
    curl_sign: int = 0           # the INTRINSIC curl's sign for this term,
                                 # from curl_signs. `sign` above is this times
                                 # `family_sign(target)`, so on ONE term the
                                 # two carry the same information and `sign` is
                                 # the cheaper read. They part company only
                                 # where two stretched terms are COMPOSED:
                                 # there the order of the pair is the grouping,
                                 # and the grouping has to follow the intrinsic
                                 # curl with the family flip applied once at
                                 # the end -- meep's single family-wide stride
                                 # flip (`step_db.cpp`: `if (ft == D_stuff) {
                                 # stride_p = -stride_p; stride_m = -stride_m;
                                 # }`). Ordering that branch by `sign` instead
                                 # fails 12 of 48 tests; see make_curl.
    wrap: int = 0                # roll shift for a PERIODIC differenced axis,
                                 # or 0 when the axis is sliced. A wrap has no
                                 # fencepost: the neighbour of the last row is
                                 # the first, so `centre` and `neighbour` both
                                 # keep the full extent on that axis and the
                                 # roll supplies the offset the slice pair
                                 # would have. Sign follows `centre_is_upper`
                                 # exactly as the slices do -- see curl_recipe.
    phase: str = ""              # coefficient holding this axis's wavenumber
                                 # when the axis is PHASED rather than
                                 # differenced; "" for an ordinary term. A
                                 # phased term has no diff, no psi and no
                                 # CPML -- see BlockSpec.phases.
    real_phase: int = 0          # 0: the phase is `i k f` (complex block).
                                 # +1 / -1: the block stores one polarisation
                                 # as `i * F` (BlockSpec.imaginary), so the
                                 # phase is the REAL `+k f` / `-k f` --
                                 # `i^(a-b+1)` with a/b = target/source
                                 # imaginary-stored. Separate from `sign`
                                 # on purpose: `sign` keeps meaning
                                 # `family_sign * curl_sign` on every branch
                                 # of make_curl, and this is the extra factor
                                 # the representation adds on top.


def _family(component: str) -> str:
    """``e`` for an E-target, ``h`` for an H-target -- the coefficient family.

    `stepper_tm` reads `m.b_ex`/`m.b_ey` in the Ez update and `m.b_hx`/`m.b_hy`
    in the H update, so the family follows the TARGET and the letter follows
    the DIFFERENCE AXIS. Deriving it beats a lookup table that has to be
    rewritten per stepper.
    """
    return component[0].lower()


def _phased_term(spec, target, source, axis, curl_sign, fam, coeff, want, ndim):
    """A curl term on an axis the block carries as a phase rather than a grid.

    There is no difference, so there is no neighbour, no CPML memory and no
    stretch -- the whole term is ``sign * i k * source`` at the source's own
    nodes. What DOES survive is the trim: the value still has to land on the
    target's psi extent, and a phased source that does not already sit there is
    a staggering error rather than something to pad around.

    ``curl_sign`` comes from the same ``curl_of`` lookup the differenced
    branch uses, not from ``curl_terms``' rhs sign directly -- the caller used
    to pass the rhs sign in as ``curl_sign`` too, which is only correct when
    ``family_sign(target) == +1`` and was never caught because nothing reads a
    phased term's ``curl_sign``.

    THE REPRESENTATION lands here and nowhere else. With the target stored as
    ``i^a F`` and the source as ``i^b G`` (``BlockSpec.imaginary``), the
    coupling ``sign * i k * G`` becomes ``sign * k * i^(a-b+1) * G'``: real
    ``+k`` when the SOURCE is the imaginary-stored one, real ``-k`` when the
    TARGET is, and still complex when both or neither are -- which is a block
    that asked for real storage and cannot have it, so it raises rather than
    quietly building a complex step under a real-sounding spec.
    """
    real_phase = 0
    if spec.imaginary:
        a, b = target in spec.imaginary, source in spec.imaginary
        if a == b:
            raise ValueError(
                f"i*{coeff}*{source} at {target}: spec.imaginary stores "
                f"{'both' if a else 'neither'} side as i*F, so the phase stays "
                "complex; exactly one polarisation must be imaginary-stored")
        real_phase = 1 if b else -1
    src_shape = field_shape(source, spec.extents, spec.boundaries,
                            origin=spec.origin)
    sl = [slice(None)] * ndim
    for b in range(ndim):
        trim = src_shape[b] - want[b]
        if not trim:
            continue
        if spec.boundaries[b] is FOLD:
            sl[b] = slice(None, -trim)
        elif trim % 2:
            raise ValueError(
                f"i*{coeff}*{source} at {target}: axis {AXES[b]} needs an odd "
                f"trim of {trim} without a fold")
        else:
            sl[b] = slice(trim // 2, -(trim // 2))
    return CurlTerm(source=source, axis=axis, sign=fam * curl_sign, diff=None,
                    curl_sign=curl_sign, centre=tuple(sl), neighbour=tuple(sl),
                    psi="", coeff="", ghost="", phase=coeff,
                    real_phase=real_phase)


def curl_recipe(spec: BlockSpec, target: str) -> tuple:
    """Resolve every term of ``target``'s curl against ``spec``'s grid.

    Returned terms carry slices, not indices, so the step does no arithmetic on
    shapes. The trim is the part worth stating: a term is differenced along its
    own axis and trimmed on the OTHERS, because the result has to land on the
    target's psi extent -- which is what makes one rule cover both terms of a
    curl instead of two hand-written slicings.

    The spec comes FIRST for a reason that is not style: every call site of the
    old signature passed ``ndim == len(extents)``, so the argument was
    redundant, and a transposed ``(target, spec)`` now dies on the spot instead
    of silently indexing a string. Three separate argument-order mistakes in
    this file were caught by a bitwise gate rather than by a type.
    """
    extents, boundaries = spec.extents, spec.boundaries
    ndim = len(extents)
    phases = dict(spec.phases)
    curl_of = {(src, ax): g for src, ax, g in curl_signs(target)}
    # The one family-wide flip, computed once per curl rather than per term --
    # see CurlTerm.sign. `curl_terms(target)`'s own rhs sign is `fam *
    # curl_of[(source, axis)]` by construction (`family_sign`'s own
    # derivation asserts this constant across every term), so it is used below
    # only to enumerate which (source, axis) pairs exist, never for the value
    # stored on the term.
    fam = family_sign(target)
    unknown = [c for c in spec.imaginary if c not in spec.components]
    if unknown:
        raise ValueError(
            f"spec.imaginary names {unknown}, which this block does not carry")
    on_grid = [a for a in phases if AXES.index(a) < ndim]
    if on_grid:
        # A phase only applies to an axis the block does NOT carry. Naming an
        # axis that is on the grid would be silently ignored -- it gets
        # differenced, the phase never fires, and the run converges on the
        # wrong physics. Same shape of lie as a misspelled damping name.
        raise ValueError(
            f"{target}: phases name {on_grid}, which this block carries as "
            f"grid axes ({AXES[:ndim]}); an axis is differenced or phased, "
            "never both")
    want = psi_extent(target, extents, boundaries, origin=spec.origin)
    out = []
    for source, axis, _rhs_sign in curl_terms(target):
        a = AXES.index(axis)
        csign = curl_of[(source, axis)]
        if a >= ndim:
            # The axis is not on this block. WHICH WAY it left decides whether
            # the term goes with it: a 2D reduction assumes nothing varies
            # along it, so the derivative is zero; a phased reduction assumes
            # `e^{i k q}`, so the derivative survives as `i k f` with no
            # difference to attach to. Silently dropping the second kind is the
            # eighth BlockSpec gap.
            if axis not in phases:
                continue
            out.append(_phased_term(spec, target, source, axis, csign, fam,
                                    phases[axis], want, ndim))
            continue
        src_shape = field_shape(source, extents, boundaries,
                                origin=spec.origin)
        lo, hi = [slice(None)] * ndim, [slice(None)] * ndim
        for b in range(ndim):
            if b == a:
                if boundaries[b] is PERIODIC:
                    # the wrap supplies the row the difference would have eaten,
                    # so nothing is sliced away and nothing is trimmed: the
                    # offset moves into `wrap` below.
                    lo[b] = hi[b] = slice(None)
                    have = src_shape[b]
                else:
                    # the difference itself removes the row, and
                    # `stencil_offsets` has already fixed which one
                    lo[b], hi[b] = slice(None, -1), slice(1, None)
                    # a fold hands the difference an extra row at the cut, so
                    # the source is one longer there than its own allocation
                    have = src_shape[b] - 1 + (1 if boundaries[b] is FOLD else 0)
            else:
                have = src_shape[b]
            trim = have - want[b]
            if b == a:
                if trim:
                    raise ValueError(
                        f"d{source}/d{axis} at {target}: differenced axis is "
                        f"{have}, psi extent wants {want[b]}")
                continue
            if trim:
                if boundaries[b] is FOLD:
                    # A cut replaces ONE of the two walls, so the trim is
                    # asymmetric: the cut end stays live and the far wall is
                    # trimmed. Every fold in this package sits at the LOW end
                    # (`y_mirror_lo`, `y_rot2_lo`, `rot4_lo`), which is what
                    # lets the far end be the high one.
                    lo[b] = hi[b] = slice(None, -trim)
                else:
                    # symmetric: both ends are the same kind of boundary
                    if trim % 2:
                        raise ValueError(
                            f"d{source}/d{axis} at {target}: axis {AXES[b]} "
                            f"needs an odd trim of {trim} without a fold")
                    k = trim // 2
                    lo[b] = hi[b] = slice(k, -k)
        # WHICH slice is the field's own node is not a choice either.
        # `compiled_difference` takes (centre, neighbour) where centre is the
        # sample at relative index 0, and `stencil_offsets` says where that
        # sits: for delta = +1/2 the samples are k = -1, 0 so k = 0 is the
        # HIGHER index; for delta = -1/2 they are k = 0, +1 and it is the
        # lower. Passing these the wrong way round negates the derivative, and
        # a negated curl still runs -- it diverges quietly rather than raising.
        table = yee_table("cartesian")
        delta = table[source][a] - table[target][a]
        ks = stencil_offsets(delta)
        centre_is_upper = max(ks) == 0
        c_sl, n_sl = ((hi, lo) if centre_is_upper else (lo, hi))
        # On a wrapped axis the two slices are the same full extent, so the
        # offset the pair would have encoded has to be carried separately:
        # centre-upper means the neighbour sits one row BELOW, which is
        # `roll(f, +1)`; centre-lower means one above, `roll(f, -1)`.
        wrap = 0
        if boundaries[a] is PERIODIC:
            assert c_sl[a] == n_sl[a] == slice(None), (target, source, axis)
            wrap = 1 if centre_is_upper else -1
        # the identity the one-term branches rely on, checked where it is
        # derived rather than restated per branch: verified across all 12 terms
        # of all 6 components, fam = +1 on E and -1 on H, no exceptions.
        assert _rhs_sign == fam * csign, (target, source, axis, _rhs_sign, fam, csign)
        out.append(CurlTerm(
            source=source, axis=axis, sign=fam * csign,
            curl_sign=csign,
            diff=compiled_difference(source, axis, target),
            centre=tuple(c_sl), neighbour=tuple(n_sl),
            ghost=(axis if boundaries[a] is FOLD else ""),
            wrap=wrap,
            # A wrapped axis carries no absorber -- a periodic boundary and
            # a PML are contradictory conditions on the same axis -- so there
            # is no CPML memory to name and no coefficient family to read.
            # The psi-free path is DERIVED from the boundary, not declared:
            # `stepper_bloch2` (all axes periodic) has no psi at all and
            # `stepper_fv2d` (x walled, y wrapped) has psi on x only, which is
            # this rule read off two independent steppers.
            psi=("" if wrap else f"psi_{source}{axis}"),
            coeff=("" if wrap else f"{_family(target)}{axis}"),
        ))
    if not out:
        raise ValueError(
            f"{target} has no live curl terms on a {ndim}D block: every source "
            "is differenced along an axis the reduction removed")
    if len(out) > 2:
        raise ValueError(f"{target} has {len(out)} curl terms, expected 1 or 2")
    if len([t for t in out if t.phase]) > 1:
        # Two phased terms would need a composition that does not exist here --
        # `make_curl`'s mixed branch unpacks exactly one of each. No cartesian
        # block reaches it: a curl component has two terms on two DIFFERENT
        # axes, so both being phased means a block with no grid axes left.
        raise ValueError(
            f"{target} has two phased terms, which would leave the block with "
            "no differenced axis at all")
    return tuple(out), want


def make_curl(spec: BlockSpec, target: str):
    """``f(state, coeffs, dx) -> (curl, {psi_name: new_psi})``.

    The spec work happens HERE, once. The returned closure touches tensors and
    slice objects only.

    THE FAMILY SIGN MATTERS AS AN ASSOCIATION ONLY WHERE TWO STRETCHED TERMS
    ARE COMPOSED. That is the two-term branch and nowhere else: there the order
    decides the grouping, and grouping is what bitwise parity is. Measured --
    ordering that branch by the rhs sign instead of the intrinsic curl sign
    fails 12 of 48 tests.

    Every other branch uses `t.sign` directly, which IS `family_sign(target) *
    curl_sign` by construction (asserted in `curl_recipe`). One term has no
    association to preserve, so splitting the flip into a curl-sign branch plus
    a family flip would express one negation as two -- the same bits, an extra
    full-array op, and a second idiom for one rule.

    An earlier version did exactly that in all four branches and needed a
    comment per branch to say which idiom applied and why the other would
    double-negate. Reverted on review: eleven honest fields and one rule stated
    once beats five fields and a legend.
    """
    terms, want = curl_recipe(spec, target)
    fam = family_sign(target)

    def _src(state, t, fields):
        return (fields[t.source] if fields and t.source in fields
                else getattr(state, t.source))

    def _neighbour(f, t):
        """The other sample, wrapped if the differenced axis is periodic."""
        if t.wrap:
            f = torch.roll(f, t.wrap, dims=AXES.index(t.axis))
        return f[t.neighbour]

    def _one(state, coeffs, dx, t, psis, fields):
        f = _src(state, t, fields)
        d = t.diff(f[t.centre], _neighbour(f, t), dx)
        if not t.psi:
            return d, None, None
        p = PSI_NEW(d, getattr(state, t.psi),
                    getattr(coeffs, f"b_{t.coeff}"),
                    getattr(coeffs, f"a_{t.coeff}"))
        psis[t.psi] = p
        return d, p, getattr(coeffs, f"k_{t.coeff}")

    def _stretch(t):
        """The term's contribution to the curl, resolved at BUILD time.

        An unstretched term is the difference itself, not `d/1 + 0`: the
        second is two extra full-array ops per step and is not even the same
        bits at a signed zero.
        """
        return (lambda d, p, k: STRETCHED(d, p, k)) if t.psi else (
            lambda d, p, k: d)

    def _phase(state, coeffs, t, fields):
        f, k = _src(state, t, fields)[t.centre], getattr(coeffs, t.phase)
        if not t.real_phase:
            return PHASE(f, k)
        # The stored-i representation: `k f` in place of `i k f`, the sign
        # from which side carries the i. Bitwise with the complex block's
        # real/imaginary part on the mapped state, since `(a + i b) * i k` is
        # an exact component swap with one negation and no rounding.
        v = k * f
        return v if t.real_phase > 0 else -v

    phased = [t for t in terms if t.phase]
    if phased and len(terms) == 1:
        # The whole curl is a phase: `stepper_kz`'s Ex sees only -i k_z Hy,
        # since its other driver is differenced along the axis the block does
        # not carry. No difference, no psi, nothing to stretch.
        (t,) = terms

        def curl(state, coeffs, dx, fields=None):
            v = _phase(state, coeffs, t, fields)
            return (v if t.sign > 0 else -v), {}
    elif phased:
        # One phased, one differenced -- `stepper_kz`'s Hy. Only the
        # x-derivative is PML-stretched (z is the propagation axis and k_z is
        # fixed), so the phased term joins the stretched one as a plain add.
        # The two terms commute exactly in IEEE, so the hand-written
        # `-i k_z Ex + ce_x` and this `ce_x -+ i k_z Ex` are the same bits;
        # association, not order, is what parity depends on.
        (t_p,), (t_d,) = phased, [t for t in terms if not t.phase]
        stretch_d = _stretch(t_d)

        def curl(state, coeffs, dx, fields=None):
            psis = {}
            out = stretch_d(*_one(state, coeffs, dx, t_d, psis, fields))
            if t_d.sign < 0:
                out = -out
            v = _phase(state, coeffs, t_p, fields)
            return (out + v if t_p.sign > 0 else out - v), psis
    elif len(terms) == 1:
        # A reduced block's H phase: one term, so there is no association to
        # preserve and no composition to derive. `dF/dt = sign * stretched`,
        # and a caller writing `F - dt*stretched` gets the same floats as
        # `F + dt*(-stretched)` -- IEEE negation is exact.
        (t,) = terms
        stretch = _stretch(t)

        def curl(state, coeffs, dx, fields=None):
            psis = {}
            out = stretch(*_one(state, coeffs, dx, t, psis, fields))
            return (out if t.sign > 0 else -out), psis
    else:
        # Each term's stretch flag goes INTO the derivation, so a bare term
        # contributes `-d` and never appears as `-d/1 - 0`.
        # Ordered by the INTRINSIC CURL's sign, not the rhs's. meep composes
        # the curl and applies the family minus at the field update -- for an H
        # target those differ, and the difference is an association rather than
        # an algebra: `stepper_te` builds `(dEy/dx/k + psi) - dEx/dy/k - psi`
        # and negates it, while ordering by the rhs sign builds the same four
        # numbers grouped the other way. Measured at 8.882e-16 on Hz before this
        # was derived, with both psi arrays already bitwise.
        (t_pos,), (t_neg,) = ([t for t in terms if t.curl_sign > 0],
                              [t for t in terms if t.curl_sign < 0])
        compose = compiled_curl_composition(
            n_terms=2, stretched=(bool(t_pos.psi), bool(t_neg.psi)))
        stretch_p = _stretch(t_pos)

        def curl(state, coeffs, dx, fields=None):
            psis = {}
            first = stretch_p(*_one(state, coeffs, dx, t_pos, psis, fields))
            dn, pn, kn = _one(state, coeffs, dx, t_neg, psis, fields)
            out = (compose(first, dn, kn, pn) if t_neg.psi
                   else compose(first, dn))
            # the family minus, applied AFTER the composition exactly where
            # meep applies it. IEEE negation is exact, so `c1*F + c2*(-curl)`
            # and `c1*F - c2*curl` are the same bits.
            return (out if fam > 0 else -out), psis

    curl.terms = terms
    curl.shape = want
    return curl


#: The TM block's damping pair per component, as `BlockSpec.damping` wants it.
#: NOT derivable and not a default: it is `stepper_tm`'s own convention, which
#: `stepper_te` does not share (`d1x`/`d2x`, `hz1`/`hz2` -- note `hz1`, not
#: `h1z`). Recorded as the seventh BlockSpec gap in
#: tasks/fdtd_sympy-stepper-kernels, and closed by carrying it as data.
TM_DAMPING = (("Ez", "d1", "d2"), ("Hx", "h1x", "h2x"), ("Hy", "h1y", "h2y"))


def _pad_spec(spec, target):
    """Zero-pad that puts a curl back on its target's own grid.

    Derived, not typed: the curl lands on `psi_extent` and the field lives on
    `field_shape`, so the difference is what has to be padded -- and a FOLD
    takes its row at the far end only, because the cut end stayed live.
    Returned in torch's reversed pairwise order.
    """
    full = field_shape(target, spec.extents, spec.boundaries,
                       origin=spec.origin)
    want = psi_extent(target, spec.extents, spec.boundaries,
                      origin=spec.origin)
    pads = []
    for b in range(len(spec.extents)):
        total = full[b] - want[b]
        lo = 0 if spec.boundaries[b] is FOLD else total // 2
        pads.append((lo, total - lo))
    out = []
    for b in reversed(range(len(spec.extents))):
        out.extend(pads[b])
    return tuple(out)


def make_step(spec: BlockSpec):
    """Compose meep's phase order around :func:`make_curl`.

    ``step(state, coeffs, dx, currents=None, fields=None)``, where ``currents``
    maps a component name to ``(amplitude, mask)``.

    THE CYCLE IS TWO HALVES and the order is not an implementation detail. The
    B half advances H from the current E; the D half advances D from the
    current H and then solves E. H from the OLD E and E from the NEW H is the
    leapfrog, and running the phases in any other order changes the answer
    while still converging, which is the worst kind of wrong.

    WHICH HALF STARTS is the ninth BlockSpec gap and is not derivable.
    ``stepper_tm`` enters at B; ``stepper_te`` enters at D, and says why: the
    TM-TE duality (``Hz <-> Ez``, ``(Ex, Ey) <-> (-Hx, -Hy)``) staggers Hz on
    integer steps, so entering at D is what keeps TE's discrete Hz update graph
    isomorphic to TM's Ez graph and every TE monitor convention intact. A
    rotation of the same cycle, and it changes which fields are old at each
    phase -- so it changes the bits.

    Everything else conventional comes off the spec too, each one a gap that
    was a TM constant hard-typed here until a second polarisation arrived:
    ``damping_names`` for the CN pair, ``pole_name`` for the polarisation sum,
    ``eps_name`` for the permittivity at each E node, and ``driven`` for which
    components a current may be injected into -- TE drives ``Dy`` ALONE and
    puts a magnetic current into ``Hz``.

    Scope: no ADE poles and no Kerr. The polarisation sum is read, so a block
    with ``n_poles = 0`` is exact, and one with poles needs ``update_pols``
    between the source and the E solve -- `kernels.ADE_POLE`, composition
    rather than derivation.
    """
    components = spec.components
    h_comps = [c for c in components if c[0] == "H"]
    e_comps = [c for c in components if c[0] == "E"]
    curls = {c: make_curl(spec, c) for c in components}
    pads = {c: _pad_spec(spec, c) for c in components}
    damp = {c: spec.damping_names(c) for c in components}
    poles = {c: spec.pole_name(c) for c in e_comps}
    eps_of = {c: spec.eps_name(c) for c in e_comps}
    if spec.entry not in ("B", "D"):
        raise ValueError(f"entry must be 'B' or 'D', got {spec.entry!r}")
    unknown = [c for c in spec.sources if c not in components]
    if unknown:
        raise ValueError(
            f"spec.sources names {unknown}, which this block does not carry")

    def _advance(state, coeffs, dx, name, base, upd, fields):
        rhs, psis = curls[name](state, coeffs, dx, fields=fields)
        upd.update(psis)
        padded = torch.nn.functional.pad(rhs, pads[name])
        names = damp[name]
        if names is None:               # the block has no damping pair at all
            return base + coeffs.dt * padded
        # no getattr DEFAULT, which is the distinction: a name the coefficients
        # do not carry is a typo and raises, while a name they carry as None is
        # a conductivity that is switched off. Collapsing the two let a
        # misspelled name run the undamped update -- a field that converges and
        # is wrong only once sigma is on.
        c1, c2 = (getattr(coeffs, n) for n in names)
        return base + coeffs.dt * padded if c1 is None else c1 * base + c2 * padded

    def _snapshot(state, upd):
        return type(state)._make([upd.get(f, getattr(state, f))
                                  for f in state._fields])

    def _b_half(state, coeffs, dx, upd, currents, fields):
        """step_db(B) then step_source(B): H from the current E."""
        for name in h_comps:
            upd[name] = _advance(state, coeffs, dx, name,
                                 getattr(state, name), upd, fields)
        for name in h_comps:
            drive = currents.get(name)
            if drive and spec.driven(name):
                amp, mask = drive
                # the magnetic current takes plain dt -- sigma_b lives in the
                # CN pair that already ran, not in the injection
                upd[name] = upd[name] - coeffs.dt * amp * mask

    def _d_half(state, coeffs, dx, upd, currents, fields):
        """step_db(D), step_source(D), update_eh(E): D from the current H."""
        for name in e_comps:
            dname = "D" + name[1]
            upd[dname] = _advance(state, coeffs, dx, name,
                                  getattr(state, dname), upd, fields)
        for name in e_comps:
            drive = currents.get(name)
            if drive and spec.driven(name):
                amp, mask = drive
                dname = "D" + name[1]
                # the current is SUBTRACTED, and scaled by the damping pair's
                # second coefficient under sigma_d exactly as the fused kernel
                # does
                names = damp[name]
                scale = None if names is None else getattr(coeffs, names[1])
                scale = coeffs.dt if scale is None else scale
                upd[dname] = upd[dname] - scale * amp * mask
        for name in e_comps:
            upd[name] = EH_FROM_D(upd["D" + name[1]],
                                  getattr(state, poles[name]).sum(0),
                                  getattr(coeffs, eps_of[name]))

    halves = ((_d_half, _b_half) if spec.entry == "D" else (_b_half, _d_half))

    def step(state, coeffs, dx, currents=None, fields=None):
        currents = currents or {}
        unknown = [c for c in currents if c not in components]
        if unknown:
            raise ValueError(f"currents name {unknown}, not in this block")
        upd = {}
        first, second = halves
        first(state, coeffs, dx, upd, currents, fields)
        second(_snapshot(state, upd), coeffs, dx, upd, currents, fields)
        return _snapshot(state, upd)

    step.curls = curls
    step.entry = spec.entry
    return step
