# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Build a curl from a spec, once, and execute it many times.

Twelve steppers hand-write the same assembly: difference each source along its
axis, trim the other axes to the target's interior, run the CPML recursion,
compose in meep's order, and hand the result to the field update. None of that
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
  parity with meep is an association property.

THE MEASURED CONSTRAINT, and the reason this is a factory rather than a
generic function: resolve the spec at BUILD time. A version that re-read the
Yee table, compared sympy Rationals and looked coefficients up by string on
every step measured 1.5-2x slower, and that 1.5-2x was mistaken for the cost of
genericity by two sessions. Hoisted into a closure over plain ints, slice
objects and bound functions it runs at parity -- 0.95-0.99x in 2D, 1.00x in 3D
on CUDA. No codegen; a closure.

``coeffs`` is an ARGUMENT to the returned function, not a capture, so the same
built step serves every rebuild of the coefficients an adjoint produces and
stays a pure function of two pytrees.

SCOPE. Cartesian only, and no symmetry folds yet. `_reject_noncartesian`
guards the cylindrical frame upstream (the bare ``E_phi / r`` term has no
difference to attach to), and a fold changes which rows are live rather than
which terms exist -- see ``cut_plane_row`` and ``tasks/fdtd_sympy-stepper-kernels``.
"""
from typing import NamedTuple

import torch

from .kernels import PSI_NEW, STRETCHED
from .symbolic import (AXES, FOLD, compiled_curl_composition,
                       compiled_difference, curl_terms, field_shape,
                       psi_extent, stencil_offsets, yee_table)

__all__ = ["CurlTerm", "curl_recipe", "make_curl"]


class CurlTerm(NamedTuple):
    """One difference of a curl, fully resolved at build time."""
    source: str                  # component being differenced
    axis: str                    # axis it is differenced along
    sign: int                    # +1 or -1, from curl_terms
    diff: object                 # compiled_difference(source, axis, target)
    centre: tuple                # slice for the field's OWN node (k = 0)
    neighbour: tuple             # slice for the other sample
    psi: str                     # state field holding this term's CPML memory
    coeff: str                   # coefficient suffix, e.g. "ex" -> m.b_ex
    ghost: str                   # axis needing a cut-plane row before the
                                 # difference, or "" -- see make_curl


def _family(component: str) -> str:
    """``e`` for an E-target, ``h`` for an H-target -- the coefficient family.

    `stepper_tm` reads `m.b_ex`/`m.b_ey` in the Ez update and `m.b_hx`/`m.b_hy`
    in the H update, so the family follows the TARGET and the letter follows
    the DIFFERENCE AXIS. Deriving it beats a lookup table that has to be
    rewritten per stepper.
    """
    return component[0].lower()


def curl_recipe(target: str, extents, boundaries, ndim: int = None) -> tuple:
    """Resolve every term of ``target``'s curl against this grid.

    Returned terms carry slices, not indices, so the step does no arithmetic on
    shapes. The trim is the part worth stating: a term is differenced along its
    own axis and trimmed on the OTHERS, because the result has to land on the
    target's psi extent -- which is what makes one rule cover both terms of a
    curl instead of two hand-written slicings.
    """
    ndim = ndim if ndim is not None else len(extents)
    want = psi_extent(target, extents, boundaries)
    out = []
    for source, axis, sign in curl_terms(target):
        a = AXES.index(axis)
        if a >= ndim:                       # reduced away, e.g. z in a 2D block
            continue
        src_shape = field_shape(source, extents, boundaries)
        lo, hi = [slice(None)] * ndim, [slice(None)] * ndim
        for b in range(ndim):
            if b == a:
                # the differenced axis: the difference itself removes the row,
                # and `stencil_offsets` has already fixed which one
                lo[b], hi[b] = slice(None, -1), slice(1, None)
                # a fold hands the difference an extra row at the cut, so the
                # source is one longer there than its own allocation
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
        out.append(CurlTerm(
            source=source, axis=axis, sign=sign,
            diff=compiled_difference(source, axis, target),
            centre=tuple(c_sl), neighbour=tuple(n_sl),
            ghost=(axis if boundaries[a] is FOLD else ""),
            psi=f"psi_{source}{axis}",
            coeff=f"{_family(target)}{axis}",
        ))
    if not out:
        raise ValueError(
            f"{target} has no live curl terms on a {ndim}D block: every source "
            "is differenced along an axis the reduction removed")
    if len(out) > 2:
        raise ValueError(f"{target} has {len(out)} curl terms, expected 1 or 2")
    return tuple(out), want


def make_curl(target: str, extents, boundaries, ndim: int = None):
    """``f(state, coeffs, dx) -> (curl, {psi_name: new_psi})``.

    The spec work happens HERE, once. The returned closure touches tensors and
    slice objects only.
    """
    terms, want = curl_recipe(target, extents, boundaries, ndim=ndim)

    def _one(state, coeffs, dx, t, psis, fields):
        f = fields[t.source] if fields and t.source in fields else getattr(state, t.source)
        d = t.diff(f[t.centre], f[t.neighbour], dx)
        p = PSI_NEW(d, getattr(state, t.psi),
                    getattr(coeffs, f"b_{t.coeff}"),
                    getattr(coeffs, f"a_{t.coeff}"))
        psis[t.psi] = p
        return d, p, getattr(coeffs, f"k_{t.coeff}")

    if len(terms) == 1:
        # A reduced block's H phase: one term, so there is no association to
        # preserve and no composition to derive. `dF/dt = sign * stretched`,
        # and a caller writing `F - dt*stretched` gets the same floats as
        # `F + dt*(-stretched)` -- IEEE negation is exact.
        (t,) = terms
        sign = t.sign

        def curl(state, coeffs, dx, fields=None):
            psis = {}
            d, p, k = _one(state, coeffs, dx, t, psis, fields)
            out = STRETCHED(d, p, k)
            return (out if sign > 0 else -out), psis
    else:
        compose = compiled_curl_composition(n_terms=2)
        (t_pos,), (t_neg,) = ([t for t in terms if t.sign > 0],
                              [t for t in terms if t.sign < 0])

        def curl(state, coeffs, dx, fields=None):
            psis = {}
            dp, pp, kp = _one(state, coeffs, dx, t_pos, psis, fields)
            dn, pn, kn = _one(state, coeffs, dx, t_neg, psis, fields)
            # meep's association, emitted by the derivation: the positive term
            # is stretched and the negative one enters as (d/k) and psi apart.
            return compose(STRETCHED(dp, pp, kp), dn, kn, pn), psis

    curl.terms = terms
    curl.shape = want
    return curl
