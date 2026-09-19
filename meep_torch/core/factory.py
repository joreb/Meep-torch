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

from .kernels import EH_FROM_D, PSI_NEW, STRETCHED
from .symbolic import (AXES, FOLD, BlockSpec, compiled_curl_composition,
                       compiled_difference, curl_terms, field_shape,
                       psi_extent, stencil_offsets, yee_table)

__all__ = ["CurlTerm", "curl_recipe", "make_curl", "make_step", "TM_DAMPING"]


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
    want = psi_extent(target, extents, boundaries, origin=spec.origin)
    out = []
    for source, axis, sign in curl_terms(target):
        a = AXES.index(axis)
        if a >= ndim:                       # reduced away, e.g. z in a 2D block
            continue
        src_shape = field_shape(source, extents, boundaries,
                                origin=spec.origin)
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


def make_curl(spec: BlockSpec, target: str):
    """``f(state, coeffs, dx) -> (curl, {psi_name: new_psi})``.

    The spec work happens HERE, once. The returned closure touches tensors and
    slice objects only.
    """
    terms, want = curl_recipe(spec, target)

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

    ``step(state, coeffs, dx, src_amp=0.0, src_mask=None, fields=None)``.

    The order is meep's `fields::step`, and it is NOT an implementation detail:
    ``step_db(B) -> update_eh(H) -> step_db(D) -> step_source -> update_pols
    -> update_eh(E)``. H is advanced from the OLD E and E from the NEW H, which
    is the leapfrog; running the phases in any other order changes the answer
    while still converging, which is the worst kind of wrong.

    Everything conventional comes off the spec: `damping_names` for the CN
    coefficient pair and `pole_name` for where a component's polarisation sum
    lives. Both were TM constants hard-typed here, and both are wrong for TE --
    `Px_cur`/`Py_cur` rather than one shared `P_cur`, which no TM gate can see.

    Scope: no poles beyond the sum, no Kerr, no magnetic poles yet. Those are
    `kernels.ADE_POLE` and the fixed point, and they are composition rather
    than derivation -- see the task page. Source injection hits every E
    component with the same mask, which is the FOURTH BlockSpec gap and holds
    only while one E component exists; TE is where it breaks.
    """
    components = spec.components
    h_comps = [c for c in components if c[0] == "H"]
    e_comps = [c for c in components if c[0] == "E"]
    curls = {c: make_curl(spec, c) for c in components}
    pads = {c: _pad_spec(spec, c) for c in components}
    damp = {c: spec.damping_names(c) for c in components}
    poles = {c: spec.pole_name(c) for c in e_comps}

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

    def step(state, coeffs, dx, src_amp=0.0, src_mask=None, fields=None):
        upd = {}
        # step_db(B): H from the OLD E
        for name in h_comps:
            upd[name] = _advance(state, coeffs, dx, name,
                                 getattr(state, name), upd, fields)
        mid = type(state)._make([upd.get(f, getattr(state, f))
                                 for f in state._fields])
        # step_db(D): D from the NEW H
        for name in e_comps:
            dname = "D" + name[1]
            upd[dname] = _advance(mid, coeffs, dx, name,
                                  getattr(mid, dname), upd, fields)
        # step_source(D): the current is SUBTRACTED, and scaled by the damping
        # pair's second coefficient under sigma_d exactly as the fused kernel
        # does.
        if src_mask is not None and src_amp:
            for name in e_comps:
                dname = "D" + name[1]
                names = damp[name]
                scale = None if names is None else getattr(coeffs, names[1])
                scale = coeffs.dt if scale is None else scale
                upd[dname] = upd[dname] - scale * src_amp * src_mask
        # update_eh(E)
        for name in e_comps:
            upd[name] = EH_FROM_D(upd["D" + name[1]],
                                  getattr(mid, poles[name]).sum(0),
                                  coeffs.eps_inf)
        return type(state)._make([upd.get(f, getattr(state, f))
                                  for f in state._fields])

    step.curls = curls
    return step
