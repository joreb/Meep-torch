# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Maxwell's curl equations from sympy.vector, compiled to torch.

The twelve steppers in this package advance the same six field components
under different reductions and boundary conditions. The parts that differ per
stepper — ghost cells for mirrors, PMC and rotational symmetry, the nonlinear
E-solve, the ADE pole recurrence — are boundary and solver logic and stay
hand-written. The parts that do NOT differ are the curl, the Yee stagger and
the PML stretch, and those are derived here once.

Three things fall out of the derivation rather than being typed:

**The polarisation split.** Write the 6x6 system, impose ``d/dz = 0``, and it
becomes block-diagonal under ``[Ez, Hx, Hy | Hz, Ex, Ey]``. Those blocks are TM
and TE. ``assert_decoupled`` proves the off-diagonal blocks vanish;
``polarisation_blocks`` labels them by which carries E along the invariant
axis. Membership is derived, only the label is conventional.

**The Yee stagger.** Each component carries its half-cell offset, and
``difference`` evaluates a derivative at the location of the field it updates.
Forward-vs-backward is then a consequence of the two offsets:

    d(Ez)/dy for the Hx update: Ez at y+0,   Hx at y+1/2  -> forward
    d(Hy)/dx for the Ez update: Hy at x+1/2, Ez at x+0    -> backward

**The PML split.** ``d/du -> (1/s_u) d/du`` with
``s_u = k_u + sigma_u/(a_u + i w)`` separates into an unstretched part scaled
by ``1/k_u`` plus a recursive-convolution memory. The closed forms of ``b`` and
``a`` are NOT derived here: they come from the caller's coefficient table, so
the steppers keep meep's own discrete PML coefficients and stay comparable with
meep bit for bit rather than approximately.

The compiled stencils are backend-agnostic despite the ``"torch"`` argument to
``lambdify``. These expressions are pure arithmetic, so sympy emits plain
Python operators -- ``(-c + nb)/h``, ``a*d + b*psi`` -- with no ``torch.`` call
in them. Dispatch is therefore by operator overloading on whatever is passed
in: hand them numpy arrays and the result is numpy, hand them torch tensors and
it is torch, bit for bit the same either way. There is no separate CPU path to
write and no module choice that matters.

Bitwise parity is an evaluation-order property, not an algebra property. Two
details carry it, and both are load-bearing:

* Each difference must come out over a common denominator. Differencing at the
  field's own node returns the expanded ``-c/h + nb/h`` — two divisions — where
  meep divides the difference once, and two divisions round differently from
  one: measured 3.97e-23 on Ez after three steps, which is the whole gap
  between matching meep and not. ``x0`` fixes which samples the difference
  uses; it does not change how sympy prints the result, so ``together`` is
  still what puts it over one denominator.
* Callers must combine the compiled pieces in meep's own association. A curl
  written ``d1/k1 + psi1 - d2/k2 - psi2`` and one written
  ``(d1/k1 + psi1) - (d2/k2 + psi2)`` are the same algebra and different
  floats.
"""
from __future__ import annotations

import os

import sympy as sp
from sympy.calculus.finite_diff import finite_diff_weights
from sympy.vector import CoordSys3D, curl as _vector_curl

HALF = sp.Rational(1, 2)

#: Standard Yee cell offsets in half-cells on (x, y, z). E components sit on
#: edge midpoints, H components on face centres.
YEE: dict[str, tuple] = {
    "Ex": (HALF, 0, 0),
    "Ey": (0, HALF, 0),
    "Ez": (0, 0, HALF),
    "Hx": (0, HALF, HALF),
    "Hy": (HALF, 0, HALF),
    "Hz": (HALF, HALF, 0),
}

E_COMPONENTS = ("Ex", "Ey", "Ez")
H_COMPONENTS = ("Hx", "Hy", "Hz")
COMPONENTS = E_COMPONENTS + H_COMPONENTS
AXES = ("x", "y", "z")

class _Operators(dict):
    """Partial-derivative operator symbols, created on demand.

    Cartesian needs x/y/z; cylindrical needs r/theta/z. Creating on miss keeps
    one table for every coordinate system rather than one per frame.
    """

    def __missing__(self, axis):
        self[axis] = sp.Symbol(f"D{axis}")
        return self[axis]


#: Symbolic partial-derivative operators, one per axis.
D = _Operators({a: sp.Symbol(f"D{a}") for a in AXES})
EPS = sp.Symbol("eps", positive=True)


# --------------------------------------------------------------- the system --
# Every vector operation goes through sympy.vector. Nothing about the curl --
# not a sign, not a metric factor -- is typed here. The operator form the rest
# of this module needs is obtained by REWRITING sympy's result, not by
# transcribing it.

_FRAMES: dict[str, tuple] = {}


def frame(system: str = "cartesian"):
    """``(CoordSys3D, axis_names, base_scalars)`` for a coordinate system.

    ``"cartesian"`` gives (x, y, z); ``"cylindrical"`` gives (r, theta, z),
    where sympy supplies the 1/r metric factors that make a hand-written
    cylindrical curl error-prone.
    """
    if system not in _FRAMES:
        N = CoordSys3D("N")
        if system == "cartesian":
            C, axes = N, ("x", "y", "z")
        elif system == "cylindrical":
            C, axes = N.create_new("Cy", transformation="cylindrical"), \
                ("r", "theta", "z")
        else:
            raise ValueError(f"unknown coordinate system {system!r}")
        _FRAMES[system] = (C, axes, C.base_scalars())
    return _FRAMES[system]


def _operatorise(expr, fns, syms, scalars, axes):
    """``Derivative(f, u)`` -> ``D[u] * sym(f)``.

    The inverse of realising the operator form back into derivatives. This is
    what lets ``system_matrix`` take a Jacobian and ``reduce_axes`` kill an
    axis, while the curl itself still comes from sympy.vector. Terms that are
    not derivatives (the bare ``E_theta / r`` in the cylindrical curl, for
    instance) pass through untouched and land in the matrix as themselves.
    """
    out = sp.expand(expr)
    for axis, u in zip(axes, scalars):
        for name in fns:
            out = out.subs(sp.Derivative(fns[name], u), D[axis] * syms[name])
    # Bare (underived) components have to be collapsed too, and only AFTER the
    # derivatives -- substituting f -> sym first would stop Derivative(f, u)
    # from matching. Cartesian curls are all-derivative so this is a no-op
    # there; the cylindrical curl carries a bare E_theta/r metric term, and
    # leaving it as a Function would hide it from every Jacobian taken w.r.t.
    # the symbol.
    for name in fns:
        out = out.subs(fns[name], syms[name])
    return out


def curl(vec, system: str = "cartesian"):
    """curl of ``(v1, v2, v3)`` in operator form, computed by sympy.vector.

    ``vec`` is a triple of plain symbols standing for the three components in
    ``system``'s basis. The components are lifted to functions of the
    coordinates, handed to ``sympy.vector.curl``, and the result rewritten back
    into ``D[axis] * symbol`` form.
    """
    C, axes, scalars = frame(system)
    names = [str(v) for v in vec]
    fns = {n: sp.Function(n)(*scalars) for n in names}
    syms = {n: sp.Symbol(n) for n in names}
    field = sum((fns[n] * b for n, b in zip(names, C.base_vectors())),
                sp.vector.Vector.zero)
    result = _vector_curl(field)
    return tuple(_operatorise(result.dot(b), fns, syms, scalars, axes)
                 for b in C.base_vectors())


def maxwell_rhs(system: str = "cartesian"):
    """The six right-hand sides of ``dH/dt = -curl E``, ``dE/dt = curl H/eps``.

    Returns ``(rhs, F)`` where ``rhs`` maps component name -> expression and
    ``F`` maps component name -> its symbol.
    """
    F = {c: sp.Symbol(c) for c in COMPONENTS}
    cE = curl((F["Ex"], F["Ey"], F["Ez"]), system)
    cH = curl((F["Hx"], F["Hy"], F["Hz"]), system)
    rhs = {c: cH[i] / EPS for i, c in enumerate(E_COMPONENTS)}
    rhs.update({c: -cE[i] for i, c in enumerate(H_COMPONENTS)})
    return rhs, F


def system_matrix(reduce_axes: tuple = ()):
    """The 6x6 coupling matrix ``M`` with ``d(F)/dt = M F``.

    ``reduce_axes`` names the axes along which nothing varies, i.e. the axes
    whose derivative operator is set to zero. ``()`` is full 3D, ``("z",)`` is
    the 2D reduction that produces TM and TE, ``("y", "z")`` is 1D.
    """
    rhs, F = maxwell_rhs()
    subs = {D[a]: 0 for a in reduce_axes}
    return sp.Matrix([
        [sp.simplify(sp.diff(sp.expand(rhs[r].subs(subs)), F[c]))
         for c in COMPONENTS]
        for r in COMPONENTS])


# The permutation that exposes the 2D block structure. Deliberately not named
# "TM" and "TE" here — these are two groups of three components, and which is
# which polarisation is read off afterwards by polarisation_blocks().
BLOCK_A = ("Ez", "Hx", "Hy")
BLOCK_B = ("Hz", "Ex", "Ey")


def _submatrix(M, rows, cols):
    ri = [COMPONENTS.index(r) for r in rows]
    ci = [COMPONENTS.index(c) for c in cols]
    return sp.Matrix([[M[r, c] for c in ci] for r in ri])


def assert_decoupled(reduce_axes: tuple = ("z",)):
    """Prove the reduced system is block-diagonal; return the reduced matrix.

    This is the claim the polarisation split rests on: that TM and TE are a
    consequence of the reduction, not an assumption. Raises if either
    off-diagonal 3x3 block is non-zero.
    """
    M = system_matrix(reduce_axes)
    for rows, cols, tag in ((BLOCK_A, BLOCK_B, "A<-B"), (BLOCK_B, BLOCK_A, "B<-A")):
        off = _submatrix(M, rows, cols)
        if not all(sp.simplify(e) == 0 for e in off):
            raise AssertionError(
                f"{tag} block is not zero under d/d{reduce_axes}; the system "
                f"does not decouple:\n{off}")
    return M


def polarisation_blocks(reduce_axes: tuple = ("z",)) -> dict:
    """Read the polarisations off the decoupled system.

    With propagation in the plane, the block carrying E along the invariant
    axis is TM and the one carrying H along it is TE. The MEMBERSHIP of each
    block is derived; only the label is conventional.
    """
    M = assert_decoupled(reduce_axes)
    invariant = reduce_axes[0] if reduce_axes else "z"
    out = {}
    for group in (BLOCK_A, BLOCK_B):
        key = "tm" if f"E{invariant}" in group else "te"
        out[key] = {"components": group, "matrix": _submatrix(M, group, group)}
    return out


# ---------------------------------------------------------------- stencils --

def difference(field: str, axis: str, target: str, h, sym, order: int = 2):
    """The finite difference of ``field`` along ``axis`` at ``target``'s node.

    The half-cell offset between the two components is handed to sympy as the
    evaluation point ``x0``, so the Yee stagger is an INPUT to the finite
    difference rather than a branch around it. sympy then picks the weights,
    and forward-vs-backward is simply which side of ``x0`` the samples fall on.

    Accuracy order becomes a parameter rather than a rewrite: ``order=4`` gives
    the standard fourth-order staggered stencil
    ``(-27 f(v) + f(v-h) + 27 f(v+h) - f(v+2h)) / (24 h)``.

    The weights themselves come from :func:`difference_at`, which is the
    Fornberg algorithm and does not assume uniform spacing. This function is
    the uniform-grid Yee convenience wrapper over it.

    ``sym`` maps a relative sample index (an integer multiple of ``h`` from the
    field's own node) to the symbol standing for that neighbour.
    """
    delta = YEE[field][AXES.index(axis)] - YEE[target][AXES.index(axis)]
    if abs(delta) != HALF:
        raise ValueError(
            f"d({field})/d{axis} at {target}: offset {delta} is not +/-1/2, so "
            "these components are not Yee-staggered along this axis")
    ks = stencil_offsets(delta, order)
    # coordinates measured from the field's own node; the target sits at -delta
    return difference_at([k * h for k in ks], -delta * h,
                         [sym(k) for k in ks])


def stencil_offsets(delta, order: int = 2):
    """Which integer sample offsets straddle a node ``delta`` cells away.

    ``order`` samples, symmetric about the target. For the Yee case
    (``delta = -/+ 1/2``) order 2 gives ``(0, 1)`` or ``(-1, 0)``, order 4
    gives ``(-1, 0, 1, 2)`` or ``(-2, -1, 0, 1)``.
    """
    if order < 2 or order % 2:
        raise ValueError(f"accuracy order must be even and >= 2, got {order}")
    m = order // 2
    return tuple(range(-(m - 1), m + 1) if delta < 0 else range(-m, m))


def difference_at(coords, x0, values, order: int = 1):
    """First (or ``order``-th) derivative at ``x0`` from samples at ``coords``.

    The weights come from :func:`sympy.calculus.finite_diff.finite_diff_weights`
    -- the Fornberg algorithm -- so ``coords`` need NOT be uniformly spaced and
    ``x0`` need not be one of them. That is what lets the same routine serve a
    uniform Yee grid, a graded mesh, and a finite-volume cell: the node
    positions enter the weights symbolically rather than being assumed equal.

    The result is put over a common denominator. The weighted sum
    ``sum(w_i * y_i)`` expands to one division per sample, where meep divides a
    difference once, and the two round differently -- 3.97e-23 on Ez after
    three steps, which is the whole distance between matching meep and not.
    """
    w = finite_diff_weights(order, list(coords), x0)[order][-1]
    return sp.together(sum(wi * yi for wi, yi in zip(w, values)))


def pml_split(diff_expr, psi, b, a, k):
    """The stretched-coordinate PML term, derived from ``1/s_u``.

    Returns ``(psi_new, stretched)`` where::

        psi_new   = b*psi + a*(d/du)
        stretched = (d/du)/k + psi_new

    ``b``, ``a`` and ``k`` are supplied by the caller rather than derived: they
    are meep's own discrete PML coefficients, so the result matches meep's
    recursion exactly instead of approximately.
    """
    psi_new = b * psi + a * diff_expr
    return psi_new, diff_expr / k + psi_new


def discrete_curl(vec, target: str, h, sym, system: str = "cartesian",
                  order: int = 2, stretch=None):
    """The curl of ``vec``, discretised at ``target``'s node, in one step.

    Where :func:`curl` returns the continuous operator form and
    :func:`difference` discretises one derivative at a time, this walks the
    curl expression and replaces every ``Derivative`` with its staggered finite
    difference -- so the continuous vector calculus and the discretisation are
    joined rather than related by hand.

    ``sym(component_name, offset)`` supplies the symbol for a component sampled
    ``offset`` cells from its own node along the axis being differentiated.

    ``stretch(component_name, axis, difference)`` optionally wraps each derived
    difference before it is combined. That hook is what makes the PML derivable
    rather than bolted on: meep does not apply the stretch to the assembled
    curl, it applies a separate recursion and ``1/k`` to EACH difference and
    then combines, so the stretch has to enter per-derivative, here, rather
    than around the result.
    """
    C, axes, scalars = frame(system)
    names = [str(v) for v in vec]
    fns = {n: sp.Function(n)(*scalars) for n in names}
    field = sum((fns[n] * b for n, b in zip(names, C.base_vectors())),
                sp.vector.Vector.zero)
    idx = {str(u): a for a, u in zip(axes, scalars)}

    def discretise(expr, *var_counts):
        # replace(Derivative, ...) hands the callback the Derivative's args,
        # where each variable arrives as a (symbol, count) pair rather than a
        # bare symbol.
        first = var_counts[0]
        var = first[0] if isinstance(first, (tuple, sp.Tuple)) else first
        name = expr.func.__name__
        axis = idx[str(var)]
        delta = YEE[name][AXES.index(axis)] - YEE[target][AXES.index(axis)]
        ks = stencil_offsets(delta, order)
        d = difference_at([k * h for k in ks], -delta * h,
                          [sym(name, k) for k in ks])
        return d if stretch is None else stretch(name, axis, d)

    out = []
    for basis in C.base_vectors():
        comp = _vector_curl(field).dot(basis)
        out.append(comp.replace(sp.Derivative, discretise))
    return tuple(out)


# ---------------------------------------------------------------- compiled --

_DIFF_CACHE: dict[tuple, object] = {}


def compiled_difference(field: str, axis: str, target: str):
    """``f(centre, neighbour, h)`` in torch for the derived difference.

    Which neighbour is the +1 and which the -1 sample is fixed by the Yee
    offsets inside :func:`difference`, not by the caller. Cached: the
    derivation runs once per (field, axis, target).
    """
    key = (field, axis, target)
    if key not in _DIFF_CACHE:
        c, nb, h = sp.symbols("c nb h")
        expr = difference(field, axis, target, h, lambda i: c if i == 0 else nb)
        _DIFF_CACHE[key] = sp.lambdify((c, nb, h), expr, "torch")
    return _DIFF_CACHE[key]


def compiled_uniform_difference(direction: str, order: int = 2):
    """``f(centre, neighbour, h)`` for a plain forward or backward difference.

    The Yee-aware :func:`compiled_difference` derives WHICH direction from the
    two components' offsets. Some steppers already know the direction and have
    no field pair to derive it from -- the Bloch steppers take their neighbour
    by periodic `torch.roll` rather than by slicing, so there is no "target
    component" whose node the difference lands on. This is the same derivation
    with the direction given rather than inferred, so those callers do not have
    to borrow an unrelated component pair to get the right form.
    """
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be forward or backward, got {direction!r}")
    h = sp.Symbol("h")
    delta = -HALF if direction == "forward" else HALF
    ks = stencil_offsets(delta, order)
    # names must be valid Python identifiers: sympy silently swaps a Dummy in
    # for anything lambdify cannot use as a parameter name, and "nb-1" is not
    # one, so the generated source came out referencing Dummy_NN.
    syms = {k: sp.Symbol("c" if k == 0 else
                         f"nb_{'p' if k > 0 else 'm'}{abs(k)}") for k in ks}
    expr = difference_at([k * h for k in ks], -delta * h, [syms[k] for k in ks])
    # Signature is pinned to (centre, neighbours..., h) EXPLICITLY. Building it
    # from `ks` instead puts the centre second for a backward difference --
    # stencil_offsets returns (0, 1) forward but (-1, 0) backward -- so every
    # caller passing (c, neighbour, h) silently bound them swapped and got the
    # negated derivative. Measured as divergence to 1e+04 / 1e+09 rather than a
    # float-level miss. The generated BODY looked right either way; only the
    # parameter order was wrong, which is not visible from the body.
    args = [syms[0]] + [syms[k] for k in ks if k != 0] + [h]
    return sp.lambdify(args, expr, "torch")


def _compile_pml():
    """``(psi_new_fn, stretched_fn)`` in torch.

    ``stretched`` is compiled against a bare symbol standing for ``psi_new`` so
    the recursion is evaluated once per step and reused, rather than recomputed
    inside the curl.
    """
    d, psi, b, a, k = sp.symbols("d psi b a k")
    psi_sym = sp.Symbol("psi_new")
    psi_new, stretched = pml_split(d, psi, b, a, k)
    return (sp.lambdify((d, psi, b, a), psi_new, "torch"),
            sp.lambdify((d, psi_sym, k), stretched.subs(psi_new, psi_sym),
                        "torch"))


#: ``psi_new(d, psi, b, a)`` — the CPML recursion.
#: ``stretched(d, psi_new, k)`` — the stretched derivative ``d/k + psi_new``.
PSI_NEW, STRETCHED = _compile_pml()


def whole_curl_enabled() -> bool:
    """Whether steppers assemble the D-phase curl as ONE derived expression.

    Two forms of the same algebra are available. The default is
    :func:`compiled_stretched_curl`, which hands the whole curl to sympy and
    lets it assemble its own canonical ``Add`` -- derived in one piece. The
    alternative composes the derived per-difference pieces in meep's
    left-to-right association, ``d1/k1 + psi1 - d2/k2 - psi2``, and is
    bitwise-identical to meep.

    The two are physically indistinguishable. Against the analytic 2D Green's
    function ``(i/4) H_0^(1)(kr)`` both land at rel-L2 ``4.796918e-02`` -- the
    resolution-20 grid error, identical to seven digits -- and they differ from
    each other by ``1.520e-11``, nine orders below that (see
    ``symbolic_test/plot_cw_comparison.py``). That gap is negligible against
    every error the discretisation already carries, so the derived form is the
    default: the curl is derived, not hand-composed, everywhere.

    The one real cost is Krylov conditioning. ``solve_cw``'s BiCGSTAB(2) has a
    residual floor set by rounding in the operator application, and sympy's
    canonical ``Add`` sits just the wrong side of ``tol=1e-10``; it converges
    from ``1e-9`` down, to the same answer. Set
    ``MEEP_TORCH_WHOLE_CURL=0`` to get meep's bitwise left-to-right
    association back -- needed for bit-for-bit meep comparison and for a
    ``solve_cw`` at ``1e-10``.

    Read at CALL time, not captured at import, so a test can flip it.

    Only the TM D phase has a whole-curl alternative today; stepper_te and
    stepper_1d compose the pieces unconditionally.
    """
    return os.environ.get("MEEP_TORCH_WHOLE_CURL", "1").lower() not in (
        "0", "false", "no")


def compiled_stretched_curl(vec, target: str, system: str = "cartesian",
                            order: int = 2):
    """Compile the CPML-stretched discrete curl at ``target`` into torch.

    Returns ``(fn, argnames)``. ``fn`` takes, in ``argnames`` order: each
    sampled component (``Hx``, ``Hx_m1``, ...), the OLD psi memory per
    (component, axis), then ``b_<axis>``, ``a_<axis>``, ``k_<axis>``, then the
    spacing.

    NOTE ON PARITY. This assembles the whole curl as one sympy expression, so
    the evaluation order is sympy's canonical Add, not meep's left-to-right
    ``d1/k1 + psi1 - d2/k2 - psi2``. Measured against meep's composition the
    two agree to ~9e-18 relative (7.3e-12 absolute on unit-scale inputs) --
    mathematically identical, one ulp apart, NOT bitwise. This is the default
    form; composing the per-difference pieces in the stepper (see
    :func:`compiled_difference`) is what reproduces meep exactly, and
    :func:`whole_curl_enabled` is the switch between them.
    """
    h = sp.Symbol("h")
    used: list = []

    def sym(name, k):
        t = sp.Symbol(name if k == 0 else
                      f"{name}_{'p' if k > 0 else 'm'}{abs(k)}")
        if t not in used:
            used.append(t)
        return t

    psis: list = []
    coeffs: list = []

    def stretch(name, axis, d):
        psi = sp.Symbol(f"psi_{name}{axis}")
        b, a, k = (sp.Symbol(f"b_{axis}"), sp.Symbol(f"a_{axis}"),
                   sp.Symbol(f"k_{axis}"))
        if psi not in psis:
            psis.append(psi)
        for c in (b, a, k):
            if c not in coeffs:
                coeffs.append(c)
        return pml_split(d, psi, b, a, k)[1]

    comps = discrete_curl(vec, target, h, sym, system, order, stretch=stretch)
    # the component of the curl that drives `target`: the one whose axis
    # matches the target's own vector direction
    axis_of = {"x": 0, "y": 1, "z": 2}[target[-1]]
    expr = comps[axis_of]
    # Take the arguments from the SELECTED component only. discrete_curl builds
    # all three, so collecting as they are created would drag in the other
    # components and axes -- for a 2D TM curl that is Hz and every z-coefficient,
    # none of which appear in the expression actually returned.
    free = expr.free_symbols
    args = ([a for a in used if a in free] + [a for a in psis if a in free]
            + [a for a in coeffs if a in free] + [h])
    return sp.lambdify(args, expr, "torch"), [str(a) for a in args]


# ------------------------------------------------------------- symmetries --
# A symmetry operation acts on TWO things, and only one of them is algebra.
#
#   the ADDRESS -- which grid sample is the partner of this one. That is a
#   permutation of grid points; `flip`, `transpose` and the slice arithmetic in
#   the steppers ARE that permutation matrix, applied in O(N) instead of
#   materialised as N x N. It stays index algebra.
#
#   the VALUE -- what each field component becomes. That is an orthogonal
#   matrix acting on the component index, and it is derived here. E is a polar
#   vector and transforms as ``R E``; H, B and every curl of a polar vector is
#   an AXIAL vector and transforms as ``det(R) R H``. That ``det(R)`` is the
#   entire reason a mirror flips the H signs the other way round from E, and it
#   is the sign most often typed wrong by hand.


def mirror_matrix(normal) -> sp.Matrix:
    """Reflection in the plane through the origin with unit ``normal``.

    Householder: ``R = I - 2 n n^T``. Built from the normal rather than
    entered, so no sign is typed.
    """
    n = sp.Matrix(normal)
    n = n / sp.sqrt((n.T * n)[0, 0])
    return sp.eye(3) - 2 * n * n.T


def rotation_matrix(turns) -> sp.Matrix:
    """Active rotation about +z by ``turns`` quarter-turns (1 = 90 degrees).

    Active: it maps a POINT to its image, so ``rotation_matrix(1)`` sends the
    +x axis onto +y. sympy's ``rot_axis3`` rotates the frame, which is the
    inverse, hence the transpose.
    """
    return sp.rot_axis3(sp.Rational(turns, 1) * sp.pi / 2).T


#: The symmetries the engine supports, as the orthogonal matrix of each.
SYMMETRIES = {
    "mirror_x": mirror_matrix((1, 0, 0)),
    "mirror_y": mirror_matrix((0, 1, 0)),
    "rot2_z": rotation_matrix(2),
    "rot4_z": rotation_matrix(1),
}


def field_transform(R: sp.Matrix, family: str) -> sp.Matrix:
    """How a field's three components mix under ``R``.

    ``family`` is ``"E"`` for a polar vector (E, D, and the current J) or
    ``"H"`` for an axial one (H, B, M). The axial case carries ``det(R)``:
    under a reflection ``det(R) = -1``, which is why E and H pick up opposite
    signs about the same mirror plane, and under a rotation ``det(R) = +1``,
    which is why they pick up the same ones about a rotation axis.
    """
    if family not in ("E", "H"):
        raise ValueError(f"family must be E or H, got {family!r}")
    return R if family == "E" else R.det() * R


#: Which family each component belongs to, from its own name. D and B follow
#: E and H because the constitutive relations are scalar (or, for an
#: anisotropic medium, a tensor that the symmetry must itself preserve).
_FAMILY = {"E": "E", "D": "E", "J": "E", "H": "H", "B": "H", "M": "H"}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def component_transform(kind: str, component: str) -> dict:
    """``{source component: coefficient}`` for ``component`` under ``kind``.

    The image of a component is in general a COMBINATION -- a quarter turn
    about z sends ``Hx`` to ``-Hy`` -- so the return is a map, not a sign. A
    caller that can only carry one scalar per component should use
    :func:`component_sign`, which refuses exactly the mixing cases.
    """
    if kind not in SYMMETRIES:
        raise ValueError(f"unknown symmetry {kind!r}, have {sorted(SYMMETRIES)}")
    field, axis = component[0], component[-1]
    if field not in _FAMILY or axis not in _AXIS_INDEX:
        raise ValueError(f"not a field component: {component!r}")
    row = field_transform(SYMMETRIES[kind], _FAMILY[field])[_AXIS_INDEX[axis], :]
    return {f"{field}{a}": sp.nsimplify(row[i])
            for a, i in _AXIS_INDEX.items() if row[i] != 0}


def component_sign(kind: str, component: str) -> int:
    """``+1``/``-1`` for ``component`` under ``kind``, when one exists.

    Raises when the symmetry MIXES components, because then no per-component
    sign exists and silently returning one would be wrong rather than
    unsupported. A quarter turn about z is the case that mixes: ``Hx -> -Hy``.
    """
    t = component_transform(kind, component)
    if list(t) != [component]:
        image = " + ".join(f"{c}*{s}" for c, s in t.items()) or "0"
        raise ValueError(
            f"{component} has no sign under {kind}: it maps to {image}. "
            f"Use component_transform and carry the mixing.")
    return int(t[component])


def describe() -> None:
    """Print the derivation: 3D system, 2D reduction, and the two blocks."""
    print("3D system  d(F)/dt = M F,  F =", COMPONENTS)
    sp.pprint(system_matrix())
    print("\nafter d/dz = 0:")
    sp.pprint(system_matrix(("z",)))
    for name, blk in sorted(polarisation_blocks().items()):
        print(f"\n{name.upper()} block {blk['components']}:")
        sp.pprint(blk["matrix"])


if __name__ == "__main__":
    describe()
