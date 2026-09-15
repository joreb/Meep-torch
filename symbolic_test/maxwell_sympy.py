# SPDX-License-Identifier: MIT
# Maxwell's curl equations in sympy: a 3D state of all six field components,
# PML-stretched, from which the 2D polarisations are DERIVED rather than
# declared. Pure sympy -- this module imports nothing else, not torch, not
# meep_torch. stepper_sympy.py compiles what comes out of here.
#
# The point: meep_torch carries stepper_tm.py and stepper_te.py as two separate
# hand-written implementations, and fdtd_pylops_2d / fdtd_devito_2d each hand-
# write a TM curl again. All of those are the same six equations with different
# components zeroed. Here the six are written once, in 3D:
#
#   dH/dt = -curl E          dD/dt = curl H,  D = eps E   (mu = 1)
#
# Imposing d/dz = 0 makes the 6x6 system block-diagonal under the permutation
# [Ez, Hx, Hy | Hz, Ex, Ey], and those two 3x3 blocks ARE TM and TE. Nothing in
# this file names a polarisation until after the blocks fall out --
# `polarisation_blocks()` reads them off the reduced matrix, and
# `assert_decoupled()` proves the off-diagonal blocks vanish.
#
# The Yee stagger is likewise not typed out. Each component carries its
# half-cell offset (the standard Yee cell, YEE below), and `difference()`
# evaluates a derivative AT THE LOCATION OF THE FIELD IT UPDATES. Whether that
# lands on a forward or a backward difference is then a consequence of the two
# offsets, not a choice:
#
#   dEz/dy for the Hx update: Ez at y+0,   Hx at y+1/2  -> forward
#   dHy/dx for the Ez update: Hy at x+1/2, Ez at x+0    -> backward
#
# getting that pair backwards is exactly what made the devito straight-
# waveguide diverge to 2.1e+82 (see fdtd_devito.py).
#
# PML: the stretched-coordinate form d/du -> (1/s_u) d/du with
# s_u = k_u + sigma_u/(a_u + i*omega) splits as
#
#   (1/s_u) d/du  ->  (1/k_u) d/du  +  psi_u
#
# and the convolution term psi_u obeys the recursion psi^{n+1} = b*psi^n +
# a*(d/du)^{n+1}. `pml_split()` derives that split. The closed forms of b and a
# are NOT derived here -- they come from the caller's coefficient table, so
# this stepper uses meep's own discrete PML coefficients and can be compared
# with step_tm bitwise instead of approximately.
import sympy as sp

HALF = sp.Rational(1, 2)

# Standard Yee cell offsets in half-cells on (x, y, z). E components sit on
# edge midpoints, H components on face centres.
YEE = {
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

# symbolic partial-derivative operators, one per axis
D = {a: sp.Symbol(f"D{a}", commutative=True) for a in AXES}
EPS = sp.Symbol("eps", positive=True)


def curl(vec):
    """curl of (vx, vy, vz) with the symbolic D operators."""
    vx, vy, vz = vec
    return (D["y"] * vz - D["z"] * vy,
            D["z"] * vx - D["x"] * vz,
            D["x"] * vy - D["y"] * vx)


def maxwell_rhs():
    """The six right-hand sides, in COMPONENTS order.

    dH/dt = -curl E ; dE/dt = curl H / eps. Returns a dict name -> expression
    in the six field symbols and the D operators."""
    F = {c: sp.Symbol(c) for c in COMPONENTS}
    cE = curl((F["Ex"], F["Ey"], F["Ez"]))
    cH = curl((F["Hx"], F["Hy"], F["Hz"]))
    rhs = {}
    for i, c in enumerate(E_COMPONENTS):
        rhs[c] = cH[i] / EPS
    for i, c in enumerate(H_COMPONENTS):
        rhs[c] = -cE[i]
    return rhs, F


def system_matrix(reduce_axes=()):
    """The 6x6 coupling matrix M with d(F)/dt = M F, optionally after setting
    d/du = 0 for each axis in `reduce_axes`.

    Rows and columns are in COMPONENTS order. Entries are linear in the D
    operators, so M is the symbolic Jacobian of the RHS."""
    rhs, F = maxwell_rhs()
    subs = {D[a]: 0 for a in reduce_axes}
    rows = []
    for r in COMPONENTS:
        expr = sp.expand(rhs[r].subs(subs))
        rows.append([sp.simplify(sp.diff(expr, F[c])) for c in COMPONENTS])
    return sp.Matrix(rows)


# The permutation that exposes the block structure. Deliberately NOT called
# "TM" and "TE" here -- these are just two groups of three components; which
# one is which polarisation is read off afterwards by polarisation_blocks().
BLOCK_A = ("Ez", "Hx", "Hy")
BLOCK_B = ("Hz", "Ex", "Ey")


def _submatrix(M, rows, cols):
    ri = [COMPONENTS.index(r) for r in rows]
    ci = [COMPONENTS.index(c) for c in cols]
    return sp.Matrix([[M[r, c] for c in ci] for r in ri])


def assert_decoupled(reduce_axes=("z",)):
    """Prove the reduced system is block-diagonal: the two off-diagonal 3x3
    blocks must be identically zero. This is the whole claim -- that the
    polarisations are a consequence of d/dz = 0, not an assumption."""
    M = system_matrix(reduce_axes)
    for rows, cols, tag in ((BLOCK_A, BLOCK_B, "A<-B"), (BLOCK_B, BLOCK_A, "B<-A")):
        off = _submatrix(M, rows, cols)
        if not all(sp.simplify(e) == 0 for e in off):
            raise AssertionError(f"{tag} block is not zero under d/d{reduce_axes}"
                                 f" -- system does not decouple:\n{off}")
    return M


def polarisation_blocks(reduce_axes=("z",)):
    """Read the polarisations off the decoupled system.

    TM/TE is a naming convention over the reduced blocks: with propagation in
    the xy plane, the block carrying Ez (E along the invariant axis) is TM and
    the one carrying Hz is TE. The MEMBERSHIP of each block is derived;
    only the label is conventional."""
    M = assert_decoupled(reduce_axes)
    invariant = reduce_axes[0] if reduce_axes else "z"
    out = {}
    for group in (BLOCK_A, BLOCK_B):
        e_along = f"E{invariant}" in group
        out["tm" if e_along else "te"] = {
            "components": group,
            "matrix": _submatrix(M, group, group),
        }
    return out


def difference(field, axis, target, h, sym):
    """The finite difference of `field` along `axis`, evaluated at `target`'s
    Yee location, built by sympy's own as_finite_difference.

    `sym` maps a relative sample index (0, +1, -1) to the symbol standing for
    that neighbour. The offset between the two components decides which
    samples are used -- forward or backward is derived, never chosen:

        target is half a cell AHEAD of field  -> samples (0, +1), forward
        target is half a cell BEHIND field    -> samples (-1, 0), backward
    """
    ai = AXES.index(axis)
    delta = YEE[field][ai] - YEE[target][ai]
    v = sp.Symbol("v")
    f = sp.Function("f")
    if delta == -HALF:
        points, bind = [v, v + h], {f(v): sym(0), f(v + h): sym(+1)}
    elif delta == +HALF:
        points, bind = [v - h, v], {f(v): sym(0), f(v - h): sym(-1)}
    else:
        raise ValueError(
            f"d({field})/d{axis} at {target}: offset {delta} is not +/-1/2, so "
            "these two components are not Yee-staggered along this axis")
    return sp.Derivative(f(v), v).as_finite_difference(points).subs(bind)


def pml_split(diff_expr, psi, b, a, k):
    """The stretched-coordinate PML term, derived from 1/s_u.

    With s_u = k_u + sigma_u/(a_u + i w), the operator (1/s_u) d/du separates
    into an unstretched part scaled by 1/k_u plus a convolution term. In the
    time domain that convolution is the recursive-convolution memory psi, and
    the standard CPML discretisation of it is a one-pole recursion:

        psi^{n+1} = b*psi^n + a*(d/du)^{n+1}
        (1/s) d/du  ->  (d/du)/k + psi^{n+1}

    Returns (psi_new, stretched_derivative). b, a and k are supplied by the
    caller rather than derived: they are meep's own discrete PML coefficients,
    so the result matches meep's recursion exactly instead of approximately."""
    psi_new = b * psi + a * diff_expr
    return psi_new, diff_expr / k + psi_new


def describe():
    """Print the derivation: full system, reduced system, and the two blocks."""
    print("3D system  d(F)/dt = M F,  F =", COMPONENTS)
    sp.pprint(system_matrix())
    print("\nafter d/dz = 0:")
    sp.pprint(system_matrix(("z",)))
    for name, blk in sorted(polarisation_blocks().items()):
        print(f"\n{name.upper()} block {blk['components']}:")
        sp.pprint(blk["matrix"])


if __name__ == "__main__":
    describe()
