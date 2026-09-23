# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""State shapes are derivable from (Yee offset, per-axis boundary).

Twelve steppers allocate their state by hand, and the shapes are not choices:
H lives on the gaps between E nodes, so a half-offset component is short by one
along an axis that has a WALL, and keeps the full extent along one that WRAPS.
`core.symbolic.field_shape` derives that; this checks it against what the
steppers really allocate, which is the only way to learn whether the rule holds.

The periodic half is the part that matters and the part a Yee table alone does
not give. A rule read off the PEC steppers reproduces them and then contradicts
every Bloch stepper: `stepper_bloch` has `Hx (nx, ny)` where the PEC rule says
`(nx, ny-1)`, because its y axis wraps.

Nothing calls `field_shape` in production yet. It is the first piece of a
spec-driven stepper, and it is gated before it is used rather than after.
"""
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.symbolic import (PERIODIC, WALL, cut_plane_row,
                                     field_shape, psi_extent)

NX, NY = 12, 10


def _shapes(state, names):
    return {n: tuple(getattr(state, n).shape) for n in names}


def test_tm_shapes_are_derived():
    """PEC walls on both axes: the case the naive rule also gets right."""
    from meep_torch.core.stepper_tm import init_tm_state

    s = init_tm_state(Grid(nx=NX, ny=NY, dx=0.05))
    got = _shapes(s, ("Ez", "Hx", "Hy"))
    for c, actual in got.items():
        assert field_shape(c, (NX, NY), (WALL, WALL)) == actual, c


def test_te_shapes_are_derived():
    from meep_torch.core.stepper_te import init_te_state

    s = init_te_state(Grid(nx=NX, ny=NY, dx=0.05))
    for c, actual in _shapes(s, ("Hz", "Ex", "Ey")).items():
        assert field_shape(c, (NX, NY), (WALL, WALL)) == actual, c


def test_bloch_shapes_need_the_periodic_rule():
    """The discriminating case, and the reason the rule takes boundaries.

    stepper_bloch wraps in y. Hx is half-offset in y and keeps the FULL extent
    there, while Hy is half-offset in x and is short by one, because x is
    walled. A rule that only knows Yee offsets cannot produce both.
    """
    from meep_torch.core.stepper_bloch import init_bloch_tm_state

    s = init_bloch_tm_state(Grid(nx=NX, ny=NY, dx=0.05))
    bc = (WALL, PERIODIC)
    for field, comp in (("Ez_c", "Ez"), ("Hx_c", "Hx"), ("Hy_c", "Hy")):
        actual = tuple(getattr(s, field).shape)
        assert field_shape(comp, (NX, NY), bc) == actual, (field, actual)

    # and the naive PEC rule must NOT reproduce it -- otherwise this test is
    # covered by the two above and proves nothing extra
    assert field_shape("Hx", (NX, NY), (WALL, WALL)) != tuple(s.Hx_c.shape)


@pytest.mark.parametrize("phase,stores", [(-1, True), (+1, False)])
def test_the_cut_plane_row_follows_the_sector(phase, stores):
    """Hx0 exists for an odd y-mirror and not an even one, derived.

    The on-plane node is a fixed point of the reflection, so the sector relation
    reads f = phase * M_cc * f. H is axial, so M_Hx = -1 about the y cut, and
    phase = +1 leaves only f = 0 -- nothing to store. The stepper already
    agreed; now the rule says so instead of the code.
    """
    from meep_torch.core.stepper_tm import init_tm_state

    s = init_tm_state(Grid(nx=NX, ny=NY, dx=0.05), y_mirror=phase)
    assert cut_plane_row("Hx", "y", phase) is stores
    assert (s.Hx0 is not None) is stores


def test_the_rule_is_not_vacuous():
    """A derivation that agrees with everything proves nothing.

    Feed it the wrong boundary and the shapes must change: if walled and
    periodic gave the same answer, the tests above would pass for any rule.
    """
    walled = field_shape("Hx", (NX, NY), (WALL, WALL))
    wrapped = field_shape("Hx", (NX, NY), (WALL, PERIODIC))
    assert walled != wrapped, "the boundary kind does not affect the shape"
    assert walled == (NX, NY - 1) and wrapped == (NX, NY)


def test_this_package_has_one_te_lattice():
    """Both TE steppers are on Yee now, and the dual origin still exists.

    `stepper_bloch`'s TE half used to put Hz on the integer node -- Yee shifted
    by (1/2, 1/2, 0) -- inherited by transcribing step_bloch_tm position for
    position, so the same problem gave different-shaped output depending on
    which TE stepper ran it. That was found by the allocator rule failing on
    bloch TE, which is the useful way to find a convention split that no
    document recorded and nothing in the code named.

    The split is closed; `LATTICE_DUAL` stays because the rule has to be able
    to express it, and the last assertion keeps the two origins genuinely
    different so a future stepper on the dual lattice is still describable.
    """
    from meep_torch.core.stepper_bloch import init_bloch_te_state
    from meep_torch.core.stepper_te import init_te_state
    from meep_torch.core.symbolic import LATTICE_DUAL, LATTICE_YEE

    g = Grid(nx=NX, ny=NY, dx=0.05)

    yee = init_te_state(g)
    for c in ("Hz", "Ex", "Ey"):
        assert field_shape(c, (NX, NY), (WALL, WALL),
                           origin=LATTICE_YEE) == tuple(getattr(yee, c).shape), c

    bloch = init_bloch_te_state(g)
    for field, c in (("Hz_c", "Hz"), ("Ex_c", "Ex"), ("Ey_c", "Ey")):
        assert field_shape(c, (NX, NY), (WALL, PERIODIC),
                           origin=LATTICE_YEE) == tuple(getattr(bloch, field).shape), c

    # the two origins must still disagree, or LATTICE_DUAL means nothing
    assert (field_shape("Hz", (NX, NY), (WALL, WALL), origin=LATTICE_YEE)
            != field_shape("Hz", (NX, NY), (WALL, WALL), origin=LATTICE_DUAL))


def test_the_bloch_te_psi_follows_the_unified_lattice():
    """The two psi arrays swapped extents when the lattice did, and so did the
    CPML profiles they use.

    On the dual lattice the Ey update's memory took `b_hx` -- the H-family
    profile -- because dHz/dx landed at the half offset. On Yee it lands on
    Ey's pinned interior and takes `b_ex`. That the names now match what they
    are is the check that the conversion went the right way rather than merely
    producing consistent shapes.
    """
    from meep_torch.core.stepper_bloch import init_bloch_te_state

    g = Grid(nx=NX, ny=NY, dx=0.05)
    st = init_bloch_te_state(g)
    b = (WALL, PERIODIC)
    assert tuple(st.psi_Eyx_c.shape) == psi_extent("Hz", (NX, NY), b)
    assert tuple(st.psi_Hzx_c.shape) == psi_extent("Ey", (NX, NY), b)
    assert st.psi_Eyx_c.shape != st.psi_Hzx_c.shape, (
        "the two psi must differ in x, or the lattice is not staggered there")


_PSI_TARGET = {"psi_Ezy": "Hx", "psi_Ezx": "Hy", "psi_Hyx": "Ez", "psi_Hxy": "Ez",
               "psi_Eyx": "Hz", "psi_Exy": "Hz", "psi_Hzy": "Ex", "psi_Hzx": "Ey"}


@pytest.mark.parametrize("stepper", ["tm", "te"])
def test_psi_extents_are_derived(stepper):
    """CPML memory lives on its target's grid, minus the rows never written.

    A PEC wall pins TANGENTIAL E, so an E component with a node on that wall is
    not updated there and needs no memory for it. H is free at a PEC wall and
    keeps its full extent. Both facts are needed: a rule using only the offset
    predicts psi_Ezy at (10, 9) where stepper_tm allocates (12, 9).
    """
    from meep_torch.core.symbolic import psi_extent

    if stepper == "tm":
        from meep_torch.core.stepper_tm import init_tm_state as init
    else:
        from meep_torch.core.stepper_te import init_te_state as init
    s = init(Grid(nx=NX, ny=NY, dx=0.05))

    checked = 0
    for name in s._fields:
        v = getattr(s, name)
        if not (name.startswith("psi") and isinstance(v, torch.Tensor)):
            continue
        target = _PSI_TARGET[name]
        assert psi_extent(target, (NX, NY), (WALL, WALL)) == tuple(v.shape), (
            f"{name} targets {target}: derived "
            f"{psi_extent(target, (NX, NY), (WALL, WALL))}, allocated {tuple(v.shape)}")
        checked += 1
    assert checked == 4, f"expected 4 psi arrays, checked {checked}"


def test_the_psi_rule_needs_both_halves():
    """Offset alone, or family alone, each reproduce only part of the table.

    Without this the rule could be carrying a redundant condition and nobody
    would notice: it is the E-vs-H half that the first attempt was missing.
    """
    from meep_torch.core.symbolic import psi_extent

    bc = (WALL, WALL)
    # H targets are NOT trimmed even though Hx has a node on the x wall
    assert psi_extent("Hx", (NX, NY), bc) == field_shape("Hx", (NX, NY), bc)
    # E targets ARE, and only on the axes where they have a node
    assert psi_extent("Ez", (NX, NY), bc) == (NX - 2, NY - 2)
    assert psi_extent("Ex", (NX, NY), bc) == (NX - 1, NY - 2)   # half in x, no node
    # and a periodic axis is never trimmed: no wall, nothing pinned
    assert psi_extent("Ez", (NX, NY), (WALL, PERIODIC)) == (NX - 2, NY)


@pytest.mark.parametrize("n_poles,n_mpoles", [(0, 0), (2, 0), (0, 3), (2, 3)])
def test_tm_companions_are_derived(n_poles, n_mpoles):
    """D/P per E component, B/M per H component when magnetic poles exist.

    The asymmetry is the content: D and the pole pair are allocated even with
    zero poles (the pole axis is length zero), while B and M appear only with
    magnetic poles, because without magnetic dispersion B = mu H folds into the
    coefficients and B is not a separate unknown.
    """
    from meep_torch.core.stepper_tm import init_tm_state
    from meep_torch.core.symbolic import companion_arrays

    s = init_tm_state(Grid(nx=NX, ny=NY, dx=0.05),
                      n_poles=n_poles, n_mpoles=n_mpoles)
    want = companion_arrays(("Ez", "Hx", "Hy"), (NX, NY), (WALL, WALL),
                            n_poles, n_mpoles)
    # a one-E-component block drops the suffix: Dz, P_cur, P_prev
    assert want[("D", "Ez")] == tuple(s.Dz.shape)
    assert want[("P_cur", "Ez")] == tuple(s.P_cur.shape)
    assert want[("P_prev", "Ez")] == tuple(s.P_prev.shape)

    for c, b, mc in (("Hx", "Bx", "Mx_cur"), ("Hy", "By", "My_cur")):
        if n_mpoles:
            assert want[("B", c)] == tuple(getattr(s, b).shape)
            assert want[("M_cur", c)] == tuple(getattr(s, mc).shape)
        else:
            assert ("B", c) not in want
            assert getattr(s, b) is None, f"{b} allocated without magnetic poles"


@pytest.mark.parametrize("n_poles", [0, 2])
def test_te_companions_are_derived(n_poles):
    """Two E components, so the suffix is kept: Dx/Dy, Px_*/Py_*."""
    from meep_torch.core.stepper_te import init_te_state
    from meep_torch.core.symbolic import companion_arrays

    s = init_te_state(Grid(nx=NX, ny=NY, dx=0.05), n_poles=n_poles)
    want = companion_arrays(("Hz", "Ex", "Ey"), (NX, NY), (WALL, WALL), n_poles)
    for c, d, pc, pp in (("Ex", "Dx", "Px_cur", "Px_prev"),
                         ("Ey", "Dy", "Py_cur", "Py_prev")):
        assert want[("D", c)] == tuple(getattr(s, d).shape), d
        assert want[("P_cur", c)] == tuple(getattr(s, pc).shape), pc
        assert want[("P_prev", c)] == tuple(getattr(s, pp).shape), pp
    # Hz carries no companions at all without magnetic poles
    assert not any(k[1] == "Hz" for k in want)


# --- the spec: every array a block allocates, from one object ---------------

def _allocated(state):
    return {n: tuple(getattr(state, n).shape) for n in state._fields
            if isinstance(getattr(state, n), torch.Tensor)}


_TM = ("Ez", "Hx", "Hy")
_TM_CASES = [
    ("plain", {}, dict(boundaries=(WALL, WALL))),
    ("poles", dict(n_poles=2), dict(boundaries=(WALL, WALL), n_poles=2)),
    ("mpoles", dict(n_mpoles=3), dict(boundaries=(WALL, WALL), n_mpoles=3)),
    ("both", dict(n_poles=2, n_mpoles=3),
     dict(boundaries=(WALL, WALL), n_poles=2, n_mpoles=3)),
    ("y_mirror_odd", dict(y_mirror=-1),
     dict(boundaries=(WALL, "fold"), folds=(("y", -1),))),
    ("y_mirror_even", dict(y_mirror=+1),
     dict(boundaries=(WALL, "fold"), folds=(("y", +1),))),
    ("x_mirror_odd", dict(x_mirror=-1),
     dict(boundaries=("fold", WALL), folds=(("x", -1),))),
]


@pytest.mark.parametrize("label,init_kw,spec_kw", _TM_CASES,
                         ids=[c[0] for c in _TM_CASES])
def test_the_spec_reproduces_the_tm_allocation(label, init_kw, spec_kw):
    """Every array stepper_tm allocates, derived from one BlockSpec.

    Names and shapes both, across materials and folds. The fold cases are the
    ones that found a missing rule: a cut replaces ONE of the two PEC walls, so
    psi loses one row there and not two, and `init_tm_state` allocates ny-1
    where a two-wall axis gives ny-2.
    """
    from meep_torch.core.symbolic import BlockSpec
    from meep_torch.core.stepper_tm import init_tm_state

    spec = BlockSpec(_TM, (NX, NY), **spec_kw)
    assert spec.state_shapes() == _allocated(init_tm_state(
        Grid(nx=NX, ny=NY, dx=0.05), **init_kw))


@pytest.mark.parametrize("n_poles", [0, 2])
def test_the_spec_reproduces_the_te_allocation(n_poles):
    from meep_torch.core.symbolic import BlockSpec
    from meep_torch.core.stepper_te import init_te_state

    spec = BlockSpec(("Hz", "Ex", "Ey"), (NX, NY), (WALL, WALL), n_poles=n_poles)
    assert spec.state_shapes() == _allocated(init_te_state(
        Grid(nx=NX, ny=NY, dx=0.05), n_poles=n_poles))


def test_the_spec_can_be_wrong():
    """The comparison above is only evidence if a wrong spec fails it.

    Two blocks that differ only in a boundary must allocate differently; if
    they did not, every case above would pass for any spec at all.
    """
    from meep_torch.core.symbolic import BlockSpec

    walled = BlockSpec(_TM, (NX, NY), (WALL, WALL)).state_shapes()
    wrapped = BlockSpec(_TM, (NX, NY), (WALL, PERIODIC)).state_shapes()
    assert walled != wrapped
    assert walled["Hx"] != wrapped["Hx"]      # the fencepost
    assert walled["psi_Hyx"] != wrapped["psi_Hyx"]   # and the psi trim

    # a fold trims one row where a wall trims two -- the rule the fold cases found
    folded = BlockSpec(_TM, (NX, NY), (WALL, "fold"),
                       folds=(("y", -1),)).state_shapes()
    assert folded["psi_Hyx"][1] == walled["psi_Hyx"][1] + 1


_PSI3D_CASES = [
    ("pec", (12, 11, 10), ()),
    ("pmc_xy", (9, 9, 12), (0, 1)),
    ("pmc_z", (10, 9, 11), (2,)),
]


@pytest.mark.parametrize("label,dims,pmc", _PSI3D_CASES,
                         ids=[c[0] for c in _PSI3D_CASES])
def test_3d_psi_is_allocated_at_what_it_writes(label, dims, pmc):
    """A step must not change the shape of its own CPML memory.

    The six E-update memories used to carry the raw derivative extent, two
    slabs of which the interior trim discarded before they reached a curl.
    `_psi_e_shapes` allocates the written extent instead, which makes the step
    shape-preserving -- and that is the property worth asserting, because a
    stepper whose psi grows or shrinks under iteration breaks the checkpoint
    plumbing long before it breaks the physics.

    The pmc cases are the ones that need the boundary: a PMC wall keeps the
    tangential E alive, so that axis is not trimmed and the memory is wider.
    """
    from meep_torch.core.stepper_3d import (_psi_e_shapes, build_3d_coeffs,
                                            init_3d_state, step_3d)

    nx, ny, nz = dims
    c = build_3d_coeffs(nx, ny, nz, dx=0.05, npml=3, pmc_axes=pmc)
    st = init_3d_state(nx, ny, nz, npml=3, pmc=pmc)

    before = {k: tuple(getattr(st, k).shape)
              for k in st._fields if k.startswith("psi_")}
    assert {k: v for k, v in before.items() if k.startswith("psi_H")} \
        == _psi_e_shapes(nx, ny, nz, pmc)

    out = step_3d(st, c, 1.0, ("Ez", nx // 2, ny // 2, nz // 2))
    after = {k: tuple(getattr(out, k).shape)
             for k in out._fields if k.startswith("psi_")}
    assert after == before, "the step reshaped its own psi"


def test_the_3d_psi_rule_is_not_vacuous():
    """The shapes above must actually depend on the trim and on the walls.

    Without this the test would pass against the old untrimmed allocation --
    which it has to fail, since that allocation is what the change removed.
    """
    from meep_torch.core.stepper_3d import _psi_e_shapes

    nx, ny, nz = 12, 11, 10
    pec = _psi_e_shapes(nx, ny, nz)
    untrimmed = {"psi_Hz_y": (nx - 1, ny - 2, nz), "psi_Hy_z": (nx - 1, ny, nz - 2),
                 "psi_Hx_z": (nx, ny - 1, nz - 2), "psi_Hz_x": (nx - 2, ny - 1, nz),
                 "psi_Hy_x": (nx - 2, ny, nz - 1), "psi_Hx_y": (nx, ny - 2, nz - 1)}
    assert pec != untrimmed
    assert all(pec[k] != untrimmed[k] for k in pec)

    # under PEC the two terms of one curl meet on a common region; a PMC wall
    # on one of the trimmed axes breaks that and widens only the term that
    # owns it.
    assert pec["psi_Hz_y"] == pec["psi_Hy_z"]
    pmc = _psi_e_shapes(nx, ny, nz, (1,))
    assert pmc["psi_Hy_z"] != pec["psi_Hy_z"]
    assert pmc["psi_Hz_y"] == pec["psi_Hz_y"]


def test_the_old_3d_psi_allocation_is_rejected():
    """Handing the step the untrimmed memory must fail loudly.

    This is what makes the shapes above evidence: the previous allocation is a
    concrete wrong input, and the step refuses it rather than broadcasting.
    Broadcasting is the real hazard -- a transverse-uniform test on a 3-cell
    axis has `ny - 2 == 1`, which expands silently and agrees for exactly the
    fields such a test drives.
    """
    from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                            step_3d)

    nx, ny, nz = 12, 11, 10
    c = build_3d_coeffs(nx, ny, nz, dx=0.05, npml=3)
    st = init_3d_state(nx, ny, nz, npml=3)
    untrimmed = st._replace(
        psi_Hz_y=torch.zeros(nx - 1, ny - 2, nz, dtype=torch.float64),
        psi_Hy_z=torch.zeros(nx - 1, ny, nz - 2, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="must match the size"):
        step_3d(untrimmed, c, 1.0, ("Ez", nx // 2, ny // 2, nz // 2))


def test_3d_psi_matches_psi_extent():
    """The 3D stepper and the derived rule must agree, or there are two rules.

    `psi_extent` was written against the 2D steppers and recorded `stepper_3d`
    as a known divergence -- it carried the untrimmed derivative extent and
    threw two slabs away inside the curl. Moving that trim ahead of the psi
    recursion closed the gap bitwise. This is the gate that keeps it closed.

    PMC is excluded on purpose: a PMC wall makes the memory per-term rather
    than per-component (one curl term is full-length at the wall, the other is
    a derivative along it and gets padded out afterwards), which a
    component-shaped rule cannot express. See `psi_extent`.
    """
    from meep_torch.core.stepper_3d import _psi_e_shapes
    from meep_torch.core.symbolic import psi_extent

    dims = (12, 11, 10)
    target = {"psi_Hz_y": "Ex", "psi_Hy_z": "Ex", "psi_Hx_z": "Ey",
              "psi_Hz_x": "Ey", "psi_Hy_x": "Ez", "psi_Hx_y": "Ez"}
    got = _psi_e_shapes(*dims)
    for name, comp in target.items():
        assert got[name] == psi_extent(comp, dims, (WALL, WALL, WALL)), name

    # both terms of one curl land on their component's region -- that is what
    # makes a single per-component rule possible at all
    assert got["psi_Hz_y"] == got["psi_Hy_z"]
    assert got["psi_Hx_z"] == got["psi_Hz_x"]
    assert got["psi_Hy_x"] == got["psi_Hx_y"]


# Field sums after 30 steps. Hex floats, so the comparison is exact rather
# than "close". Originally captured at 7df3716, BEFORE the E-update psi arrays
# were trimmed to what step_3d writes -- that is the property this gate exists
# for and it has not moved.
#
# REBASELINED THREE TIMES, each deliberately. Third (2026-09-20): the E-side
# curls joined the same composition, using a kappa padded with ONE where the
# derivative is padded with zero, so the padded region still contributes
# 0/1 + 0 = 0. Worst change across both 3D edits, against the numbers before
# either: 6.493e-16 relative to each case's own field scale.
#
# Second (2026-09-20): the H curls moved
# onto `compiled_curl_composition`, the grouping stepper_tm and stepper_te
# already used -- this file had `(d0/k0 + psi0) - (d1/k1 + psi1)` where they
# have `((d0/k0 + psi0) - d1/k1) - psi1`. Same algebra, different association.
# Worst change after 30 steps: 8.658e-16 relative to each case's own field
# scale, which is what an association change should cost and is the evidence
# that nothing structural moved with it.
#
# First, when the point source was corrected: it had
# been injecting with the wrong SIGN (meep subtracts, `dD/dt = curl H - J`) and
# without its CELL MEASURE (a point is a delta in three directions, so `1/dx^3`).
# Both show up here rather than hiding: every sum flipped sign and grew by
# ~8000, which is `1/dx^3` at `dx = 0.05` to the digit. A pin that moves for a
# reason you can read off the numbers is the pin working.
_PSI3D_BITWISE = [
    ("plain", dict(pml_alpha=0.05, nx=16, ny=15, nz=14),
     {"Ex": "0x1.ec78a0b1b8818p+2", "Ey": "0x1.0000000000000p-48",
      "Ez": "0x1.dbf22c366e6bcp+8", "Hx": "-0x1.6000000000000p-46",
      "Hy": "-0x1.3a7d87fb01dbap+5", "Hz": "0x1.8000000000000p-97"}),
    ("pmc01", dict(pml_alpha=0.05, nx=10, ny=10, nz=18, pmc=(0, 1)),
     {"Ex": "-0x1.8000000000000p-47", "Ey": "-0x1.0000000000000p-45",
      "Ez": "0x1.63f9a60b4064ep+9", "Hx": "-0x1.0000000000000p-47",
      "Hy": "-0x1.8000000000000p-47", "Hz": "-0x1.b377affa4b1f4p-52"}),
    ("poles", dict(pml_alpha=0.05, nx=14, ny=13, nz=12, n_poles=1),
     {"Ex": "0x1.43e7064fb26e2p+2", "Ey": "0x1.c000000000000p-49",
      "Ez": "0x1.06a1304d3e174p+7", "Hx": "0x1.0000000000000p-46",
      "Hy": "0x1.4235cac66e32cp+6", "Hz": "0x1.4000000000000p-99"}),
    ("mixed", dict(pml_alpha=0.05, nx=12, ny=12, nz=16, n_abs=4, absorber_axes=(2,)),
     {"Ex": "0x1.1d6847a5f07ddp+3", "Ey": "0x1.1d6847a5f07e2p+3",
      "Ez": "-0x1.95ca2a28f0820p+5", "Hx": "-0x1.c4d43f5adcaffp+6",
      "Hy": "0x1.c4d43f5adcb02p+6", "Hz": "0x1.c31055812b23ep-49"}),
]


@pytest.mark.parametrize("label,kw,want", _PSI3D_BITWISE,
                         ids=[c[0] for c in _PSI3D_BITWISE])
@pytest.mark.usefixtures("cpu_rng_pinned")
def test_3d_psi_trim_changed_no_number(label, kw, want):
    """Trimming the psi arrays must not move a single bit.

    The recursion `psi' = b psi + a d` is pointwise, so slicing `d` before it
    and slicing the result after it give the same numbers -- but only because
    the axis being trimmed is one the coefficients are uniform along. `b`, `a`
    and `k` vary along the DERIVATIVE axis; the interior trim is on the other
    axis of the pair. Get that wrong and the recursion meets the wrong
    coefficient row, which is the off-by-one this project has now hit three
    times in stencil code.

    So the claim is exact equality against numbers recorded before the change,
    not a tolerance. The four cases cover every branch the E-phase psi reaches:
    plain CPML, PMC walls (where a pmc axis is NOT trimmed), ADE poles, and the
    mixed absorber (identity CPML profile on the absorber axis).
    """
    import math
    from meep_torch.core.stepper_3d import (build_3d_coeffs, init_3d_state,
                                            step_3d)

    kw = dict(kw)
    nx, ny, nz = kw.pop("nx"), kw.pop("ny"), kw.pop("nz")
    pmc, n_poles = kw.pop("pmc", ()), kw.pop("n_poles", 0)
    c = build_3d_coeffs(nx, ny, nz, dx=0.05, npml=4, pmc_axes=pmc,
                        poles=((1.2, 0.0, 0.3),) if n_poles else (), **kw)
    st = init_3d_state(nx, ny, nz, npml=4, n_poles=n_poles, pmc=pmc)
    src = ("Ez", nx // 2, ny // 2, nz // 2)
    for n in range(30):
        amp = math.sin(0.3 * n) * math.exp(-((n - 10) / 6.0) ** 2)
        st = step_3d(st, c, amp, src)

    got = {f: float(getattr(st, f).sum()).hex() for f in want}
    assert got == want
    assert float(st.Ez.abs().max()) > 1e-6, "a dead field would match anything"


# ---------------------------------------------------------------------------
# The allocator rule is frame-independent: it is about walls and half offsets,
# not about x and y. stepper_bor is the test of that, because it was written by
# hand from the physics and never saw this rule.
# ---------------------------------------------------------------------------

_BOR_SHAPES = {                       # meep_torch/core/stepper_bor.py:39-45
    "Er": (-1, 0), "Etheta": (0, 0), "Ez": (0, -1),
    "Hr": (0, -1), "Htheta": (-1, -1), "Hz": (-1, 0),
}


def test_the_rule_predicts_stepper_bor_allocations():
    """Six shapes, from the cylindrical Yee table and the same wall rule.

    `init_bor_state` allocates `Er (nr-1, nz)`, `Ephi (nr, nz)`,
    `Ez (nr, nz-1)`, `Hr (nr, nz-1)`, `Hphi (nr-1, nz-1)`, `Hz (nr-1, nz)`,
    chosen from the physics of a body of revolution months before
    `field_shape` existed. If the rule reproduces them, "short by one on a
    walled axis IFF half-offset" is not a cartesian fact.

    `axes=("r", "z")` is the load-bearing argument: a BOR reduces THETA, the
    MIDDLE axis, so its two extents are frame indices 0 and 2. Positional
    slicing of the offset tuple would silently take r and theta.
    """
    from meep_torch.core.stepper_bor import init_bor_state

    nr, nz = 12, 10
    st = init_bor_state(nr, nz)
    actual = {"Er": st.Er, "Etheta": st.Ephi, "Ez": st.Ez,
              "Hr": st.Hr, "Htheta": st.Hphi, "Hz": st.Hz}
    for name, tensor in actual.items():
        got = field_shape(name, (nr, nz), (WALL, WALL),
                          system="cylindrical", axes=("r", "z"))
        assert got == tuple(tensor.shape), f"{name}: {got} vs {tuple(tensor.shape)}"


def test_naming_the_axes_matters():
    """Reducing theta is not the same as reducing z, and the rule must show it.

    Without this, `axes` could be ignored and every test above would still
    pass, because `("r", "theta")` and `("r", "z")` agree for four of the six
    components. Hz is one that separates them.
    """
    rz = field_shape("Hz", (12, 10), (WALL, WALL),
                     system="cylindrical", axes=("r", "z"))
    rt = field_shape("Hz", (12, 10), (WALL, WALL),
                     system="cylindrical", axes=("r", "theta"))
    assert rz != rt, "theta and z reductions gave the same shape"


def test_the_cartesian_path_is_untouched():
    """Default arguments must reproduce exactly what the 2D steppers get."""
    from meep_torch.core.grid import Grid
    from meep_torch.core.stepper_tm import init_tm_state

    s = init_tm_state(Grid(nx=NX, ny=NY, dx=0.05))
    for c in ("Ez", "Hx", "Hy"):
        assert field_shape(c, (NX, NY), (WALL, WALL)) == tuple(getattr(s, c).shape)
        assert field_shape(c, (NX, NY), (WALL, WALL), system="cartesian",
                           axes=("x", "y")) == tuple(getattr(s, c).shape)


def test_an_unknown_component_names_the_frame():
    """`Ex` is not a cylindrical component, and the error should say so."""
    with pytest.raises(KeyError, match="cylindrical"):
        field_shape("Ex", (12, 10), (WALL, WALL),
                    system="cylindrical", axes=("r", "z"))


def test_the_pole_name_follows_the_same_suffix_rule_as_the_shapes():
    """One rule, read twice: by `state_shapes` and by the step that sums poles.

    The axis letter appears only when the block has more than one E component,
    so TM carries `P_cur` and TE carries `Px_cur`/`Py_cur`. `make_step` used to
    hard-type `P_cur`, which is right for TM, wrong for TE, and invisible to
    every TM gate -- one sum shared across two components instead of one each.
    """
    from meep_torch.core.symbolic import BlockSpec

    tm = BlockSpec(_TM, (NX, NY), (WALL, WALL), n_poles=1)
    te = BlockSpec(("Hz", "Ex", "Ey"), (NX, NY), (WALL, WALL), n_poles=1)

    assert tm.pole_name("Ez") == "P_cur"
    assert (te.pole_name("Ex"), te.pole_name("Ey")) == ("Px_cur", "Py_cur")
    # and the names the allocator actually files them under agree
    for spec, names in ((tm, ["Ez"]), (te, ["Ex", "Ey"])):
        shapes = spec.state_shapes()
        for c in names:
            assert spec.pole_name(c) in shapes, (c, sorted(shapes))
