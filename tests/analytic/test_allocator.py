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
from meep_torch.core.symbolic import PERIODIC, WALL, cut_plane_row, field_shape

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


def test_this_package_has_two_te_lattices():
    """`stepper_te` uses Yee; `stepper_bloch`'s TE half uses the dual lattice.

    Hz sits at (1/2, 1/2) in one and at the integer node in the other -- Yee
    shifted by (1/2, 1/2, 0). Both are internally consistent and they disagree
    about which fields are short by one. `stepper_bloch2` hides it because every
    axis wraps there, so nothing is short at all.

    This was found by the allocator rule failing on bloch TE, which is the
    useful way to find it: a convention split that no document recorded and that
    nothing in the code names.
    """
    from meep_torch.core.stepper_bloch import init_bloch_te_state
    from meep_torch.core.stepper_te import init_te_state
    from meep_torch.core.symbolic import LATTICE_DUAL, LATTICE_YEE

    g = Grid(nx=NX, ny=NY, dx=0.05)

    yee = init_te_state(g)
    for c in ("Hz", "Ex", "Ey"):
        assert field_shape(c, (NX, NY), (WALL, WALL),
                           origin=LATTICE_YEE) == tuple(getattr(yee, c).shape), c

    dual = init_bloch_te_state(g)
    for field, c in (("Hz_c", "Hz"), ("Ex_c", "Ex"), ("Ey_c", "Ey")):
        assert field_shape(c, (NX, NY), (WALL, PERIODIC),
                           origin=LATTICE_DUAL) == tuple(getattr(dual, field).shape), c

    # the two origins must actually disagree, or this proves nothing
    assert (field_shape("Hz", (NX, NY), (WALL, WALL), origin=LATTICE_YEE)
            != field_shape("Hz", (NX, NY), (WALL, WALL), origin=LATTICE_DUAL))


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
