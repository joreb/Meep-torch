# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""C++ binding parity (math/geometry group): vec/ivec/volume/grid_volume/
symmetry/material_function vs hand values from meep/src/meep/vec.hpp semantics."""
import math

import pytest

import meep_torch.mp as mp
from meep_torch.mp.cpp_bindings_geom import (
    D1, D2, D3, Dcyl, geom_box, geom_epsilon, grid_volume, identity, ivec,
    material_function, mirror, r_to_minus_r_symmetry, rotate2, rotate4,
    signed_direction, simple_material_function, symm_matrix, symmetry, vec,
    vol2d, volume, volume_list, zero_ivec, zero_vec,
)

X, Y, Z, R, P = 0, 1, 2, 3, 4


# ---------------------------------------------------------------- vec / ivec
def test_vec_construction_dims():
    assert vec(2.5).dim == D1 and vec(2.5).z() == 2.5
    v = vec(1.0, 2.0)
    assert v.dim == D2 and (v.x(), v.y()) == (1.0, 2.0)
    v3 = vec(1.0, 2.0, 3.0)
    assert v3.dim == D3 and v3.z() == 3.0


def test_vec_arithmetic():
    assert vec(1, 2) + vec(3, 4) == vec(4, 6)
    assert vec(3, 4) - vec(1, 2) == vec(2, 2)
    assert -vec(1, -2) == vec(-1, 2)
    assert vec(1, 2) * 2.0 == vec(2, 4)
    assert vec(2, 4) / 2.0 == vec(1, 2)
    assert vec(1, 2) != vec(1, 3)


def test_vec_direction_access():
    v = vec(1.0, 2.0)
    assert v.in_direction(Y) == 2.0
    v.set_direction(X, 7.0)
    assert v.x() == 7.0


def test_vec_vector3_roundtrip():
    v = vec(1.0, 2.0, 3.0)
    v3 = v.to_vector3()
    assert (v3.x, v3.y, v3.z) == (1.0, 2.0, 3.0)
    assert vec.from_vector3(v3) == v


def test_ivec_arithmetic_and_compare():
    assert ivec(1, 2) + ivec(3, 4) == ivec(4, 6)
    assert ivec(1, 2) + 1 == ivec(2, 3)
    assert ivec(3, 4) - ivec(1, 2) == ivec(2, 2)
    assert ivec(2, 3) * 2 == ivec(4, 6)
    assert ivec(2, 3) * ivec(2, 2) == ivec(4, 6)  # element-wise
    assert ivec(4, 6) / 2 == ivec(2, 3)
    assert ivec(1, 2) <= ivec(1, 3) and ivec(1, 2) < ivec(2, 3)
    assert not (ivec(1, 2) < ivec(1, 3))  # strict in every direction


def test_ivec_round_down_to_even():
    r = ivec(3, -3).round_down_to_even()
    assert (r.x(), r.y()) == (2, -4)


def test_ivec_times_float_is_vec():
    v = ivec(4, 6) * 0.5
    assert isinstance(v, vec) and v == vec(2.0, 3.0)


def test_ivec_yucky_val():
    # D2: n = 0,1,2 -> Z, X, Y (vec.cpp yucky_dir); Z absent in D2 -> 0
    iv = ivec(5, 7)
    assert iv.yucky_val(0) == 0
    assert iv.yucky_val(1) == 5
    assert iv.yucky_val(2) == 7


# ------------------------------------------------------------------- volume
def test_volume_basics():
    v = volume(vec(0, 0), vec(2, 1))
    assert v.center() == vec(1.0, 0.5)
    assert v.get_min_corner() == vec(0, 0) and v.get_max_corner() == vec(2, 1)
    assert v.in_direction(X) == 2.0 and v.in_direction_min(Y) == 0.0
    assert v.computational_volume() == pytest.approx(2.0)
    assert v.full_volume() == pytest.approx(2.0)
    assert v.diameter() == pytest.approx(2.0)


def test_volume_corner_order_normalized():
    v = volume(vec(2, 1), vec(0, 0))  # meep sorts corners
    assert v.get_min_corner() == vec(0, 0)


def test_volume_contains():
    v = volume(vec(0, 0), vec(2, 1))
    assert v.contains(vec(1, 0.5)) and not v.contains(vec(3, 0.5))
    assert v.contains(volume(vec(0.5, 0.2), vec(1.5, 0.8)))
    assert not v.contains(volume(vec(0.5, 0.2), vec(2.5, 0.8)))


def test_volume_intersection():
    a = volume(vec(0, 0), vec(2, 1))
    b = volume(vec(1, -1), vec(3, 3))
    i = a.intersect_with(b)
    assert i.get_min_corner() == vec(1, 0) and i.get_max_corner() == vec(2, 1)
    assert a.intersects(b)
    c = volume(vec(5, 5), vec(6, 6))
    assert not a.intersects(c)
    assert a.intersect_with(c).computational_volume() == 0.0


def test_volume_normal_direction():
    assert volume(vec(0, 0), vec(0, 2)).normal_direction() == X
    assert volume(vec(0, 0), vec(2, 0)).normal_direction() == Y
    assert volume(vec(0, 0, 0), vec(1, 1, 0)).normal_direction() == Z


def test_volume_integral_volume_skips_zero_extent():
    v = volume(vec(0, 0), vec(0, 2))  # line: X extent zero
    assert v.integral_volume() == pytest.approx(2.0)


# -------------------------------------------------------------- grid_volume
def _gv():
    gv = vol2d(8, 4, 10)
    gv.center_origin()
    return gv


def test_grid_volume_size():
    gv = _gv()
    assert (gv.nx(), gv.ny(), gv.nz()) == (80, 40, 0)
    assert gv.ntot() == 81 * 41
    assert gv.a == 10 and gv.inva == pytest.approx(0.1)


def test_grid_volume_center_and_corners():
    gv = _gv()
    assert gv.center() == vec(0.0, 0.0)
    assert gv.little_corner() == ivec(-80, -40)
    assert gv.big_corner() == ivec(80, 40)
    s = gv.surroundings()
    assert s.get_min_corner() == vec(-4.0, -2.0)
    assert s.get_max_corner() == vec(4.0, 2.0)


def test_grid_volume_contains():
    gv = _gv()
    assert gv.contains(vec(3.99, 1.99))
    assert not gv.contains(vec(4.2, 0.0))
    assert gv.contains(ivec(0, 0)) and not gv.contains(ivec(200, 0))


def test_grid_volume_index_iloc_roundtrip():
    gv = _gv()
    Ez = 4
    p = ivec(6, 10)  # even (centered-grid point for Ez in 2D: yee shift 0)
    idx = gv.index(Ez, p)
    assert gv.iloc(Ez, idx) == p
    lc = gv.loc(Ez, idx)
    assert lc.x() == pytest.approx(0.3) and lc.y() == pytest.approx(0.5)


def test_grid_volume_dV():
    gv = _gv()
    cell = gv.dV(ivec(0, 0))
    assert cell.get_min_corner() == vec(-0.05, -0.05)
    assert cell.get_max_corner() == vec(0.05, 0.05)


def test_grid_volume_chunk_machinery_raises():
    gv = _gv()
    for call in (lambda: gv.get_cost(), lambda: gv.tile_split(),
                 lambda: gv.find_best_split(2, False),
                 lambda: gv.split_at_fraction(True, 40, 0),
                 lambda: gv.subvolume(ivec(0, 0), ivec(2, 2), 4)):
        with pytest.raises(NotImplementedError):
            call()


# ---------------------------------------------------------------- symmetry
def test_mirror_transform_vec():
    s = mirror(X, _gv())
    assert s.multiplicity() == 2
    assert s.transform(vec(1.0, 0.5), 1) == vec(-1.0, 0.5)
    assert s.transform(vec(1.0, 0.5), 0) == vec(1.0, 0.5)
    assert s.transform(ivec(6, 10), 1) == ivec(-6, 10)


def test_mirror_phase_shifts():
    s = mirror(X, _gv())
    Ex_, Ey_, Hz_ = 0, 1, 9
    assert s.phase_shift(Ex_, 1) == -1.0  # E parallel to mirror normal: odd
    assert s.phase_shift(Ey_, 1) == 1.0
    assert s.phase_shift(Hz_, 1) == -1.0  # H is a pseudovector


def test_rotate4_transform():
    gv = vol2d(4, 4, 10)
    gv.center_origin()
    s = rotate4(Z, gv)
    assert s.multiplicity() == 4
    assert s.transform(vec(1.0, 0.5), 1) == vec(0.5, -1.0)  # (x,y) -> (y,-x)
    # four applications = identity
    v = vec(1.0, 0.5)
    for _ in range(4):
        v = s.transform(v, 1)
    assert v == vec(1.0, 0.5)


def test_identity_and_signed_direction():
    s = identity()
    assert s.multiplicity() == 1
    sd = s.transform(X, 1)
    assert isinstance(sd, signed_direction)
    assert sd.d == X and not sd.flipped
    assert signed_direction(Y, True) == signed_direction(Y, True)
    assert signed_direction(Y, True) != signed_direction(Y, False)


def test_r_to_minus_r_phase():
    assert r_to_minus_r_symmetry(1.0).ph == -1.0
    assert r_to_minus_r_symmetry(2.0).ph == 1.0


def test_symmetry_reduce_flux_volumes():
    s = mirror(X, _gv())
    v = volume(vec(-1.0, 0.0), vec(1.0, 0.0))
    Sy_ = 101
    out = s.symmetry_reduce(volume_list(v, Sy_))
    assert out is not None and out.next is None
    assert out.weight == 2.0  # even component doubles on the half-volume
    assert out.v.in_direction(X) == pytest.approx(1.0)  # halved extent
    Sx_ = 100
    assert s.symmetry_reduce(volume_list(v, Sx_)) is None  # odd: cancels


# ------------------------------------------------- material_function family
def test_material_function_defaults():
    m = material_function()
    r = vec(1.0, 2.0)
    assert m.eps(r) == 1.0 and m.mu(r) == 1.0
    assert m.chi1p1(0, r) == 1.0
    assert m.conductivity(0, r) == 0.0
    assert not m.has_mu() and not m.has_chi2(0) and not m.has_chi3(0)
    assert m.sigma_row(0, r) == [0.0, 0.0, 0.0]


def test_simple_material_function_delegates():
    f = simple_material_function(lambda r: 12.0 if r.x() < 0 else 1.0)
    assert f.eps(vec(-1.0, 0.0)) == 12.0
    assert f.eps(vec(1.0, 0.0)) == 1.0
    assert f.mu(vec(-1.0, 0.0)) == 12.0
    assert f.sigma_row(0, vec(-1.0, 0.0)) == [12.0, 0.0, 0.0]  # Ex row
    assert f.sigma_row(1, vec(-1.0, 0.0)) == [0.0, 12.0, 0.0]  # Ey row
    assert isinstance(f, material_function)


def test_geom_epsilon_honest_raises():
    ge = geom_epsilon([], [], volume(vec(0, 0), vec(1, 1)))
    assert isinstance(ge, material_function)
    with pytest.raises(NotImplementedError):
        ge.chi1p1(0, vec(0, 0))
    with pytest.raises(NotImplementedError):
        ge.get_material_pt(None, vec(0, 0))
    ge.set_volume(volume(vec(0, 0), vec(1, 1)))  # restriction volume: real
    ge.unset_volume()


def test_plain_data_classes():
    sm = symm_matrix()
    assert (sm.m00, sm.m01, sm.m02, sm.m11, sm.m12, sm.m22) == (0,) * 6
    gb = geom_box()
    assert gb.low is not None and gb.high is not None
    vl = volume_list(volume(vec(0, 0), vec(1, 1)), 4)
    assert vl.c == 4 and vl.weight == 1.0 and vl.next is None


def test_module_exports():
    for name in ("vec", "ivec", "volume", "grid_volume", "volume_list",
                 "symmetry", "symm_matrix", "signed_direction", "geom_box",
                 "geom_epsilon", "material_function", "simple_material_function",
                 "vol1d", "vol2d", "vol3d", "volcyl", "identity", "mirror",
                 "rotate2", "rotate4", "zero_vec", "one_ivec"):
        assert hasattr(mp, name), name
