# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Geometry-class parity gate: new mp shapes rasterize to their analytic area.

Each shape is painted through the standard scene_items -> rasterize_scene path
on a small grid and the covered area (recovered from the eps_tm blend) must be
within 2% of the analytic value. Vector3 ops are checked against numpy.
"""
import math

import numpy as np
import pytest

from meep_torch.core.grid import Grid
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.mp.coords import grid_spec
from meep_torch.mp.geometry import (
    Block, Cone, Cylinder, Ellipsoid, GeometricObject, Prism, Sphere, Wedge,
    build_scene_items,
)
from meep_torch.mp.vec import Vector3

EPS = 12.0
GS = grid_spec(Vector3(8, 8), resolution=20)
GRID = Grid(nx=GS.nx, ny=GS.ny, dx=GS.dx)


def _raster(obj):
    eps_items, _ = build_scene_items([obj], GS)
    return rasterize_scene(eps_items, GRID, background=1.0)


def _area(obj) -> float:
    """Painted area um^2 recovered from the arithmetic eps_tm blend."""
    rm = _raster(obj)
    return float((rm.eps_tm - 1.0).sum().item()) / (EPS - 1.0) * GS.dx ** 2


def _eps_at(rm, x_world: float, y_world: float) -> float:
    i = int((x_world + GS.sx / 2.0) / GS.dx)
    j = int((y_world + GS.sy / 2.0) / GS.dx)
    return float(rm.eps_tm[i, j].item())


# ---------------------------------------------------------------- shapes

def test_sphere_area():
    s = Sphere(1.2, center=Vector3(0.5, -0.5), material=EPS)
    assert _area(s) == pytest.approx(math.pi * 1.2 ** 2, rel=0.02)


def test_ellipsoid_area():
    e = Ellipsoid(size=Vector3(2.0, 1.0, 1.0), center=Vector3(-1, 1), material=EPS)
    assert _area(e) == pytest.approx(math.pi * 1.0 * 0.5, rel=0.02)


def test_cone_trapezoid_area():
    c = Cone(1.0, radius2=0.5, height=2.0, center=Vector3(0, 0),
             material=EPS, axis=Vector3(1, 0, 0))
    assert _area(c) == pytest.approx((1.0 + 0.5) * 2.0, rel=0.02)


def test_cone_sharp_triangle_area():
    c = Cone(1.0, height=2.0, center=Vector3(0, 0), material=EPS,
             axis=Vector3(0, 1, 0))
    assert _area(c) == pytest.approx(1.0 * 2.0, rel=0.02)


def test_cone_out_of_plane_axis_raises():
    c = Cone(1.0, height=2.0, material=EPS)  # default axis (0,0,1)
    with pytest.raises(NotImplementedError):
        c.scene_items(GS)


def test_wedge_quarter_area():
    w = Wedge(1.5, wedge_angle=math.pi / 2, wedge_start=Vector3(0, 1, 0),
              center=Vector3(0, 0), material=EPS)
    assert _area(w) == pytest.approx(math.pi * 1.5 ** 2 / 4.0, rel=0.02)


def test_wedge_full_circle_area():
    w = Wedge(1.0, center=Vector3(0, 0), material=EPS)  # default angle 2*pi
    assert _area(w) == pytest.approx(math.pi, rel=0.02)


def test_block_rotated_45deg():
    b = Block(size=Vector3(2, 1, 1), center=Vector3(), material=EPS,
              e1=Vector3(1, 1, 0), e2=Vector3(-1, 1, 0))
    assert _area(b) == pytest.approx(2.0, rel=0.02)
    rm = _raster(b)
    # inside the rotated rect but outside the axis-aligned one, and vice versa
    assert _eps_at(rm, 0.6, 0.8) == pytest.approx(EPS, rel=0.05)
    assert _eps_at(rm, 0.95, -0.4) == pytest.approx(1.0, rel=0.05)


def test_block_default_basis_unchanged():
    b = Block(size=Vector3(2, 1, 1), center=Vector3(), material=EPS)
    assert _area(b) == pytest.approx(2.0, rel=0.02)


def test_block_out_of_plane_basis_raises():
    b = Block(size=Vector3(1, 1, 1), material=EPS,
              e1=Vector3(0, 0, 1), e2=Vector3(0, 1, 0), e3=Vector3(1, 0, 0))
    with pytest.raises(NotImplementedError):
        b.scene_items(GS)


# ---------------------------------------------------------------- base class

def test_geometric_object_shift():
    s = Sphere(1.0, center=Vector3(1, 0, 0), material=EPS)
    s2 = s.shift(Vector3(0.5, 0.5, 0))
    assert isinstance(s2, Sphere) and s2.radius == 1.0 and s2.material == EPS
    assert s2.center == Vector3(1.5, 0.5, 0)
    assert s.center == Vector3(1, 0, 0)          # original untouched
    b = Block(size=Vector3(1, 1, 1))             # None center -> origin
    assert b.shift(Vector3(0, -1, 0)).center == Vector3(0, -1, 0)


def test_all_shapes_are_geometric_objects():
    for cls in (Block, Cylinder, Prism, Sphere, Ellipsoid, Cone, Wedge):
        assert issubclass(cls, GeometricObject)
    # meep inheritance structure
    assert issubclass(Wedge, Cylinder) and issubclass(Cone, Cylinder)
    assert issubclass(Ellipsoid, Block)


def test_constructor_signatures_match_meep():
    import inspect
    expect = {
        GeometricObject: ["material", "center", "epsilon_func", "label"],
        Sphere: ["radius"],
        Cylinder: ["radius", "axis", "height"],
        Wedge: ["radius", "wedge_angle", "wedge_start"],
        Cone: ["radius", "radius2"],
        Block: ["size", "e1", "e2", "e3"],
        Ellipsoid: [],
        Prism: ["vertices", "height", "axis", "center", "sidewall_angle"],
    }
    for cls, want in expect.items():
        have = [p.name for p in inspect.signature(cls.__init__).parameters.values()
                if p.name not in ("self", "kwargs")]
        have = [p for p in have if p != "n_pts"]     # meep-torch ext, appended
        assert have == want, cls.__name__


def test_epsilon_func_raises():
    with pytest.raises(NotImplementedError):
        Sphere(1.0, epsilon_func=lambda p: 12.0)


def test_prism_recenters_on_center():
    v = [Vector3(-0.5, -0.5), Vector3(0.5, -0.5), Vector3(0.5, 0.5), Vector3(-0.5, 0.5)]
    p = Prism(v, height=math.inf, center=Vector3(1, 0.5), material=EPS)
    assert p.center.close(Vector3(1, 0.5))
    rm = _raster(p)
    assert _eps_at(rm, 1.0, 0.5) == pytest.approx(EPS, rel=0.05)
    assert _eps_at(rm, 0.0, 0.0) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------- Vector3

V1, V2 = Vector3(1.0, 2.0, 3.0), Vector3(-2.0, 0.5, 4.0)
A1, A2 = np.array([1.0, 2.0, 3.0]), np.array([-2.0, 0.5, 4.0])


def _np(v):
    return np.array([v.x, v.y, v.z])


def test_vector3_dot_cross_cdot():
    assert V1.dot(V2) == pytest.approx(float(A1 @ A2))
    assert V1.cdot(V2) == pytest.approx(float(A1 @ A2))  # real components
    assert np.allclose(_np(V1.cross(V2)), np.cross(A1, A2))


def test_vector3_norm_unit():
    assert V1.norm() == pytest.approx(float(np.linalg.norm(A1)))
    assert np.allclose(_np(V1.unit()), A1 / np.linalg.norm(A1))


def test_vector3_div_neg_eq():
    assert np.allclose(_np(V1 / 2.0), A1 / 2.0)
    assert np.allclose(_np(V1 / V2), A1 / A2)
    assert np.allclose(_np(-V1), -A1)
    assert V1 == Vector3(1.0, 2.0, 3.0)
    assert V1 != V2


def test_vector3_rotate_close():
    r = Vector3(1, 0, 0).rotate(Vector3(0, 0, 2), math.pi / 2)
    assert r.close(Vector3(0, 1, 0))
    assert not r.close(Vector3(0, 1, 0.1))
    # arbitrary axis vs Rodrigues via numpy
    axis, th = np.array([1.0, 1.0, 0.5]), 0.7
    u = axis / np.linalg.norm(axis)
    expect = (A1 * math.cos(th) + np.cross(u, A1) * math.sin(th)
              + u * (u @ A1) * (1 - math.cos(th)))
    got = V1.rotate(Vector3(*axis), th)
    assert np.allclose(_np(got), expect)
