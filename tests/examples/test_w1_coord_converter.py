# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Hard tests for the meep center-origin -> meep-torch corner-origin converter.

Validated against the raw port examples/straight-waveguide.py numbers:
cell 16x8, resolution 10 -> dx=0.1, nx=160, ny=80; waveguide y in [-0.5,0.5]
(center 0, size 1) -> rows 35:45; source at x=-7 -> ix=10; PML(1.0) -> npml=10.
"""
import math

from meep_torch.mp.coords import (
    GridSpec, cell_index, grid_spec, pml_npml, rect_corners, span_index_range, to_corner,
)
from meep_torch.mp.vec import Vector3


def test_grid_spec_resolution_to_dx_and_shape():
    gs = grid_spec(Vector3(16, 8, 0), 10)
    assert gs.dx == 0.1
    assert (gs.nx, gs.ny) == (160, 80)
    assert (gs.sx, gs.sy) == (16, 8)


def test_centered_block_maps_to_corner_rows():
    # waveguide Block size (inf,1,inf) center (0,0): y in [-0.5,0.5]
    gs = grid_spec(Vector3(16, 8, 0), 10)
    x0, y0, x1, y1 = rect_corners(Vector3(0, 0, 0), Vector3(math.inf, 1, math.inf), gs)
    assert (x0, x1) == (0.0, 16.0)                 # infinite x -> full cell
    assert (y0, y1) == (3.5, 4.5)                 # [-0.5,0.5] shifted +sy/2=4
    i0, i1 = span_index_range(y0, y1, gs, "y")
    assert (i0, i1) == (35, 45)                    # matches raw eps[:, 35:45]


def test_source_center_to_cell_index():
    gs = grid_spec(Vector3(16, 8, 0), 10)
    # meep source at x=-7 -> corner x=1.0 -> ix=10 (cell center at 10.5*dx=1.05? no:
    # ix=round(1.0/0.1 - 0.5)=round(9.5)=10, matches raw LineSource ix=10)
    assert cell_index(-7.0, gs) == 10
    # source at x=0 (center) -> corner x=8.0 -> ix=round(80-0.5)=80
    assert cell_index(0.0, gs) == 80


def test_pml_thickness_to_npml():
    gs = grid_spec(Vector3(16, 8, 0), 10)
    assert pml_npml(1.0, gs) == 10
    assert pml_npml(2.0, gs) == 20


def test_off_center_object_and_source():
    # bent-waveguide-ish: object centered at (4, -1) size (8, 0.6)
    gs = grid_spec(Vector3(20, 8, 0), 10)
    x0, y0, x1, y1 = rect_corners(Vector3(4, -1, 0), Vector3(8, 0.6, math.inf), gs)
    assert x0 == 10.0 and x1 == 18.0              # 4-4+10, 4+4+10
    assert (y0, y1) == (2.7, 3.3)                 # -1-0.3+4, -1+0.3+4
    i0, i1 = span_index_range(y0, y1, gs, "y")
    assert (i0, i1) == (27, 33)


def test_to_corner_inf_clipped_to_cell():
    gs = grid_spec(Vector3(10, 6, 0), 20)
    x, y = to_corner(Vector3(math.inf, math.inf, 0), gs)
    assert (x, y) == (gs.sx, gs.sy)
    x, y = to_corner(Vector3(-100, 100, 0), gs)
    assert x == 0.0 and y == gs.sy                 # clipped to [0, L]


def test_source_index_outside_grid_clipped_is_caller_error():
    # cell_index does not clip (sources must land inside); validate raw value
    gs = grid_spec(Vector3(16, 8, 0), 10)
    # x just inside left edge: corner 0.2 -> ix=round(2-0.5)=round(1.5)=2
    assert cell_index(-7.8, gs) == 2


def test_higher_resolution_rounding():
    gs = grid_spec(Vector3(10, 10, 0), 50)
    assert gs.dx == 0.02 and (gs.nx, gs.ny) == (500, 500)
    i0, i1 = span_index_range(0.0, 5.0, gs, "x")   # half cell
    assert (i0, i1) == (0, 250)