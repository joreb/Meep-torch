# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import numpy as np
import pytest
import torch

from meep_torch.core.grid import Grid
from meep_torch.core.materials import LorentzPole, Material, SiO2, cSi
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene

G = Grid(nx=40, ny=30, dx=0.1)


def rect(x0, y0, x1, y1):
    return Polygon(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float))


def test_background_only():
    rm = rasterize_scene([], G, background=2.25)
    assert torch.all(rm.eps_tm == 2.25)
    assert torch.all(rm.eps_te_xx == 2.25)
    assert rm.poles == [] and torch.all(rm.chi3 == 0)


def test_later_wins_in_overlap():
    a = rect(0.5, 0.5, 2.5, 2.5)
    b = rect(1.5, 1.5, 3.5, 2.9)   # cell-aligned edges
    rm = rasterize_scene([(a, 4.0), (b, 9.0)], G, background=1.0)
    assert rm.eps_tm[10, 10].item() == pytest.approx(4.0)   # only a
    assert rm.eps_tm[20, 20].item() == pytest.approx(9.0)   # overlap: b wins
    assert rm.eps_tm[32, 20].item() == pytest.approx(9.0)   # only b
    assert rm.eps_tm[2, 2].item() == pytest.approx(1.0)     # background


def test_dispersive_material_poles_and_masks():
    core = rect(1.0, 1.0, 3.0, 2.0)
    rm = rasterize_scene([(core, cSi)], G, background=SiO2)
    # union of pole sets: 3 cSi poles + 1 SiO2 pole
    assert len(rm.poles) == len(cSi.poles) + len(SiO2.poles)
    i = rm.poles.index(cSi.poles[0])
    j = rm.poles.index(SiO2.poles[0])
    assert rm.pole_masks[i][20, 15].item() == pytest.approx(1.0)   # inside core
    assert rm.pole_masks[j][20, 15].item() == pytest.approx(0.0)   # SiO2 absent inside core
    assert rm.pole_masks[j][2, 2].item() == pytest.approx(1.0)     # background cell
    # eps_inf fields are eps_inf blends (both materials have eps_inf=1.0)
    assert rm.eps_tm[20, 15].item() == pytest.approx(1.0)


def test_te_fields_harmonic_on_vertical_edge():
    core = rect(1.0, -1.0, 2.05, 4.0)   # vertical edges at x=1.0 (aligned), x=2.05 (mid-cell)
    rm = rasterize_scene([(core, 12.0)], G, background=2.0)
    h = 1 / (0.5 / 12 + 0.5 / 2)
    a = 0.5 * 12 + 0.5 * 2
    assert rm.eps_te_xx[20, 15].item() == pytest.approx(h, rel=1e-9)
    assert rm.eps_te_yy[20, 15].item() == pytest.approx(a, rel=1e-9)
    assert rm.eps_tm[20, 15].item() == pytest.approx(a, rel=1e-9)


def test_same_material_overlap_pole_mask_is_one():
    """Two overlapping cSi rects over SiO2: the cSi pole mask must be 1.0 in the
    overlap interior, not 2.0. Same-pole painter's compositing (``(1-cov)*old +
    cov``) must not double-count a pole when two shapes carrying it overlap.
    T4 review follow-up."""
    a = rect(0.5, 0.5, 2.5, 2.5)
    b = rect(1.5, 1.5, 3.5, 2.9)   # overlaps a; cell-aligned edges
    rm = rasterize_scene([(a, cSi), (b, cSi)], G, background=SiO2)
    idx = rm.poles.index(cSi.poles[0])
    assert rm.pole_masks[idx][20, 20].item() == pytest.approx(1.0)   # overlap interior
    assert rm.pole_masks[idx][2, 2].item() == pytest.approx(0.0)    # background, cSi absent
