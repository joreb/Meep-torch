# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import numpy as np
import pytest

from meep_torch.geometry.polygons import Polygon


def test_area_unit_square():
    p = Polygon(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    assert p.area == pytest.approx(1.0)


def test_rejects_degenerate():
    with pytest.raises(ValueError, match="vertices"):
        Polygon(np.array([[0, 0], [1, 0]], dtype=float))
    with pytest.raises(ValueError, match="area"):
        Polygon(np.array([[0, 0], [1, 0], [2, 0]], dtype=float))


def test_rejects_self_intersection():
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
    with pytest.raises(ValueError, match="self-intersect"):
        Polygon(bowtie)
