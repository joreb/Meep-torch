# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import pytest

gdstk = pytest.importorskip("gdstk")

from meep_torch.geometry.gds import polygons_from_gds


def test_roundtrip(tmp_path):
    lib = gdstk.Library()
    top = lib.new_cell("TOP")
    top.add(gdstk.rectangle((0, 0), (2.0, 0.4), layer=1))
    top.add(gdstk.rectangle((0, 1.0), (2.0, 1.4), layer=2))
    sub = lib.new_cell("SUB")
    sub.add(gdstk.rectangle((0, 0), (0.5, 0.5), layer=1))
    top.add(gdstk.Reference(sub, (3.0, 0.0)))
    p = tmp_path / "t.gds"
    lib.write_gds(str(p))

    polys = polygons_from_gds(p, layer=1)
    assert len(polys) == 2                       # rect + referenced rect, layer 1 only
    areas = sorted(pp.area for pp in polys)
    assert areas[0] == pytest.approx(0.25)
    assert areas[1] == pytest.approx(0.8)


def test_missing_layer_empty(tmp_path):
    lib = gdstk.Library()
    lib.new_cell("TOP").add(gdstk.rectangle((0, 0), (1, 1), layer=1))
    p = tmp_path / "t.gds"
    lib.write_gds(str(p))
    assert polygons_from_gds(p, layer=99) == []
